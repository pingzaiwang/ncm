# CPU checks for the CP output head and the TC probe on a tiny randomly initialised model; no checkpoints needed.
# Run with `python tests/test_cp_head.py` or pytest.
import copy, importlib.util, math, os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer, _CPMixtureHead

LV, V = [1, 4, 9, 16], 32


def tiny(cp_head=None):
    cfg = OmegaConf.load(os.path.join(ROOT, "config/eval/owt.yaml"))
    vc = cfg.tokenizer.vqvae
    vc.vqvae_vocab_size = [V] * len(LV)
    vc.embed_dim, vc.context_length, vc.n_layers = 16, LV[-1], 2
    vc.vector_quantizer_config.levels.manual_levels = LV
    kw = {k: v for k, v in OmegaConf.to_container(vc).items() if k not in ("checkpoint_path", "vector_quantizer_config")}
    torch.manual_seed(0)
    vq = VQVAE(**kw, vector_quantizer_config=vc.vector_quantizer_config).eval()
    m = cfg.model
    m.n_layer, m.n_head, m.n_embd = 2, 2, 32
    m.vocab_size, m.context_length = [V] * (len(LV) - 1), 1 + sum(LV)
    base = None
    for c in (None, cp_head):
        mm = copy.deepcopy(m)
        if c is not None:
            mm.cp_head = c
        torch.manual_seed(1)
        net = NextConceptTransformer(vq, mm).eval()
        if base is None:
            base = net
        else:
            missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
            assert not unexpected and all(k.startswith("cp_heads.") for k in missing)
    return vq, base, (net if cp_head is not None else None)


def inputs(B=3):
    torch.manual_seed(2)
    T = sum(LV[1:])
    return torch.randn(B, 1 + T, 16), torch.randint(0, V, (B, T))


def test_rank1_is_the_base_head():
    vq, base, r1 = tiny({"enabled": True, "rank": 1})
    x, tgt = inputs()
    ref = F.cross_entropy(base(x).reshape(-1, V), tgt.reshape(-1), reduction="none").view(tgt.shape).sum(1)
    assert torch.allclose(r1.cp_loss(x, tgt)[1], ref, atol=1e-4)
    assert torch.allclose(base.cp_loss(x, tgt)[1], ref, atol=1e-4)
    idx0, pfx = torch.randint(0, V, (3, 1)), torch.randint(0, 50, (3, 1))
    torch.manual_seed(5); a = base.generate(idx0, pfx, top_k=0, top_p=1.0)
    torch.manual_seed(5); b = r1.generate(idx0, pfx, top_k=0, top_p=1.0)
    assert torch.equal(a, b)


def test_block_nll_matches_brute_force():
    R, bs = 3, 3
    _, _, net = tiny({"enabled": True, "rank": R, "levels": [1, 2], "block_size": bs})
    torch.manual_seed(7)
    for p in net.cp_heads.parameters():
        nn.init.normal_(p, std=0.3)
    x, tgt = inputs()
    nll = net.cp_loss(x, tgt)[1]
    h = net(x, return_hidden=True)[:, 1:]
    brute, off = torch.zeros(tgt.shape[0]), 0
    for i, l in enumerate(LV[1:]):
        hi, ti = h[:, off:off + l], tgt[:, off:off + l]
        if str(i) in net.cp_heads:
            cp = net.cp_heads[str(i)]
            for s in range(0, l, bs):  # 16 = 5 * 3 + 1: the last block is short
                hb, tb = hi[:, s:s + bs], ti[:, s:s + bs]
                comp = torch.stack([torch.log_softmax(net.lm_heads[i](hb * (1 + cp.g[r]) + cp.u[r]), -1)
                                    .gather(-1, tb[..., None]).squeeze(-1).sum(1) for r in range(R)], 1)
                brute += torch.logsumexp(torch.log_softmax(cp.gate(hb.mean(1)), -1) + comp, 1)
        else:
            brute += torch.log_softmax(net.lm_heads[i](hi), -1).gather(-1, ti[..., None]).squeeze(-1).sum(1)
        off += l
    assert torch.allclose(nll, -brute, atol=1e-4)
    # each position's logits use its own block's component, and every decoding path runs
    cp, hh = net.cp_heads["2"], torch.randn(2, 16, 32)
    r = cp.sample_component(hh, 1.0)
    lg = cp.component_logits(net.lm_heads[2], hh, r)
    for b in range(2):
        for j in range(16):
            ref = net.lm_heads[2](hh[b, j] * (1 + cp.g[r[b, j // bs]]) + cp.u[r[b, j // bs]])
            assert torch.allclose(lg[b, j], ref, atol=1e-5)
    idx0, pfx = torch.randint(0, V, (3, 1)), torch.randint(0, 50, (3, 1))
    with torch.no_grad():
        for kw in ({}, {"use_kv_cache": True}, {"cfg_scale": 1.0}, {"temperature": 0}):
            assert net.generate(idx0, pfx, top_k=0, **kw).shape == (3, sum(LV) - LV[0] + 1)


def _fit_planted(kind, R, block_size, steps=400, l=16, d=32):
    # h is constant, so whatever the head gains over the product of marginals comes from the mixture alone
    torch.manual_seed(0)
    head, cp = nn.Linear(d, V), _CPMixtureHead(d, R, init_std=0.5, block_size=block_size)
    opt = torch.optim.Adam(list(head.parameters()) + list(cp.parameters()), lr=1e-2)

    def data(n):
        if kind == "copy":  # x_j = x_0 for the whole level
            return torch.randint(0, V, (n, 1)).expand(n, l).clone()
        return torch.randint(0, V, (n, l // 4)).repeat_interleave(4, dim=1)  # each block of 4 copies its head

    h = torch.ones(1, l, d)
    for _ in range(steps):
        lp, log_pi = cp.seq_log_prob(head, h.expand(256, -1, -1), data(256))
        loss = -lp.mean() / l + 0.01 * cp.balance_loss(log_pi)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return l * math.log(V) + cp.seq_log_prob(head, h.expand(2048, -1, -1), data(2048))[0].mean().item()


def test_mixture_learns_planted_dependence():
    tc_copy, tc_block = 15 * math.log(V), 12 * math.log(V)
    assert abs(_fit_planted("copy", 1, 0)) < 0.5                    # rank 1 cannot see in-level dependence
    assert _fit_planted("copy", 32, 0) > 0.7 * tc_copy              # one component per value recovers most of TC
    blocks, whole = _fit_planted("block4", 32, 4), _fit_planted("block4", 32, 0)
    assert blocks > 0.7 * tc_block and blocks > 2 * whole           # local structure wants per-block components


def test_level_tc_recovers_planted_tc():
    spec = importlib.util.spec_from_file_location("level_tc", os.path.join(ROOT, "scripts/level_tc.py"))
    lt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lt)
    vq, base, _ = tiny()
    tk = type("TK", (), {"vqvae": vq, "pre_tokenizer": type("PT", (), {"eos_token_id": 50256})()})()
    net = base.requires_grad_(False)

    def planted(kind):
        def states(net_, tk_, tok):
            B = tok.shape[0]
            tgt = torch.randint(0, V, (B, 1)).expand(B, 16).clone() if kind == "copy" else torch.randint(0, V, (B, 16))
            return [None, None, (torch.zeros(B, 16, 32), tgt)]
        return states

    real = lt.level_states
    try:
        for kind, truth in (("copy", 15 * math.log(V)), ("indep", 0.0)):
            lt.level_states = planted(kind)
            r = lt.estimate(net, tk, torch.zeros(4096, 1, dtype=torch.long), torch.zeros(256, 1, dtype=torch.long), [2],
                            steps=400, bs=64, lr=3e-3, width=64, n_head=2, log_every=0)[0]
            assert abs(r["tc"] - truth) < max(1.0, 4 * r["tc_se"]), (kind, r)
    finally:
        lt.level_states = real


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
