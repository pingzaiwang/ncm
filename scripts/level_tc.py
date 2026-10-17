# Per-level total correlation (TC) of the codes, i.e. the NLL that parallel in-level sampling gives up.
#
# Positions of one level are sampled independently given the coarser levels, so the generator can at best reach
# H(X_k | C) + TC_k with TC_k = sum_j I(X_j; X_<j | C) (chain rule). Two probes sit on the frozen generator states h:
#   marg: logits = head(h + f(h))           -> sum_j H(X_j | C)       (also absorbs any calibration slack of the head)
#   cond: logits = head(h + f(h, x_<j))     -> H(X_k | C)
# both causal over the level, f zero-initialised, identical apart from cond also reading the previous codes. Their
# held-out NLL difference estimates TC_k; each probe only upper-bounds its target entropy, so this is an estimate,
# biased low when the cond probe is too small and noisy when it overfits (watch the standard error).
import argparse, json, math, os, pickle, sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from models.common import Block, LayerNorm, precompute_rope_cos_sin


class Probe(nn.Module):
    def __init__(self, d_model, d_code, width, n_layer, n_head, max_len, read_codes):
        super().__init__()
        self.read_codes = bool(read_codes)
        self.inp = nn.Linear(d_model + d_code, width)
        cfg = SimpleNamespace(n_embd=width, n_head=n_head, bias=True, dropout=0.0, use_qk_norm=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layer)])
        self.ln = LayerNorm(width, bias=True)
        self.out = nn.Linear(width, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        cos, sin = precompute_rope_cos_sin(torch.arange(max_len), width // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, h, prev):
        # prev[:, j] embeds x_{j-1} (zeros at j = 0); the marg probe sees zeros everywhere
        u = self.inp(torch.cat([h, prev if self.read_codes else torch.zeros_like(prev)], dim=-1))
        l = h.shape[1]
        for b in self.blocks:
            u = b(u, rope_cos_sin=(self.cos[:l], self.sin[:l]))  # no mask -> causal
        return h + self.out(self.ln(u))


def seq_nll(head, h, tgt):
    logits = head(h).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape).sum(1)


def load_windows(path, tk):
    # .txt: one document per line; .npy/.pt/.pkl: a 1D token stream, a 2D array or a list of token lists.
    # Each document is cut into context_length windows with the tokenizer's own padding (VQVAETokenizer._adjust_length).
    eos = tk.pre_tokenizer.eos_token_id
    if path.endswith(".txt"):
        docs = [tk.pre_tokenizer.encode(line.rstrip("\n")) + [eos] for line in open(path) if line.strip()]
    else:
        obj = np.load(path) if path.endswith(".npy") else (
            torch.load(path, map_location="cpu", weights_only=False) if path.endswith(".pt") else pickle.load(open(path, "rb")))
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        docs = [obj.tolist()] if isinstance(obj, np.ndarray) and obj.ndim == 1 else [list(map(int, d)) for d in obj]
    wins = [w for d in docs if len(d) for w in tk._adjust_length(list(map(int, d)))]
    return torch.tensor(wins, dtype=torch.long)


def teacher_inputs(net, tk, tok):
    # Teacher-forced generator input, prefix pad mask and target codes, laid out as forward() expects
    # (unconditional: an EOS prefix, as in eval_shard.py).
    vq, P = tk.vqvae.vq, net.prefix_len
    idx = tk.vqvae.encode(tok)
    B, l0 = idx.shape[0], vq.levels[0]
    xt = vq.idx_to_ncm_input(idx)
    if net.query_level0:
        x0 = net.build_l0_ar_input(idx[:, :l0], xt.dtype) if net.l0_ar else net.build_l0_query_input(B, xt.dtype)
        xt, tgt = torch.cat([x0, xt], dim=1), idx
    else:
        tgt = idx[:, l0:]
    prefix = torch.full((B, P), tk.pre_tokenizer.eos_token_id, dtype=torch.long, device=tok.device)
    pm = (prefix == net.prefix_pad_id) if net.prefix_pad_id is not None else None
    return torch.cat([net.embed_prefix(prefix), xt], dim=1), pm, tgt


@torch.no_grad()
def level_states(net, tk, tok):
    # Teacher-forced generator states and target codes, one (h, tgt) pair per head.
    x, pm, tgt = teacher_inputs(net, tk, tok)
    h = net(x, prefix_pad_mask=pm, return_hidden=True)[:, net.prefix_len:]
    out, off = [], 0
    for i, l in enumerate(net.levels[net._lvl_first:]):
        out.append((h[:, off:off + l], tgt[:, off:off + l]))
        off += l
    return out


def shift_codes(emb, tgt):
    e = emb(tgt)
    return torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], dim=1)


def estimate(net, tk, train, held, heads, steps=500, bs=16, lr=1e-3, width=256, n_layer=2, n_head=4, log_every=100,
             seed=0):
    torch.manual_seed(seed)
    dev = next(net.parameters()).device
    vq = tk.vqvae.vq
    d, C = net.config.n_embd, tk.vqvae.embed_dim
    probes = nn.ModuleDict({
        f"{i}_{kind}": Probe(d, C, width, n_layer, n_head, net.levels[i + net._lvl_first], kind == "cond")
        for i in heads for kind in ("marg", "cond")}).to(dev)
    opt = torch.optim.AdamW(probes.parameters(), lr=lr, weight_decay=0.0)
    for step in range(1, steps + 1):
        states = level_states(net, tk, train[torch.randint(0, len(train), (bs,))].to(dev))
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in heads:
            h, tgt = states[i]
            prev = shift_codes(vq.embeddings[i + net._lvl_first], tgt)
            for kind in ("marg", "cond"):
                # per-token mean so every level trains at the same step size; each graph is freed right away
                loss = seq_nll(net.lm_heads[i], probes[f"{i}_{kind}"](h, prev), tgt).mean() / tgt.shape[1]
                loss.backward()
                tot += loss.item()
        opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step}/{steps}  mean probe nll/token {tot / (2 * len(heads)):.4f}", flush=True)

    sums = {i: {"par": [], "marg": [], "cond": []} for i in heads}
    probes.eval()
    with torch.no_grad():
        for s in range(0, len(held), bs):
            states = level_states(net, tk, held[s:s + bs].to(dev))
            for i in heads:
                h, tgt = states[i]
                prev = shift_codes(vq.embeddings[i + net._lvl_first], tgt)
                sums[i]["par"].append(seq_nll(net.lm_heads[i], h, tgt))
                for kind in ("marg", "cond"):
                    sums[i][kind].append(seq_nll(net.lm_heads[i], probes[f"{i}_{kind}"](h, prev), tgt))

    rows = []
    for i in heads:
        par, marg, cond = (torch.cat(sums[i][k]).double() for k in ("par", "marg", "cond"))
        l, n = net.levels[i + net._lvl_first], par.numel()
        tc = marg - cond
        se = float(tc.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
        rows.append({"head": i, "level": i + net._lvl_first, "len": l, "n_seq": n,
                     "nll_par": float(par.mean()), "nll_marg": float(marg.mean()), "nll_cond": float(cond.mean()),
                     "tc": float(tc.mean()), "tc_se": se, "tc_per_tok": float(tc.mean()) / l,
                     "tc_frac": float(tc.mean() / par.mean()) if float(par.mean()) > 0 else float("nan")})
    return rows


def load_model(config, checkpoint_dir="", model_overrides=None):
    # Tokenizer and generator as eval_shard.py builds them; model_overrides (e.g. a cp_head block) are merged into
    # CFG.model, and only the parameters they add may be missing from the checkpoint.
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    CFG = OmegaConf.load(config)
    assert not (CFG.data and CFG.data.get(CFG.eval.get("split", "test"))), \
        "only unconditional tasks are covered; conditional ones would need their source prefixes"
    if model_overrides:
        CFG.model = OmegaConf.merge(CFG.model, model_overrides)
    ck = checkpoint_dir or os.path.join("artifacts", CFG.task)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.to(dev).eval().requires_grad_(False)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
    sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(sd["model"], strict=False)
    assert not unexpected and all(k.startswith("cp_heads.") for k in missing), (missing, unexpected)
    net.eval()
    assert not net.use_ctx_encoder, "needs the z-space prefix path (no ctx_encoder)"
    return CFG, ck, tk, net, sd


def main():
    ap = argparse.ArgumentParser(description="estimate the per-level total correlation that parallel decoding drops")
    ap.add_argument("--config", required=True, help="eval config of an unconditional task, e.g. config/eval/lm1b-small.yaml")
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--train", required=True, help="token data the probes fit (.txt, .npy, .pt or .pkl)")
    ap.add_argument("--held", required=True, help="held-out token data the estimate is reported on")
    ap.add_argument("--levels", default="", help="comma-separated lm_heads indices (default: every head with len > 1)")
    ap.add_argument("--n-train", type=int, default=0, help="cap on training windows (0: all)")
    ap.add_argument("--n-held", type=int, default=512, help="cap on held-out windows (0: all)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write the per-level rows as JSON here")
    a = ap.parse_args()

    CFG, ck, tk, net, _ = load_model(a.config, a.checkpoint_dir)
    net.requires_grad_(False)

    train, held = load_windows(a.train, tk), load_windows(a.held, tk)
    train = train[:a.n_train] if a.n_train else train
    held = held[:a.n_held] if a.n_held else held
    lens = net.levels[net._lvl_first:]
    # a length-1 level has TC 0, and an autoregressive level 0 is not decoded in parallel
    heads = [int(i) for i in a.levels.split(",")] if a.levels else \
        [i for i, l in enumerate(lens) if l > 1 and not (net.l0_ar and i == 0)]
    print(f"[level_tc] {len(train)} train / {len(held)} held-out windows, heads {heads}", flush=True)

    rows = estimate(net, tk, train, held, heads, steps=a.steps, bs=a.bs, lr=a.lr, width=a.width,
                    n_layer=a.n_layer, n_head=a.n_head, seed=a.seed)
    print(f"{'head':>4} {'len':>5} {'nll_par':>10} {'nll_marg':>10} {'nll_cond':>10} {'TC':>9} {'+-se':>7} "
          f"{'TC/tok':>7} {'TC/par':>7}")
    for r in rows:
        print(f"{r['head']:>4} {r['len']:>5} {r['nll_par']:>10.2f} {r['nll_marg']:>10.2f} {r['nll_cond']:>10.2f} "
              f"{r['tc']:>9.2f} {r['tc_se']:>7.2f} {r['tc_per_tok']:>7.4f} {r['tc_frac']:>7.1%}")
    print("nats per sequence; TC = nll_marg - nll_cond; nll_par - nll_marg is head calibration slack, not TC")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
