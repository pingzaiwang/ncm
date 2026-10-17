# Fit CP output heads (models/next_concept.py: _CPMixtureHead) onto a released generator and compare held-out NLL
# per level against the product head it wraps. By default only the CP parameters train, on frozen generator
# states; --train-trunk also fine-tunes the generator. The output directory holds a generator.pt and config.yaml
# that evaluate.py runs as they are:
#   python evaluate.py --config <out>/config.yaml --checkpoint-dir <out>
# The reported gain mixes two things: the CP modulations also recalibrate each position's marginal, which a rank-1
# head (trained g, u, no mixture) does as well. Run --rank 1 as the control; gain(R) - gain(1) is what the mixture
# itself buys, comparable to level_tc.py's TC.
import argparse, json, os, sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from level_tc import load_model, load_windows, teacher_inputs, level_states, seq_nll


@torch.no_grad()
def held_nll(net, tk, held, heads, bs):
    # mean held-out nats per sequence, per head: (product head, cp head)
    was = net.training
    net.eval()
    tot = {i: [0.0, 0.0] for i in heads}
    for s in range(0, len(held), bs):
        states = level_states(net, tk, held[s:s + bs].to(next(net.parameters()).device))
        for i in heads:
            h, tgt = states[i]
            tot[i][0] += float(seq_nll(net.lm_heads[i], h, tgt).sum())
            tot[i][1] -= float(net.cp_heads[str(i)].seq_log_prob(net.lm_heads[i], h, tgt)[0].sum())
    net.train(was)
    return {i: (p / len(held), c / len(held)) for i, (p, c) in tot.items()}


def report(tag, nll, net):
    print(f"[{tag}] {'head':>4} {'len':>5} {'product':>10} {'cp':>10} {'gain':>8}")
    for i, (p, c) in nll.items():
        print(f"[{tag}] {i:>4} {net.levels[i + net._lvl_first]:>5} {p:>10.2f} {c:>10.2f} {p - c:>8.2f}")
    P, C = sum(p for p, _ in nll.values()), sum(c for _, c in nll.values())
    print(f"[{tag}] total {P:.2f} -> {C:.2f} nats/seq (gain {P - C:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser(description="train CP output heads on a released generator")
    ap.add_argument("--config", required=True, help="eval config of an unconditional task")
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--train", required=True, help="token data to fit on (.txt, .npy, .pt or .pkl)")
    ap.add_argument("--held", required=True, help="held-out token data for the NLL report")
    ap.add_argument("--levels", required=True, help="comma-separated lm_heads indices to wrap")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=0)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--balance-weight", type=float, default=0.01)
    ap.add_argument("--train-trunk", action="store_true", help="fine-tune the generator too")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trunk-lr", type=float, default=1e-5)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-held", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    heads = [int(i) for i in a.levels.split(",")]
    cp = {"enabled": True, "rank": a.rank, "block_size": a.block_size, "levels": heads,
          "init_std": a.init_std, "balance_weight": a.balance_weight}
    CFG, ck, tk, net, sd = load_model(a.config, a.checkpoint_dir, {"cp_head": cp})
    dev = next(net.parameters()).device

    train, held = load_windows(a.train, tk), load_windows(a.held, tk)
    train = train[:a.n_train] if a.n_train else train
    held = held[:a.n_held] if a.n_held else held
    print(f"[train_cp] {len(train)} train / {len(held)} held-out windows, heads {heads}, rank {a.rank}, "
          f"block_size {a.block_size}, {'trunk + cp' if a.train_trunk else 'cp only'}", flush=True)

    cp_params = list(net.cp_heads.parameters())
    groups = [{"params": cp_params, "lr": a.lr}]
    net.requires_grad_(False)
    for p in cp_params:
        p.requires_grad_(True)
    if a.train_trunk:
        trunk = [p for n, p in net.named_parameters() if not n.startswith("cp_heads.")]
        for p in trunk:
            p.requires_grad_(True)
        groups.append({"params": trunk, "lr": a.trunk_lr})
        net.train()
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    report("step 0", held_nll(net, tk, held, heads, a.bs), net)
    for step in range(1, a.steps + 1):
        x, pm, tgt = teacher_inputs(net, tk, train[torch.randint(0, len(train), (a.bs,))].to(dev))
        loss, _, bal = net.cp_loss(x, tgt, prefix_pad_mask=pm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"  step {step}/{a.steps}  loss/token {loss.item():.4f}  balance {bal.item():.4f}", flush=True)
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            report(f"step {step}", held_nll(net, tk, held, heads, a.bs), net)
    final = held_nll(net, tk, held, heads, a.bs)

    os.makedirs(a.out_dir, exist_ok=True)
    # keep everything else the checkpoint carries (e.g. the level-0 unigram the eval config points into)
    torch.save({**sd, "model": net.state_dict()}, os.path.join(a.out_dir, "generator.pt"))
    out = OmegaConf.load(a.config)
    out.model = CFG.model
    out.checkpoints.generator = "generator.pt"
    for key in ("tokenizer", "bpe_vocab", "bpe_merges"):
        if out.checkpoints.get(key):
            out.checkpoints[key] = os.path.abspath(os.path.join(ck, out.checkpoints[key]))
    for key in ("level0_unigram", "frequency_allowance"):
        spec = str(out.generation.get(key, "") or "") if out.get("generation") else ""
        path, _, ref = spec.partition(":")
        if path and path != "generator.pt":
            out.generation[key] = os.path.abspath(os.path.join(ck, path)) + (":" + ref if ref else "")
    OmegaConf.save(out, os.path.join(a.out_dir, "config.yaml"))
    json.dump({"rank": a.rank, "block_size": a.block_size, "train_trunk": a.train_trunk,
               "heads": {str(i): {"len": net.levels[i + net._lvl_first], "nll_product": p, "nll_cp": c}
                         for i, (p, c) in final.items()}},
              open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    print(f"[train_cp] wrote {a.out_dir}/generator.pt, config.yaml and metrics.json")


if __name__ == "__main__":
    main()
