# One command for the CP-head study: estimate per-level TC (level_tc.py), keep the levels whose TC is significant,
# fit CP heads over a rank x block_size grid on them (train_cp.py) plus the rank-1 control, and tabulate what the
# mixture buys against the TC it could at most recover. Every step writes into --out-dir and is skipped when its
# output already exists, so an interrupted sweep resumes where it stopped.
import argparse, json, math, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, args)]
    print("[cp_sweep] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="TC estimate -> CP grid -> summary")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--train", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=3, help="number of highest-TC levels to wrap")
    ap.add_argument("--levels", default="", help="wrap these lm_heads indices instead of picking by TC")
    ap.add_argument("--ranks", default="2,4")
    ap.add_argument("--block-sizes", default="0,16")
    ap.add_argument("--tc-steps", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--n-held", type=int, default=512)
    ap.add_argument("--evaluate", action="store_true", help="also run evaluate.py on every fitted model (needs GPUs)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    common = ["--config", a.config, "--train", a.train, "--held", a.held, "--bs", a.bs,
              "--n-train", a.n_train, "--n-held", a.n_held] + (["--checkpoint-dir", a.checkpoint_dir] if a.checkpoint_dir else [])

    tc_path = os.path.join(a.out_dir, "tc.json")
    if not os.path.exists(tc_path):
        run("level_tc.py", common + ["--steps", a.tc_steps, "--out", tc_path])
    tc = {r["head"]: r for r in json.load(open(tc_path))}
    if a.levels:
        heads = [int(i) for i in a.levels.split(",")]
    else:
        # significant TC only: a level whose estimate is within noise has nothing for a mixture to recover
        sig = sorted((r for r in tc.values() if r["tc"] > 2 * r["tc_se"]), key=lambda r: -r["tc"])
        heads = sorted(r["head"] for r in sig[:a.top_k])
    assert heads, "no level has TC above twice its standard error; nothing to fit"
    tc_sum = sum(tc[i]["tc"] for i in heads)
    tc_se = math.sqrt(sum(tc[i]["tc_se"] ** 2 for i in heads))
    print(f"[cp_sweep] heads {heads}, TC {tc_sum:.2f} +- {tc_se:.2f} nats/seq", flush=True)

    # rank 1 has no mixture, so its block size is irrelevant and it runs once
    grid = [(1, 0)] + [(int(r), int(b)) for r in a.ranks.split(",") for b in a.block_sizes.split(",") if int(r) > 1]
    rows = []
    for rank, block in grid:
        d = os.path.join(a.out_dir, f"r{rank}_b{block}")
        if not os.path.exists(os.path.join(d, "metrics.json")):
            run("train_cp.py", common + ["--levels", ",".join(map(str, heads)), "--rank", rank, "--block-size", block,
                                         "--steps", a.steps, "--eval-every", 0, "--out-dir", d])
        m = json.load(open(os.path.join(d, "metrics.json")))
        gain = sum(v["nll_product"] - v["nll_cp"] for v in m["heads"].values())
        rows.append({"rank": rank, "block_size": block, "dir": d, "gain": gain,
                     "per_head": {k: v["nll_product"] - v["nll_cp"] for k, v in m["heads"].items()}})
        if a.evaluate:
            subprocess.run([sys.executable, os.path.join(HERE, "..", "evaluate.py"), "--config",
                            os.path.join(d, "config.yaml"), "--checkpoint-dir", d, "--out", os.path.join(d, "eval")],
                           check=True)

    base = rows[0]["gain"]
    print(f"\n{'rank':>4} {'block':>5} {'gain':>8} {'mixture':>8} {'/TC':>6}   per head (gain)")
    for r in rows:
        r["mixture_gain"] = r["gain"] - base
        # a ratio to a TC within noise of 0 means nothing
        r["mixture_over_tc"] = r["mixture_gain"] / tc_sum if tc_sum > 2 * tc_se else None
        flag = "  > TC: check for a bug or held-out leakage" if r["mixture_gain"] > tc_sum + 2 * tc_se else ""
        heads_s = " ".join(f"{k}:{v:.2f}" for k, v in r["per_head"].items())
        print(f"{r['rank']:>4} {r['block_size']:>5} {r['gain']:>8.2f} {r['mixture_gain']:>8.2f} "
              f"{'-' if r['mixture_over_tc'] is None else format(r['mixture_over_tc'], '.1%'):>6}   {heads_s}{flag}")
    print(f"nats/seq on held-out data; mixture = gain - gain(rank 1 control); TC = {tc_sum:.2f} +- {tc_se:.2f}")
    json.dump({"heads": heads, "tc": tc_sum, "tc_se": tc_se, "rows": rows},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
