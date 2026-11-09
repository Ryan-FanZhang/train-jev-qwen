"""Collect every evaluation summary under a runs/ directory into one table.

    python3 scripts/summarize_results.py /nesi/project/$NESI_PROJECT/$USER/runs

Rows: your fine-tuned checkpoints (runs/my-biz*/stepNNNNN/eval-biz) and the published adapters evaluated
without fine-tuning (runs/my-biz*/eval-published-*, eval-baseline). Sorted by top-1, with a 95% interval.
"""
import glob
import json
import math
import sys

NAME = {"my-biz": "4b", "my-biz-0.8b": "0.8b", "my-biz-2b": "2b", "my-biz-27b": "27b"}


def main():
    runs = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "runs"
    rows, seen = [], set()
    for p in sorted(glob.glob(f"{runs}/my-biz*/step*/eval-biz/summary.json")):
        run, step = p[len(runs) + 1:].split("/")[:2]
        rows.append((f"yours-{NAME.get(run, run)} ({step})", p))
    for p in sorted(glob.glob(f"{runs}/my-biz*/eval-*/summary.json")):
        label = p.split("/eval-")[-1].split("/")[0]
        label = {"baseline": "published-4b"}.get(label, label)
        if label.startswith("published") and label not in seen:
            seen.add(label)
            rows.append((label + " (no biz training)", p))
    out = []
    for label, p in rows:
        s = json.load(open(p))["test"]
        m = (s.get("calibrated") or s["raw"])["noul"]
        ci = 1.96 * math.sqrt(m["top1"] * (1 - m["top1"]) / s["n"])
        out.append((label, s["n"], m["top1"], ci, m["auroc"], m["kl"], m["ece"], s["seconds"]))
    print(f"{'model':34s} {'n':>4s} {'top1':>7s} {'±95%':>6s} {'auroc':>7s} {'kl':>7s} {'ece':>7s} {'sec':>6s}")
    for r in sorted(out, key=lambda r: -r[2]):
        print(f"{r[0]:34s} {r[1]:4d} {r[2]:7.4f} {r[3]:6.3f} {r[4]:7.4f} {r[5]:7.4f} {r[6]:7.4f} {r[7]:6.1f}")
    print("top1 = agreement with the LLM labels; sec = time for all test rows on one GPU")


if __name__ == "__main__":
    main()
