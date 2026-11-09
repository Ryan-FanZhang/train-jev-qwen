"""Step 3 - quality checks on the LLM labels before building the dataset.

    python -I data_prep/check_labels.py work/label            # alignment check for every batch
    python -I data_prep/check_labels.py work/label --spot 004 # print 12 random labels of batch 004 to read
    python -I data_prep/check_labels.py work/label --sample 40 # 40 random labels across all batches

Alignment check: a labeller that returns one label per line can silently shift labels onto the neighbouring
chunk. That is invisible in the label counts but shows up as a drop in agreement between each label and a
cheap keyword signal *of the same chunk*, compared with the chunk next to it. A batch is flagged SHIFT? when
the neighbour (offset -1 or +1) agrees better than offset 0. In the original project 3 of 28 batches were
flagged and relabelled; read flagged batches with --spot before trusting or discarding them.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Words that usually signal the positive class of the example task (delivery / availability problems).
# Only a weak signal for the alignment check; pass --keywords for your own task.
KEYWORDS = (r"\b(deliver\w*|arriv\w*|ship\w*|courier|parcel|late|delay\w*|stock|missing|"
            r"tracking|dispatch\w*|wrong item|never came)\b")


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def alignment(label_dir, kw, positive):
    for p in sorted(glob.glob(f"{label_dir}/labels/labels_*.jsonl")):
        b = re.search(r"labels_(\d+)", p).group(1)
        inp, lab = load(f"{label_dir}/batches/batch_{b}.jsonl"), load(p)
        ids_ok = [r["chunk_id"] for r in lab] == [r["chunk_id"] for r in inp]
        sig = [1 if len(kw.findall(r["text"])) >= 2 else 0 for r in inp]
        y = [1 if r.get("label") == positive else 0 for r in lab]

        def agree(off):
            pairs = [(y[i], sig[i + off]) for i in range(len(y)) if 0 <= i + off < len(sig)]
            return sum(a == b for a, b in pairs) / max(1, len(pairs))

        a = {o: agree(o) for o in (-1, 0, 1)}
        flag = "SHIFT?" if max(a[-1], a[1]) > a[0] + 0.03 else ""
        print(f"batch {b} ids_ok={ids_ok} n={len(lab)}/{len(inp)} "
              f"agree -1:{a[-1]:.2f} 0:{a[0]:.2f} +1:{a[1]:.2f} {flag}")


def show(rows, text):
    for r in rows:
        print(f"\n[{r['label']}/{r['confidence']}] {r['chunk_id']}: " + text[r["chunk_id"]][:300].replace("\n", " "))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("label_dir")
    ap.add_argument("--spot", default="", help="batch number to print random labels from")
    ap.add_argument("--sample", type=int, default=0, help="print N random labels across all batches")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--keywords", default=KEYWORDS, help="regex for the weak positive signal")
    ap.add_argument("--positive", default="yes", help="label name of the positive class")
    a = ap.parse_args()

    if a.spot:
        inp = load(f"{a.label_dir}/batches/batch_{a.spot}.jsonl")
        lab = load(f"{a.label_dir}/labels/labels_{a.spot}.jsonl")
        print("labels", len(lab), "inputs", len(inp), "ids match", {r["chunk_id"] for r in lab} == {r["chunk_id"] for r in inp})
        print(collections.Counter((r["label"], r["confidence"]) for r in lab))
        random.seed(int(a.spot))
        show(random.sample(lab, min(a.n, len(lab))), {r["chunk_id"]: r["text"] for r in inp})
    elif a.sample:
        text = {r["chunk_id"]: r["text"] for p in glob.glob(f"{a.label_dir}/batches/*.jsonl") for r in load(p)}
        lab = [r for p in sorted(glob.glob(f"{a.label_dir}/labels/*.jsonl")) for r in load(p)]
        random.seed(42)
        show(random.sample(lab, min(a.sample, len(lab))), text)
    else:
        if not os.path.isdir(f"{a.label_dir}/labels"):
            sys.exit(f"no labels in {a.label_dir}/labels yet")
        alignment(a.label_dir, re.compile(a.keywords, re.I), a.positive)


if __name__ == "__main__":
    main()
