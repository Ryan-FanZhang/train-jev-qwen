"""Step 4 - merge chunks and LLM labels into train-jev `noul` (yes/no) rows, split by group (no leakage).

    python -I data_prep/build_dataset.py work/label work/my-biz

Writes <out>/train.jsonl (80%), calibration.jsonl (10%), test.jsonl (10%) and review_test.csv, a sheet for
hand-labelling the test split (fill `your_label`) to measure real accuracy instead of agreement with the LLM.

The split is a hash of the group id (e.g. a message or product id), so every row of a group lands in the same split and
re-running gives the same split. The labeller's confidence becomes a soft target (knowledge distillation):
a "low" confidence positive trains towards P(true)=0.65 instead of 1.0, which keeps the model calibrated.
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Default question of the example task (see labeling/GUIDELINES.md). Pass --question for your own task.
QUESTION = ("Is this customer review mainly about a delivery or availability problem (late, lost or damaged "
            "delivery, wrong or missing item, out of stock)? Complaints about product quality or price do not count.")
# Labeller confidence -> soft target P(true) for the positive class; the negative class mirrors it.
P_POS = {"high": 0.97, "medium": 0.85, "low": 0.65}


def split_of(group_id, train_pct=80, cal_pct=10):
    h = int(hashlib.md5(group_id.encode()).hexdigest(), 16) % 100
    return "train" if h < train_pct else "calibration" if h < train_pct + cal_pct else "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("label_dir", help="dir with chunks.jsonl and labels/")
    ap.add_argument("out_dir")
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--positive", default="yes", help="label name of the positive class")
    ap.add_argument("--negative", default="no", help="label name of the negative class")
    a = ap.parse_args()

    p_true = {(a.positive, c): p for c, p in P_POS.items()}
    p_true.update({(a.negative, c): round(1 - p, 4) for c, p in P_POS.items()})
    chunks = {}
    for line in open(f"{a.label_dir}/chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c
    labels, bad = {}, collections.Counter()
    for path in sorted(glob.glob(f"{a.label_dir}/labels/labels_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad["bad_json"] += 1
                continue
            key = (r.get("label"), r.get("confidence"))
            if r.get("chunk_id") not in chunks:
                bad["unknown_id"] += 1
            elif key not in p_true:
                bad[f"bad_value {key}"] += 1
            else:
                labels[r["chunk_id"]] = key
    missing = [cid for cid in chunks if cid not in labels]
    print(f"chunks {len(chunks)}  labelled {len(labels)}  missing {len(missing)}  problems {dict(bad)}")

    os.makedirs(a.out_dir, exist_ok=True)
    files = {s: open(f"{a.out_dir}/{s}.jsonl", "w", encoding="utf-8") for s in ("train", "calibration", "test")}
    stats, review = collections.Counter(), []
    for cid, (lab, conf) in labels.items():
        c, p = chunks[cid], p_true[(lab, conf)]
        s = split_of(c["group_id"])
        row = {"id": cid, "kind": "noul", "state": c["text"], "question": a.question,
               "options": ["false", "true"], "target": [round(1 - p, 4), p], "weight": 1.0,
               "label": lab, "confidence": conf}   # the last two are ignored by train-jev
        files[s].write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[(s, lab)] += 1
        if s == "test":
            review.append({"chunk_id": cid, "llm_label": lab, "confidence": conf, "your_label": "", "text": c["text"]})
    for f in files.values():
        f.close()
    with open(f"{a.out_dir}/review_test.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_id", "llm_label", "confidence", "your_label", "text"])
        w.writeheader()
        w.writerows(review)
    for s in ("train", "calibration", "test"):
        print(f"{s:12s} {a.positive} {stats[(s, a.positive)]:5d}  {a.negative} {stats[(s, a.negative)]:5d}")


if __name__ == "__main__":
    main()
