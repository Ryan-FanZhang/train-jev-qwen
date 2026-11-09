"""Step 2 - turn a labelled CSV / JSONL into train-jev rows and split it into train / calibration / test.

Input: one example per row with a text column and a label column, e.g. tickets.csv:
    text,label
    快递晚到了两周,物流
    卡被扣了两次款,财务

Choice question (options default to the distinct labels, sorted):
    python 02_convert_data.py tickets.csv --out $BIZ_DATA --kind choice \
        --question "这条工单应该交给哪个团队？" --options 售后 物流 财务

Yes/no question (label must be one of 1/0, true/false, yes/no, 是/否):
    python 02_convert_data.py urgent.csv --out $BIZ_DATA --kind noul --question "是否需要人工紧急介入？"

Several questions -> run it once per question with --append, then all rows end up in the same files.
Writes <out>/train.jsonl (80%), calibration.jsonl (10%), test.jsonl (10%).
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

TRUE = {"1", "true", "yes", "y", "是", "对", "需要"}
FALSE = {"0", "false", "no", "n", "否", "不", "不需要"}


def read_rows(path, text_col, label_col):
    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".json"):
        rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    else:
        rows = list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))
    out = []
    for r in rows:
        text, label = str(r.get(text_col, "")).strip(), str(r.get(label_col, "")).strip()
        if text and label:
            out.append((text, label))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("--out", required=True, help="directory for train/calibration/test.jsonl")
    ap.add_argument("--kind", choices=["choice", "noul"], default="choice")
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", nargs="+", default=None, help="choice options (default: distinct labels)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="label smoothing, e.g. 0.05 -> target 0.95 on the gold option (helps calibration)")
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="add to existing files instead of overwriting")
    a = ap.parse_args()

    data = read_rows(a.input, a.text_col, a.label_col)
    if a.kind == "choice":
        options = a.options or sorted({lab for _, lab in data})
        if not 2 <= len(options) <= 16:
            raise SystemExit(f"choice needs 2-16 options, got {len(options)}: {options}")
    else:
        options = ["false", "true"]

    items, skipped = [], Counter()
    for i, (text, label) in enumerate(data):
        if a.kind == "choice":
            if label not in options:
                skipped[label] += 1
                continue
            gold = options.index(label)
        else:
            low = label.lower()
            if low not in TRUE | FALSE:
                skipped[label] += 1
                continue
            gold = 1 if low in TRUE else 0
        n = len(options)
        target = [a.smooth / (n - 1) if j != gold else 1 - a.smooth for j in range(n)] if a.smooth else \
                 [1.0 if j == gold else 0.0 for j in range(n)]
        items.append({"id": f"{Path(a.input).stem}-{a.kind}-{i}", "kind": a.kind, "state": text,
                      "question": a.question, "options": options, "target": target, "weight": 1.0})

    random.Random(a.seed).shuffle(items)
    fr = [float(x) for x in a.split.split(",")]
    n_tr, n_cal = int(len(items) * fr[0]), int(len(items) * fr[1])
    parts = {"train": items[:n_tr], "calibration": items[n_tr:n_tr + n_cal], "test": items[n_tr + n_cal:]}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        with (out / f"{name}.jsonl").open("a" if a.append else "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"options: {options}")
    print({k: len(v) for k, v in parts.items()}, "label counts:",
          dict(Counter(options[r["target"].index(max(r["target"]))] for r in items)))
    if skipped:
        print(f"skipped {sum(skipped.values())} rows with unknown labels: {dict(skipped.most_common(10))}")


if __name__ == "__main__":
    main()
