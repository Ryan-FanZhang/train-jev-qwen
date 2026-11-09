"""Step 1 - clean raw text chunks, drop exact duplicates, and cut them into labelling batches.

Input: a CSV with one text per row: an id, a group id (rows of one group always land in the same split,
e.g. a message id or a product id) and the text. Column names are configurable.

    python -I data_prep/prepare_chunks.py raw.csv work/label --batch_size 250

Writes
    work/label/chunks.jsonl            one cleaned, de-duplicated chunk per line
    work/label/batches/batch_000.jsonl {"chunk_id", "text"} per line, the unit sent to the labeller

The group column is kept so that build_dataset.py can split by group: rows of the same group never end up
in both train and test.
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
csv.field_size_limit(10**9)

URL = re.compile(r"\((https?://[^)\s]+)\)|https?://\S+")


def short_url(m):
    """Replace a (often very long, tracking) URL by its domain; keeps the hint, drops the noise."""
    u = m.group(1) or m.group(0)
    dom = re.sub(r"^https?://", "", u).split("/")[0]
    return f"({dom})" if m.group(1) else dom


def clean(t):
    t = URL.sub(short_url, t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--group_col", default="group_id", help="rows sharing this value stay in one split")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--min_chars", type=int, default=20)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig", newline="")))
    seen, chunks, dropped = {}, [], collections.Counter()
    for r in rows:
        text = clean(r[a.text_col])
        if len(text) < a.min_chars:
            dropped["too_short"] += 1
            continue
        key = re.sub(r"\s+", " ", text.lower())
        if key in seen:   # repeated boilerplate: keep the first copy only
            dropped["duplicate"] += 1
            seen[key]["dup_ids"].append(r[a.id_col])
            continue
        c = {"chunk_id": r[a.id_col], "group_id": r[a.group_col], "text": text, "dup_ids": []}
        seen[key] = c
        chunks.append(c)

    os.makedirs(f"{a.out_dir}/batches", exist_ok=True)
    with open(f"{a.out_dir}/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    nb = 0
    for i in range(0, len(chunks), a.batch_size):
        with open(f"{a.out_dir}/batches/batch_{nb:03d}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks[i:i + a.batch_size]:
                f.write(json.dumps({"chunk_id": c["chunk_id"], "text": c["text"]}, ensure_ascii=False) + "\n")
        nb += 1
    print(f"rows {len(rows)}  kept {len(chunks)}  dropped {dict(dropped)}  batches {nb}  "
          f"groups {len({c['group_id'] for c in chunks})}")


if __name__ == "__main__":
    main()
