"""Step 2 - label every batch with Claude (the "teacher" for distillation).

    pip install anthropic
    export ANTHROPIC_API_KEY=...          # or `ant auth login`
    python -I data_prep/label_with_claude.py work/label --guidelines labeling/GUIDELINES.md

For each work/label/batches/batch_XXX.jsonl this writes work/label/labels/labels_XXX.jsonl with one
{"chunk_id", "label", "confidence"} per input chunk, in the same order. Batches that already have a complete
labels file are skipped, so the command can be re-run after an interruption.

Note: the labels behind the published results were produced with Claude Haiku 5.5 running as Claude Code
sub-agents, with the same output format and a private guideline. This script is the equivalent API implementation;
LLM labels are not bit-for-bit reproducible, so expect small differences.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from typing import List, Literal

import anthropic
from pydantic import BaseModel, create_model

sys.stdout.reconfigure(encoding="utf-8")


def make_schema(positive, negative):
    """Structured-output schema: one {chunk_id, label, confidence} per chunk, labels restricted to the two classes."""
    item = create_model("Item", chunk_id=(str, ...), label=(Literal[positive, negative], ...),
                        confidence=(Literal["high", "medium", "low"], ...))
    return create_model("Labels", items=(List[item], ...))


def build_user_message(chunks):
    parts = [f"Label each of the {len(chunks)} chunks below. Return exactly one item per chunk, "
             "in the same order, copying chunk_id exactly."]
    for c in chunks:
        parts.append(f'<chunk id="{c["chunk_id"]}">\n{c["text"]}\n</chunk>')
    return "\n\n".join(parts)


def label_group(client, model, system, schema, chunks, max_retries=3):
    """One request per group of chunks; returns a list of dicts aligned with `chunks`, or None."""
    want = [c["chunk_id"] for c in chunks]
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.messages.parse(
                model=model,
                max_tokens=16000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": build_user_message(chunks)}],
                output_format=schema,
            )
        except anthropic.RateLimitError as e:
            wait = int(e.response.headers.get("retry-after", "30"))
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            print(f"  transient error ({type(e).__name__}), retry {attempt}/{max_retries}")
            time.sleep(5 * attempt)
            continue
        if resp.stop_reason == "refusal":
            print(f"  refusal for chunks {want[0]}..{want[-1]}; label these by hand")
            return None
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            print(f"  incomplete output ({resp.stop_reason}), retry {attempt}/{max_retries}")
            continue
        got = [i.chunk_id for i in resp.parsed_output.items]
        if got != want:   # the failure mode seen in practice: labels shifted onto neighbouring chunks
            print(f"  chunk ids do not line up with the input, retry {attempt}/{max_retries}")
            continue
        return [i.model_dump() for i in resp.parsed_output.items]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("label_dir", help="output dir of prepare_chunks.py")
    ap.add_argument("--guidelines", required=True, help="markdown labelling guideline (the system prompt)")
    ap.add_argument("--positive", default="yes", help="label name of the positive class")
    ap.add_argument("--negative", default="no", help="label name of the negative class")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--per_request", type=int, default=25, help="chunks per API request")
    ap.add_argument("--only", default="", help="regex on batch number, e.g. '00[4-9]'")
    a = ap.parse_args()

    system = open(a.guidelines, encoding="utf-8").read()
    schema = make_schema(a.positive, a.negative)
    client = anthropic.Anthropic()
    os.makedirs(f"{a.label_dir}/labels", exist_ok=True)
    for path in sorted(glob.glob(f"{a.label_dir}/batches/batch_*.jsonl")):
        b = re.search(r"batch_(\d+)", path).group(1)
        if a.only and not re.fullmatch(a.only, b):
            continue
        chunks = [json.loads(line) for line in open(path, encoding="utf-8")]
        out = f"{a.label_dir}/labels/labels_{b}.jsonl"
        if os.path.exists(out) and sum(1 for line in open(out, encoding="utf-8") if line.strip()) == len(chunks):
            print(f"batch {b}: done, skipping")
            continue
        rows, failed = [], 0
        for i in range(0, len(chunks), a.per_request):
            group = chunks[i:i + a.per_request]
            res = label_group(client, a.model, system, schema, group)
            if res is None:
                failed += len(group)
                continue
            rows += res
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"batch {b}: {len(rows)}/{len(chunks)} labelled" + (f", {failed} FAILED" if failed else ""))


if __name__ == "__main__":
    main()
