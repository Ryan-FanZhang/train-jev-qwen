#!/bin/bash -e
#SBATCH --job-name=trainjev-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
# Account, GPU and time come from submit.sh. Do not sbatch this file directly.
#
# 1. Fits the noul temperature on $BIZ_DATA/calibration.jsonl and keeps the start point's choice/score temperatures.
# 2. Scores the checkpoint on $BIZ_DATA/test.jsonl next to the baselines (published 4B, and your 0.8B / 2B / 4B biz
#    models when they exist), so the size/quality trade-off is visible in one table.
# 3. Runs two example decisions.

source "$SCRIPTS_DIR/config.sh"
activate_env
cd "$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# Default: the newest stepNNNNN checkpoint (best/ is chosen by KL on the *distill* validation set, not your data).
CKPT=${CKPT:-$(ls -d "$RUNS_DIR/$BIZ_RUN_NAME"/step* | sort | tail -1)}
TEST_N=${TEST_N:-100000}
echo "MODEL=$MODEL  checkpoint: $CKPT"

python -m trainjev.evaluate --base "$BASE_DIR" --adapter "$CKPT" --data "$BIZ_DATA" \
    --fit_calibration --splits "test:$TEST_N" --out "$CKPT/eval-biz"

# Business data is noul-only, so the fit above writes only a noul temperature. Keep the start point's
# choice/score temperatures so those question types stay calibrated too.
python - "$INIT_ADAPTER/calibration.json" "$CKPT/calibration.json" <<'EOF'
import json, os, sys
src, dst = sys.argv[1:]
new = json.load(open(dst))
if os.path.exists(src):
    base = json.load(open(src))["per_kind"]
    new["per_kind"] = dict(base, **new["per_kind"])
    new["kept_from_start_point"] = sorted(set(base) - {"noul"})
    json.dump(new, open(dst, "w"), indent=1)
print("calibration.json per_kind:", new["per_kind"])
EOF

# ---- baselines: label|base dir|adapter dir ----
FOUR_B=$SCRATCH_DIR/models/Qwen3.5-4B
baselines=("published-4b|$FOUR_B|$ADAPTER_DIR")
# Published 27B, only when it is downloaded (MODEL=27b): the "use it as-is, no business training" reference.
if [[ -d "$ADAPTER_27B_DIR" ]]; then
    baselines+=("published-27b|$SCRATCH_DIR/models/Qwen3.8-27B|$ADAPTER_27B_DIR")
fi
# Your own biz models of the other sizes, when they exist (run name|base dir|label).
for spec in "my-biz-27b|$SCRATCH_DIR/models/Qwen3.8-27B|your-27b" \
            "my-biz|$FOUR_B|your-4b" \
            "my-biz-2b|$SCRATCH_DIR/models/Qwen3.5-2B|your-2b" \
            "my-biz-0.8b|$SCRATCH_DIR/models/Qwen3.5-0.8B|your-0.8b"; do
    IFS='|' read -r run base label <<< "$spec"
    if [[ "$run" == "$BIZ_RUN_NAME" ]]; then continue; fi
    ck=$(ls -d "$RUNS_DIR/$run"/step* 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$ck" ]]; then baselines+=("$label|$base|$ck"); fi
done
summaries=("$MODEL-yours|$CKPT/eval-biz/summary.json")
for spec in "${baselines[@]}"; do
    IFS='|' read -r label base adapter <<< "$spec"
    out=$RUNS_DIR/$BIZ_RUN_NAME/eval-$label
    if [[ -d "$base" && -d "$adapter" ]]; then
        python -m trainjev.evaluate --base "$base" --adapter "$adapter" --data "$BIZ_DATA" \
            --splits "test:$TEST_N" --out "$out"
        summaries+=("$label|$out/summary.json")
    else
        echo "skip baseline $label (missing $base or $adapter)"
    fi
done

python - "${summaries[@]}" <<'EOF'
import json, sys
print(f"\n{'model':16s} {'n':>5s} {'top1':>7s} {'auroc':>7s} {'kl':>7s} {'ece':>7s} {'sec':>7s}")
for spec in sys.argv[1:]:
    label, path = spec.split("|", 1)
    s = json.load(open(path))["test"]
    m = (s.get("calibrated") or s["raw"])["noul"]
    print(f"{label:16s} {s['n']:5d} {m['top1']:7.4f} {m['auroc']:7.4f} {m['kl']:7.4f} {m['ece']:7.4f} {s['seconds']:7.1f}")
print("(test labels come from the LLM labeller: top1 = agreement with it, not human-verified accuracy)")
EOF

# Demo: the same yes/no question the model was trained on, on two invented texts.
python - "$BASE_DIR" "$CKPT" "$BIZ_DATA/test.jsonl" <<'EOF'
import json, sys, torch
from trainjev.data import encode
from trainjev.modeling import last_hidden, load_model, load_temperatures
base, ckpt, test = sys.argv[1:]
tok, body, head = load_model(base, ckpt, dtype=torch.bfloat16, device="cuda")
temps = load_temperatures(f"{ckpt}/calibration.json")
question = json.loads(open(test, encoding="utf-8").readline())["question"]
for state in ("Paid two weeks ago and the parcel still has not arrived. Tracking has not moved since Monday.",
              "Great sound for the price, but the ear cushions started peeling after a month."):
    row = {"kind": "noul", "state": state, "question": question, "options": ["false", "true"]}
    with torch.inference_mode():
        h = last_hidden(body, [encode(tok, row)], tok.pad_token_id or 0, "cuda")[0]
        p = torch.softmax(head.logits(h, "noul", 2) / temps.get("noul", 1.0), -1).tolist()
    print(f"P(true)={p[1]:.3f}  {state}")
EOF
