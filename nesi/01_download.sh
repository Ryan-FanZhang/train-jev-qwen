#!/usr/bin/env bash
# Step 1 - run on the login node (needs internet; compute jobs run offline):  bash 01_download.sh
# Downloads the base model for $MODEL (config.sh), the distillation corpus and the published 4B adapter
# (start point for MODEL=4b, comparison baseline for every model). Files already present are skipped.
set -euo pipefail
source "$(dirname "$0")/config.sh"
activate_env

echo "MODEL=$MODEL  base=$BASE_REPO@$BASE_REVISION -> $BASE_DIR"
hf download "$BASE_REPO" --revision "$BASE_REVISION" --local-dir "$BASE_DIR"
hf download --repo-type dataset SargeDev/jev-distill-corpus-v3 \
    --revision "$DISTILL_REVISION" --local-dir "$DISTILL_DIR"
hf download shibing624/train-jev-4b-lora --revision "$ADAPTER_4B_REVISION" --local-dir "$ADAPTER_DIR"
if [[ "$MODEL" == 27b ]]; then
    hf download shibing624/train-jev-27b-lora --revision "$ADAPTER_27B_REVISION" \
        --local-dir "$ADAPTER_27B_DIR"
fi

python - "$DISTILL_DIR" <<'EOF'
import sys, os
d = sys.argv[1]
for split in ("train", "validation", "calibration", "test_set_30k", "ood"):
    p = os.path.join(d, f"{split}.jsonl")
    print(f"{split:13s}", sum(1 for _ in open(p)) if os.path.exists(p) else "MISSING")
EOF
du -sh "$BASE_DIR" "$INIT_ADAPTER" "$ADAPTER_DIR" "$DISTILL_DIR" 2>/dev/null || true
