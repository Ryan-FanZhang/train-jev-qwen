#!/bin/bash -e
#SBATCH --job-name=trainjev
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
# Account, GPU, CPUs, memory and time come from submit.sh. Do not sbatch this file directly.
#
# STAGE=distill : LoRA + decision head from the bare Qwen backbone on the jev-distill corpus (train-jev's base
#                 recipe), then fits choice/score/noul temperatures on the corpus calibration split.
# STAGE=biz     : continues from $INIT_ADAPTER on your data mixed with replayed distillation rows.
# Re-submitting the same stage/run resumes automatically from <run>/state.

set -o pipefail
source "$SCRIPTS_DIR/config.sh"
activate_env
cd "$REPO"

STAGE=${STAGE:-biz}
NGPU=${GPU_SPEC##*:}
MAX_LEN=${MAX_LEN:-2048}
MICRO_TOKENS=${MICRO_TOKENS:-$DEFAULT_MICRO_TOKENS}   # lower (8192, 4096) on out-of-memory
GC_ON=${GC_ON:-$DEFAULT_GC_ON}                                   # 1 = gradient checkpointing: less memory, ~30% slower
SEED=${SEED:-17}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

common=(--base "$BASE_DIR" --data "$DISTILL_DIR" --batch 128 --lora_r 16
        --rps_w 0.5 --perm 0.3 --placeholder_w 0.05 --max_len "$MAX_LEN" --micro_tokens "$MICRO_TOKENS"
        --seed "$SEED" --save_state_every 25)
[[ "$GC_ON" == "1" ]] || common+=(--no_gc)

case "$STAGE" in
    distill)
        [[ -n "$DISTILL_RUN" ]] || { echo "MODEL=$MODEL has a released adapter; it needs no distill stage" >&2; exit 2; }
        OUT=${RUN_DIR:-$DISTILL_RUN}
        STEPS=${STEPS:-0}            # 0 = one epoch of the corpus (about 5,123 steps), as in train-jev's recipe
        MAX_STEPS=${MAX_STEPS:-0}
        EVAL_EVERY=${EVAL_EVERY:-500}
        VAL_N=${VAL_N:-4000}
        args=("${common[@]}" --out "$OUT" --lr "${LR:-1e-4}" --head_lr "${HEAD_LR:-2e-4}"
              --steps "$STEPS" --max_steps "$MAX_STEPS" --eval_every "$EVAL_EVERY" --val_n "$VAL_N")
        init_args=()
        ;;
    biz)
        OUT=${RUN_DIR:-$RUNS_DIR/$BIZ_RUN_NAME}
        STEPS=${STEPS:-2048}         # length of the LR schedule
        MAX_STEPS=${MAX_STEPS:-512}  # where to stop (checkpoints step00256, step00512 with EVAL_EVERY=256)
        EVAL_EVERY=${EVAL_EVERY:-256}
        VAL_N=${VAL_N:-2000}
        MIX_FRACTION=${MIX_FRACTION:-0.3}
        # my-biz data is all noul (yes/no): 128*MIX_FRACTION must not exceed the noul count.
        KIND_COUNTS=${KIND_COUNTS:-choice:40,noul:64,score:24}
        args=("${common[@]}" --out "$OUT" --lr "${LR:-2e-5}" --head_lr "${HEAD_LR:-4e-5}"
              --mix "$BIZ_DATA/train.jsonl" --mix_fraction "$MIX_FRACTION" --kind_counts "$KIND_COUNTS"
              --steps "$STEPS" --max_steps "$MAX_STEPS" --eval_every "$EVAL_EVERY" --val_n "$VAL_N")
        if [[ ! -f "$INIT_ADAPTER/head.pt" && ! -f "$INIT_ADAPTER/adapter_config.json" ]]; then
            echo "No start checkpoint at $INIT_ADAPTER - for MODEL=0.8b/2b run 'bash submit.sh distill' first." >&2
            exit 2
        fi
        init_args=(--init_adapter "$INIT_ADAPTER")
        ;;
    *) echo "STAGE must be distill or biz" >&2; exit 2 ;;
esac

if [[ -f "$OUT/state/meta.json" || -f "$OUT/state.old/meta.json" ]]; then
    echo "Resuming from $OUT/state"
    args+=(--resume)
else
    args+=("${init_args[@]}")
fi

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "MODEL=$MODEL STAGE=$STAGE RUN=$OUT BASE=$BASE_DIR STEPS=$STEPS MAX_STEPS=$MAX_STEPS MICRO_TOKENS=$MICRO_TOKENS"
echo "GPUs: $NGPU (global batch stays 128 -> $((128 / NGPU)) rows per GPU per step)"
torchrun --standalone --nproc_per_node "$NGPU" -m trainjev.train "${args[@]}" 2>&1 | tee -a "$OUT/train.log"
ls -d "$OUT"/step* "$OUT"/best 2>/dev/null || true

# Distill stage: fit choice/score/noul temperatures on the corpus calibration split and record general ability.
# Skipped for short smoke runs (MAX_STEPS set) to keep them quick.
if [[ "$STAGE" == "distill" && "$MAX_STEPS" == "0" && -d "$OUT/best" ]]; then
    python -m trainjev.evaluate --base "$BASE_DIR" --adapter "$OUT/best" --data "$DISTILL_DIR" \
        --fit_calibration --splits "test_set_30k:3000,ood:2000" --out "$OUT/eval-distill"
fi
