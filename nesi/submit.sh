#!/usr/bin/env bash
# Submit a Slurm job with the account / GPU from config.sh (#SBATCH lines cannot read variables).
#   bash submit.sh smoke      # 5-step test of the next stage to run (distill for 0.8b/2b until it has finished)
#   bash submit.sh distill    # 0.8b/2b stage 1: base distillation on the jev-distill corpus
#   bash submit.sh train      # stage 2: your data (re-run the same command to resume after a timeout)
#   bash submit.sh eval       # calibrate + evaluate the newest biz checkpoint against the other models
#   MODEL=27b bash submit.sh train             # any setting from config.sh / 03_train.sl can be overridden like this
#   GPU_SPEC=h100:2 bash submit.sh distill  # 2 GPUs on one node (resume with the same GPU count)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
export SCRIPTS_DIR=$PWD
mkdir -p logs

next_stage=biz
if [[ -n "$DISTILL_RUN" && ! -d "$DISTILL_RUN/best" ]]; then
    next_stage=distill
fi

case "${1:-}" in
    smoke)   job=03_train.sl; export STAGE=${STAGE:-$next_stage} MAX_STEPS=${MAX_STEPS:-5} EVAL_EVERY=${EVAL_EVERY:-5} \
                 VAL_N=${VAL_N:-200} RUN_DIR=${RUN_DIR:-$RUNS_DIR/smoke-$MODEL}
             extra=(--time=${TIME:-00:45:00}) ;;
    distill) job=03_train.sl; export STAGE=distill; extra=(--time=${TIME:-10:00:00}) ;;
    train)   job=03_train.sl; export STAGE=biz;     extra=(--time=${TIME:-$DEFAULT_TRAIN_TIME}) ;;
    eval)    job=04_eval.sl;  extra=(--time=${TIME:-02:00:00}); GPU_SPEC=${GPU_SPEC%%:*}:1 ;;
    *) echo "usage: bash submit.sh smoke|distill|train|eval" >&2; exit 2 ;;
esac

if [[ -n "${STAGE:-}" && "$STAGE" == biz && -n "$DISTILL_RUN" && ! -d "$INIT_ADAPTER" ]]; then
    echo "MODEL=$MODEL: no distill checkpoint at $INIT_ADAPTER yet - run 'bash submit.sh distill' first." >&2
    exit 2
fi

NGPU=${GPU_SPEC##*:}   # h100:2 -> 2 GPUs on one node, one torchrun process each (DDP)
echo "MODEL=$MODEL stage=${STAGE:-eval} account=$NESI_PROJECT partition=$PARTITION gpu=$GPU_SPEC"
sbatch --account="$NESI_PROJECT" --partition="$PARTITION" --gpus-per-node="$GPU_SPEC" \
    --cpus-per-task=$((8 * NGPU)) --mem=$((MEM_PER_GPU * NGPU + 48))G --export=ALL "${extra[@]}" "$job"
echo "Watch with:  squeue --me    and    tail -f logs/<jobname>-<jobid>.out"
