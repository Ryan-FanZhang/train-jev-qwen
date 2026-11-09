#!/usr/bin/env bash
# Shared settings for every step. Every other script does:  source "$(dirname "$0")/config.sh"

# Your NeSI project code (see `nn_storage_quota`), e.g.  export NESI_PROJECT=nesi12345
export NESI_PROJECT=${NESI_PROJECT:?set NESI_PROJECT to your NeSI project code first}

# Exact upstream revisions used for the published results (see docs/REPRODUCE.md).
export TRAINJEV_COMMIT=88bcc9d21eb4f3ec07da8183bb3f5898abe36bc5
export ADAPTER_4B_REVISION=2ead99963ed0daa38b8b58ac301f156a15b99986      # train-jev-4b-lora v0.4.0
export ADAPTER_27B_REVISION=3f2ef6cd3364a8792d9832d24407706e5e9c3032
export DISTILL_REVISION=75111a13a8455032f0435050bfc7e0c9026ecfe8

# ---- which backbone to train ----
#   0.8b : Qwen/Qwen3.5-0.8B. No released train-jev weights, so it is trained in two stages:
#          distill (jev-distill corpus, from scratch) -> biz (your data, starting from distill's best).
#   2b   : Qwen/Qwen3.5-2B. Same two stages as 0.8b (runs/distill-2b -> runs/my-biz-2b).
#   4b   : Qwen/Qwen3.5-4B. Starts from the released shibing624/train-jev-4b-lora (biz stage only).
#   27b  : Qwen/Qwen3.8-27B. Starts from the released shibing624/train-jev-27b-lora (biz stage only).
#          About 55 GB of bf16 weights per GPU: gradient checkpointing is on by default.
# Override per command, e.g.  MODEL=4b bash submit.sh eval
export MODEL=${MODEL:-2b}

# ---- GPU request. From `sinfo -o "%P %G"` on this cluster:
#   h100:1 / h100:2 -> partition genoa     a100:1..4 -> partition milan     l4:1 -> genoa (24 GB)
export GPU_SPEC=${GPU_SPEC:-h100:1}
case "${GPU_SPEC%%:*}" in
    a100) export PARTITION=${PARTITION:-milan} ;;
    *)    export PARTITION=${PARTITION:-genoa} ;;
esac

# ---- storage layout ----
# /home is small (20 GB): only these scripts.   /nesi/project: code, env, data, checkpoints.
# /nesi/nobackup: base models, distill corpus, caches (auto-cleaned when unused for a long time).
export PROJ_DIR=/nesi/project/$NESI_PROJECT/$USER
export SCRATCH_DIR=/nesi/nobackup/$NESI_PROJECT/$USER
export BIZ_DATA=${BIZ_DATA:-$PROJ_DIR/my-biz}   # train/calibration/test.jsonl from data_prep/
export REPO=$PROJ_DIR/train-jev
export ENV_DIR=$PROJ_DIR/envs/trainjev
export DISTILL_DIR=$SCRATCH_DIR/data/jev-distill-corpus-v3
export RUNS_DIR=$PROJ_DIR/runs
export ADAPTER_DIR=$SCRATCH_DIR/models/train-jev-4b-lora    # published 4B adapter (4B start point, eval baseline)
export ADAPTER_27B_DIR=$SCRATCH_DIR/models/train-jev-27b-lora   # published 27B adapter (27B start point, eval baseline)
export HF_HOME=$SCRATCH_DIR/hf-cache
export PIP_CACHE_DIR=$SCRATCH_DIR/pip-cache
export CONDA_PKGS_DIRS=$SCRATCH_DIR/conda-pkgs   # keep conda's package cache out of the 20 GB /home
export MINIFORGE_MODULE=Miniforge3/25.3.1-0
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1   # train-jev opens jsonl without an explicit encoding; force UTF-8

# Per-model defaults below; these are the values for the small models.
USER_MEM_PER_GPU=${MEM_PER_GPU:-}   # a MEM_PER_GPU set on the command line wins over the per-model value
export DEFAULT_GC_ON=0 DEFAULT_TRAIN_TIME=08:00:00 MEM_PER_GPU=48
case "$MODEL" in
    0.8b)
        export BASE_REPO=Qwen/Qwen3.5-0.8B
        export BASE_REVISION=2fc06364715b967f1860aea9cf38778875588b17
        export BASE_DIR=$SCRATCH_DIR/models/Qwen3.5-0.8B
        export DISTILL_RUN=$RUNS_DIR/distill-0.8b             # stage 1 output
        export INIT_ADAPTER=$DISTILL_RUN/best                 # stage 2 starts here
        export BIZ_RUN_NAME=${BIZ_RUN_NAME:-my-biz-0.8b}
        export DEFAULT_MICRO_TOKENS=16384                     # small model: bigger micro-batches fit on an H100
        ;;
    2b)
        export BASE_REPO=Qwen/Qwen3.5-2B
        export BASE_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc
        export BASE_DIR=$SCRATCH_DIR/models/Qwen3.5-2B
        export DISTILL_RUN=$RUNS_DIR/distill-2b
        export INIT_ADAPTER=$DISTILL_RUN/best
        export BIZ_RUN_NAME=${BIZ_RUN_NAME:-my-biz-2b}
        export DEFAULT_MICRO_TOKENS=8192                      # 4B used 88.6 GB at 8192; 2B needs about 60% of that
        ;;
    4b)
        export BASE_REPO=Qwen/Qwen3.5-4B
        export BASE_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
        export BASE_DIR=$SCRATCH_DIR/models/Qwen3.5-4B
        export DISTILL_RUN=""
        export INIT_ADAPTER=$ADAPTER_DIR
        export BIZ_RUN_NAME=${BIZ_RUN_NAME:-my-biz}
        export DEFAULT_MICRO_TOKENS=8192
        ;;
    27b)
        export BASE_REPO=Qwen/Qwen3.8-27B
        export BASE_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0   # the revision train-jev pins for 27B
        export BASE_DIR=$SCRATCH_DIR/models/Qwen3.8-27B
        export DISTILL_RUN=""
        export INIT_ADAPTER=$ADAPTER_27B_DIR
        export BIZ_RUN_NAME=${BIZ_RUN_NAME:-my-biz-27b}
        export DEFAULT_MICRO_TOKENS=8192    # train-jev's own 27B refinement used 8192
        export DEFAULT_GC_ON=1              # 55 GB of weights per GPU leaves little room for activations
        export DEFAULT_TRAIN_TIME=16:00:00  # estimate ~3.5 h on one H100; re-run the same command to resume
        export MEM_PER_GPU=96               # each rank reads the 55 GB checkpoint through host memory
        ;;
    *) echo "MODEL must be 0.8b, 2b, 4b or 27b (got '$MODEL')" >&2; return 2 2>/dev/null || exit 2 ;;
esac
if [[ -n "$USER_MEM_PER_GPU" ]]; then export MEM_PER_GPU=$USER_MEM_PER_GPU; fi

activate_env() {
    module purge >/dev/null 2>&1 || true
    module load "$MINIFORGE_MODULE"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"
    export PATH="$ENV_DIR/bin:$PATH"   # module purge/load can push the env behind Miniforge's own python
}
