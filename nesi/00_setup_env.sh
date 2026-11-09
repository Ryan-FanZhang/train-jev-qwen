#!/usr/bin/env bash
# Step 0 - run ONCE on the login node (it needs internet):  bash 00_setup_env.sh
# Creates the folders, clones train-jev and builds a Python 3.12 conda env with its requirements.
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p "$PROJ_DIR" "$SCRATCH_DIR" "$RUNS_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$BIZ_DATA" "$PROJ_DIR/envs"

if [[ ! -d "$REPO/.git" ]]; then
    git clone https://github.com/shibing624/train-jev.git "$REPO"
fi
git -C "$REPO" fetch -q origin
git -C "$REPO" checkout -q "$TRAINJEV_COMMIT"
echo "train-jev at $(git -C "$REPO" rev-parse HEAD)" 

module purge >/dev/null 2>&1 || true
module load "$MINIFORGE_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -d "$ENV_DIR" ]]; then
    conda create -y -p "$ENV_DIR" python=3.12
fi
conda activate "$ENV_DIR"

python -m pip install --upgrade pip
# torch wheels from PyPI bundle their own CUDA runtime, so no CUDA module is needed.
python -m pip install -r "$REPO/requirements.txt"
# Pin to the repo's tested stack: the newest torch pulls a triton that flash-linear-attention 0.5.2 cannot load
# ("key contains names that are not kernel arguments: BT"), which silently drops Qwen3.5 to a slow, memory-hungry path.
python -m pip install -r "$REPO/requirements-tested.txt"
python -m pip install -U "huggingface_hub[cli]"

python - <<'EOF'
import sys, torch, transformers, peft
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF
echo "Environment ready at $ENV_DIR"
