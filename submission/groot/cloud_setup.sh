#!/usr/bin/env bash
# One-time setup on a rented Linux x86_64 GPU machine (>= 40 GB VRAM for
# fine-tuning, CUDA 12.6+, FFmpeg 4-7 for torchcodec). Run from the repo root
# after copying the repo (branch groot) and datasets/groot/<name> to the machine:
#   bash submission/groot/cloud_setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
GROOT_DIR=${GROOT_DIR:-$HOME/Isaac-GR00T}
GROOT_COMMIT=3b49067943eb1ab2eb2203b93d461c4390c78da3  # Isaac-GR00T main, 2026-11-10 (N1.7)

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v ffmpeg > /dev/null || { sudo apt-get update && sudo apt-get install -y ffmpeg; }
ffmpeg -version | head -1

# Evaluator environment (pinned by this repo's uv.lock).
uv sync --locked --extra video

# GR00T environment, pinned to the commit the converter/config were written against.
if [ ! -d "$GROOT_DIR" ]; then
  git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_DIR"
fi
git -C "$GROOT_DIR" fetch --quiet origin "$GROOT_COMMIT" || true
git -C "$GROOT_DIR" checkout --quiet "$GROOT_COMMIT"
(cd "$GROOT_DIR" && uv sync --locked)

# Base checkpoint (~6 GB) into the Hugging Face cache.
(cd "$GROOT_DIR" && uv run hf download nvidia/GR00T-N1.7-3B)

"$GROOT_DIR/.venv/bin/python" -c "import torch, gr00t, torchcodec; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "setup done; GR00T python: $GROOT_DIR/.venv/bin/python"
