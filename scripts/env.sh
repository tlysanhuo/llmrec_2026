#!/usr/bin/env bash
# Activate the LLM-Rec 2026 environment.
# Usage:  source scripts/env.sh
# The venv inherits the container's system torch 2.3.0a + flash-attn 2.4.2 (ABI-correct),
# and layers the HuggingFace training stack (transformers 4.53.0, etc.) on the persistent volume.

export PERSONAL_VOLUME_ROOT=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
if ! mountpoint -q "$PERSONAL_VOLUME_ROOT"; then
  echo "ERROR: personal volume is not mounted at $PERSONAL_VOLUME_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT="$PERSONAL_VOLUME_ROOT/llmrec_2026"
export PROJECT_RUNTIME_ROOT="$PERSONAL_VOLUME_ROOT/ai_runtime/llmrec_2026"
export LLMREC_VENV="$PERSONAL_VOLUME_ROOT/envs/llmrec"

# Caches / scratch on the persistent volume (container / is ephemeral overlay).
export TMPDIR="$PROJECT_RUNTIME_ROOT/tmp"
export HF_HOME="$PROJECT_RUNTIME_ROOT/cache/hf"
export PIP_CACHE_DIR="$PROJECT_RUNTIME_ROOT/cache/pip"
export WANDB_DIR="$PROJECT_RUNTIME_ROOT/wandb"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR"

# Activate venv (adds venv/bin to PATH; python resolves to the venv interpreter).
source "$LLMREC_VENV/bin/activate"

echo "llmrec env ready: python=$(python -c 'import sys;print(sys.version.split()[0])') \
torch=$(python -c 'import torch;print(torch.__version__)') \
transformers=$(python -c 'import transformers;print(transformers.__version__)')"
