#!/usr/bin/env bash
# 00_install.sh — 复现官方 baseline 环境（改编自 docs/demo_baseline/scripts/00_install.sh）
# ★2026-07-03 起环境一律放个人卷(卷根 AGENTS.md 守则),不放 overlay。
# 在 lustre 建 py3.11 venv，pin torch 2.7.1+cu126 / flash-attn 2.7.4.post1 / liger 0.8.0。
set -euo pipefail

V=/lustre/prod_glm_volumes/volume-20260201002229-o7c51
mountpoint -q "$V" || { echo "ERROR: personal volume not mounted"; exit 1; }
REPRO=${REPRO:-$V/ai_runtime/llmrec_2026}
LF_DIR=$REPRO/LLaMA-Factory
VENV=$LF_DIR/.venv
export UV_CACHE_DIR=${UV_CACHE_DIR:-$REPRO/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$V/envs/uv_pythons}
export UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export UV_HTTP_TIMEOUT=300

mkdir -p "$REPRO"
command -v uv >/dev/null 2>&1 || pip install uv

# 1) clone LLaMA-Factory (官方 baseline 用 0.9.6.dev0)
if [ -d "$LF_DIR/.git" ]; then echo "[skip] cloned"; else
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LF_DIR"
fi

# 2) py3.11 venv (uv 自动拉取 py3.11)
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"

# 3) LLaMA-Factory + metrics
uv pip install -e "$LF_DIR"
uv pip install -r "$LF_DIR/requirements/metrics.txt"

# 4) pin torch 2.7.1+cu126（官方脚本一致）
SP="$VENV/lib/python3.11/site-packages"
uv pip uninstall torch torchvision torchaudio sympy 2>/dev/null || true
rm -rf "$SP/torch" "$SP/sympy" "$SP/functorch"
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126
uv pip install --force-reinstall --no-deps "sympy==1.13.3"

# 5) liger + flash-attn 官方 wheel
uv pip install --no-deps "liger-kernel==0.8.0"
uv pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
uv pip install tensorboard

# 6) patch transformers flash_attention (s_aux None guard)
FA_PY="$SP/transformers/integrations/flash_attention.py"
if ! grep -q "s_aux=s_aux.to(query.dtype) if s_aux is not None else None" "$FA_PY"; then
  sed -i 's|s_aux=s_aux.to(query.dtype),|s_aux=s_aux.to(query.dtype) if s_aux is not None else None,|' "$FA_PY"
  echo "[ok] patched flash_attention.py"
fi

# 7) verify
python - <<'PY'
import torch, flash_attn, transformers, sympy
from importlib.metadata import version
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("flash_attn:", flash_attn.__version__, "| transformers:", transformers.__version__,
      "| liger:", version("liger-kernel"), "| sympy:", sympy.__version__)
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
print("liger qwen3 hook: OK")
PY
echo "[ok] install finished"
