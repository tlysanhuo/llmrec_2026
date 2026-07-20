#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-${LLMREC_REPO:-}}"

if [[ -z "${REPO_DIR}" ]]; then
  echo "用法: bash 安装脚本.sh /path/to/llmrec_2026-main" >&2
  echo "也可设置 LLMREC_REPO 环境变量。" >&2
  exit 2
fi
REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

if [[ ! -d "${REPO_DIR}/scripts" || ! -d "${REPO_DIR}/configs" ]]; then
  echo "错误: ${REPO_DIR} 不是 llmrec_2026-main 仓库根目录。" >&2
  exit 2
fi

while IFS= read -r -d '' source_file; do
  relative_path="${source_file#${PACKAGE_DIR}/源码快照/}"
  destination="${REPO_DIR}/${relative_path}"
  mkdir -p "$(dirname "${destination}")"
  cp "${source_file}" "${destination}"
done < <(find "${PACKAGE_DIR}/源码快照" -type f -print0)

chmod +x \
  "${REPO_DIR}/scripts/data/build_world_residual_retention_v1.py" \
  "${REPO_DIR}/scripts/train/train_world_residual_retkl.py" \
  "${REPO_DIR}/scripts/train/combine_lora_adapters.py" \
  "${REPO_DIR}/scripts/eval/audit_world_residual_delta.py"

echo "I-19 复现源码已安装到: ${REPO_DIR}"
echo "下一步: bash \"${PACKAGE_DIR}/复现脚本.sh\" \"${REPO_DIR}\" check"
