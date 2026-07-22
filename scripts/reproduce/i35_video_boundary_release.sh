#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
release=$root/assets/derived/releases/i35_r96_video_boundary_retkl_r16_v1
cd "$root"

usage() {
  cat >&2 <<'EOF'
usage:
  scripts/reproduce/i35_video_boundary_release.sh verify-data
  scripts/reproduce/i35_video_boundary_release.sh restore-pool
  scripts/reproduce/i35_video_boundary_release.sh restore-retention
  scripts/reproduce/i35_video_boundary_release.sh restore-original-data
  scripts/reproduce/i35_video_boundary_release.sh self-test

restore-pool is the correct starting point for a different parent adapter.
The original formal data and sidecar are locked to the published I-35 parent.
EOF
  exit 2
}

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

verify_archive() {
  local archive=$1 archive_hash=$2 payload_hash=$3 actual
  [[ -f "$archive" ]] || { echo "missing archive: $archive" >&2; exit 2; }
  actual=$(sha256 "$archive")
  [[ "$actual" == "$archive_hash" ]] || {
    echo "archive checksum mismatch: $archive ($actual/$archive_hash)" >&2
    exit 2
  }
  actual=$(gzip -dc "$archive" | sha256sum | awk '{print $1}')
  [[ "$actual" == "$payload_hash" ]] || {
    echo "payload checksum mismatch: $archive ($actual/$payload_hash)" >&2
    exit 2
  }
}

verify_data() {
  verify_archive \
    "$release/data_i35_video_boundary_retkl_v1.jsonl.gz" \
    14109b240c02c459554843899251eb6f5cdbdf2c50291307a0348b26836aa60e \
    9c044e47d26fb7644281107a548249e49564e0f203a04795337c6a90c0927100
  verify_archive \
    "$release/data_i35_video_boundary_retkl_v1_sidecar.jsonl.gz" \
    2d3dc43de81aa6a9248b383b2b8cfddcbd10ff6c56dcabb54baff7bdab4426dd \
    366a5323350c982f73d4bcf1fbd0b809d412e844871ac98aa526cbafb31ae083
  verify_archive \
    "$release/i35_video_material_beam128_pool_v1.jsonl.gz" \
    bbe5c461674470affabac591951d86ae2fdd89116bef951a894763b6bbaaa950 \
    36a7fc7fc2319711acd6443987295356ca37402bac09563ff01cb46fb2c66aa6
  verify_archive \
    "$release/i35_video_material_beam128_pool_v1_dev.jsonl.gz" \
    95410355a16fa0fab09eb6ab7e5fec3cf0b507380ed15487daac89f618e347d3 \
    e70e9ad07d723dfa4e3b406620c6d0ae68907942165aa78fd8e55f4e04ca18a5
  verify_archive \
    "$release/data_i33_r96_material_desc2sid_retkl_v1.jsonl.gz" \
    7895cb9e124c86ba4104240cc16af8c60936f4190fe54644b6634da091310213 \
    7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd
  echo "I-35 release data verification: PASS"
}

restore_one() {
  local archive=$1 target=$2 expected=$3 actual temporary
  if [[ -f "$target" ]]; then
    actual=$(sha256 "$target")
    [[ "$actual" == "$expected" ]] || {
      echo "refusing to replace drifted target: $target ($actual/$expected)" >&2
      exit 2
    }
    echo "already restored: $target"
    return
  fi
  mkdir -p "$(dirname "$target")"
  temporary=${target}.tmp.$$
  trap 'rm -f "$temporary"' RETURN
  gzip -dc "$archive" > "$temporary"
  actual=$(sha256 "$temporary")
  [[ "$actual" == "$expected" ]] || {
    echo "restored payload checksum mismatch: $target ($actual/$expected)" >&2
    exit 2
  }
  mv "$temporary" "$target"
  trap - RETURN
  echo "restored: $target"
}

restore_pool() {
  verify_data
  restore_one \
    "$release/i35_video_material_beam128_pool_v1.jsonl.gz" \
    "$root/logs/data/i35_video_material_beam128_pool_v1.jsonl" \
    36a7fc7fc2319711acd6443987295356ca37402bac09563ff01cb46fb2c66aa6
  restore_one \
    "$release/i35_video_material_beam128_pool_v1_dev.jsonl.gz" \
    "$root/logs/data/i35_video_material_beam128_pool_v1_dev.jsonl" \
    e70e9ad07d723dfa4e3b406620c6d0ae68907942165aa78fd8e55f4e04ca18a5
}

restore_retention() {
  verify_data
  restore_one \
    "$release/data_i33_r96_material_desc2sid_retkl_v1.jsonl.gz" \
    "$root/assets/derived/processed/data_i33_r96_material_desc2sid_retkl_v1.jsonl" \
    7d6a1e4a44238a79dcb0d31384f147c02baea95cd870224e2a6815444f8470fd
}

restore_original_data() {
  restore_pool
  restore_retention
  restore_one \
    "$release/data_i35_video_boundary_retkl_v1.jsonl.gz" \
    "$root/assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl" \
    9c044e47d26fb7644281107a548249e49564e0f203a04795337c6a90c0927100
  restore_one \
    "$release/data_i35_video_boundary_retkl_v1_sidecar.jsonl.gz" \
    "$root/assets/derived/processed/data_i35_video_boundary_retkl_v1_sidecar.jsonl" \
    366a5323350c982f73d4bcf1fbd0b809d412e844871ac98aa526cbafb31ae083
}

self_test() {
  local python_bin=${LLAMAFACTORY_PYTHON:-python3}
  "$python_bin" scripts/data/build_i35_video_material_pool_v1.py --self-test
  "$python_bin" scripts/eval/generate_i35_video_material_beam128_v1.py --self-test
  "$python_bin" scripts/data/build_i35_video_boundary_retkl_v1.py --self-test
  "$python_bin" scripts/train/train_i35_video_boundary_retkl.py --self-test
  "$python_bin" scripts/train/combine_lora_adapters.py --self-test
}

command=${1:-}
case "$command" in
  verify-data)
    [[ $# -eq 1 ]] || usage
    verify_data
    ;;
  restore-pool)
    [[ $# -eq 1 ]] || usage
    restore_pool
    ;;
  restore-retention)
    [[ $# -eq 1 ]] || usage
    restore_retention
    ;;
  restore-original-data)
    [[ $# -eq 1 ]] || usage
    restore_original_data
    ;;
  self-test)
    [[ $# -eq 1 ]] || usage
    self_test
    ;;
  *)
    usage
    ;;
esac
