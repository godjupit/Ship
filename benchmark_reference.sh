#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p output

# 精度对比需要参考集标注，交给独立脚本执行，评分不进入推理计时。
case "${COMPARE_ACCURACY:-True}" in
  True|true|1|yes)
    exec "$PYTHON_BIN" -m supership.compare_reference
    ;;
esac

"$PYTHON_BIN" supership/benchmark_submission.py \
  --image-dir data/SuperShip/reference/images \
  --meta-file data/SuperShip/reference/reference_images_meta.json \
  --output-file "${OUTPUT_FILE:-output/reference_submission.json}" \
  --runs "${RUNS:-3}" \
  --init-timeout "${INIT_TIMEOUT:-300}" \
  --run-timeout "${RUN_TIMEOUT:-3600}" \
  -- env PYTHON_BIN="$PYTHON_BIN" bash run.sh --worker
