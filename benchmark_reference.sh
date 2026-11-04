#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p output

# 默认只增加每轮 AP50 行，评分在该轮计时结束后执行。
accuracy_args=()
case "${COMPARE_ACCURACY:-True}" in
  True|true|1|yes)
    accuracy_args=(--accuracy-gt data/SuperShip/reference/annotations/instances_reference.json)
    ;;
esac

"$PYTHON_BIN" supership/benchmark_submission.py \
  --image-dir data/SuperShip/reference/images \
  --meta-file data/SuperShip/reference/reference_images_meta.json \
  --output-file "${OUTPUT_FILE:-output/reference_submission.json}" \
  --runs "${RUNS:-3}" \
  "${accuracy_args[@]}" \
  --init-timeout "${INIT_TIMEOUT:-300}" \
  --run-timeout "${RUN_TIMEOUT:-3600}" \
  -- env PYTHON_BIN="$PYTHON_BIN" bash run.sh --worker
