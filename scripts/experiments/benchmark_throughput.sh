#!/usr/bin/env bash
# Synthetic workload timing only; measure accuracy on the original public splits.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
THROUGHPUT_DATA="${THROUGHPUT_DATA:-data/SuperShip/throughput4300}"

"$PYTHON_BIN" supership/benchmark_submission.py \
  --image-dir "$THROUGHPUT_DATA/images" \
  --meta-file "$THROUGHPUT_DATA/images_meta.json" \
  --output-file "${OUTPUT_FILE:-output/throughput4300_submission.json}" \
  --runs "${RUNS:-5}" \
  --warmup \
  --init-timeout "${INIT_TIMEOUT:-300}" \
  --run-timeout "${RUN_TIMEOUT:-3600}" \
  -- env PYTHON_BIN="$PYTHON_BIN" bash run.sh --worker
