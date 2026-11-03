#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" supership/generate_submission.py \
  --worker \
  --config configs/dfine/dfine_hgnetv2_s_supership.yml \
  --weights weights/baseline.pth \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-8}"
