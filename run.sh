#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIPELINE="${PIPELINE:-fast}"
if [[ "$PIPELINE" == "fast" ]]; then
  source "$ROOT/configs/fastest.env"
fi

# Large host allocations otherwise leave hundreds of thousands of 4 KiB pages
# for exit_mmap to release. Ask glibc for transparent huge pages before Python
# starts; no machine-wide THP settings or reserved HugeTLB pages are changed.
# Keep an explicit caller allocator setting, and allow HOST_THP=0 for comparison.
if [[ "${HOST_THP:-1}" == "1" && "$OSTYPE" == linux* ]]; then
  if [[ ":${GLIBC_TUNABLES:-}:" != *:glibc.malloc.hugetlb=* ]]; then
    export GLIBC_TUNABLES="${GLIBC_TUNABLES:+${GLIBC_TUNABLES}:}glibc.malloc.hugetlb=1"
  fi
fi

if [[ "${PIPELINE:-standard}" == "fast" ]]; then
  exec "$PYTHON_BIN" supership/fast_worker.py "$@"
fi

exec "$PYTHON_BIN" supership/generate_submission.py \
  --worker \
  --config "${MODEL_CONFIG:-configs/dfine/dfine_hgnetv2_s_supership.yml}" \
  --weights "${MODEL_WEIGHTS:-weights/baseline.pth}" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-8}"
