#!/usr/bin/env bash
# Collect the current production inference path without changing runtime code.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source activate_env.sh
NSYS_BIN="${NSYS_BIN:-$ROOT/.local-tools/nsight-systems/opt/nvidia/nsight-systems/2024.6.2/bin/nsys}"
if [[ ! -x "$NSYS_BIN" ]]; then
    echo "Missing local Nsight Systems; see scripts/README.md, or set NSYS_BIN." >&2
    exit 1
fi
PROFILE_DIR="${PROFILE_DIR:-analysis/nsight/$(date +%Y%m%d-%H%M%S)-$$}"
# Refuse reuse so earlier reports cannot be mistaken for a successful capture.
mkdir -p "$PROFILE_DIR"
if [[ -e "$PROFILE_DIR/inference.nsys-rep" || -e "$PROFILE_DIR/inference.sqlite" ]]; then
    echo "Existing report in $PROFILE_DIR; choose a new PROFILE_DIR." >&2
    exit 1
fi
"$NSYS_BIN" --version | tee "$PROFILE_DIR/version.txt"
"$NSYS_BIN" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    -o "$PROFILE_DIR/inference" \
    python supership/generate_submission.py \
    --config configs/dfine/dfine_hgnetv2_s_supership.yml \
    --weights weights/baseline.pth --device "${DEVICE:-cuda:0}" \
    --batch-size "${BATCH_SIZE:-8}" \
    --img-folder data/SuperShip/reference/images \
    --ann-file data/SuperShip/reference/reference_images_meta.json \
    --output "$PROFILE_DIR/submission.json" 2>&1 | tee "$PROFILE_DIR/capture.log"
"$NSYS_BIN" stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    "$PROFILE_DIR/inference.nsys-rep" 2>&1 | tee "$PROFILE_DIR/stats.txt"
python - "$PROFILE_DIR/inference.sqlite" <<'PY'
import sqlite3
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("Nsight did not export a SQLite report")
with sqlite3.connect(path) as db:
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        count = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if exists else 0
        print(f"{table}: {count} records")
        if not count:
            raise SystemExit("Incomplete CUDA capture; do not use this report for performance conclusions")
PY
echo "Report: $PROFILE_DIR/inference.nsys-rep (includes initialization; use benchmark_reference.sh for latency)"
