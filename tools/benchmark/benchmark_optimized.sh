#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Use the same verified fastest defaults as the main worker entrypoint.
source "$ROOT/configs/fastest.env"
export RUNS="${RUNS:-3}"
exec bash "$ROOT/tools/benchmark/benchmark_reference.sh" "$@"
