#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -gt 0 ]]; then
  submissions=("$1")
else
  submissions=()
  if [[ -f output/reference_submission.json ]]; then
    submissions+=(output/reference_submission.json)
  fi
  shopt -s nullglob
  for submission in output/reference_submission.run*.json; do
    submissions+=("$submission")
  done
fi

if [[ ${#submissions[@]} -eq 0 ]]; then
  echo "No reference submission found under output/." >&2
  exit 1
fi

for submission in "${submissions[@]}"; do
  echo "==> $submission"
  "$PYTHON_BIN" supership/score_submission.py \
    --gt data/SuperShip/reference/annotations/instances_reference.json \
    --submission "$submission"
done
