#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARCHIVE_NAME="SuperShip-public-data.tar.zst"
EXPECTED_SHA256="4f70f1167ee51090e30f5f3c4915f821b2c960d307610902ad666ae92ea69b71"
SOURCE="${1:-${SUPERSHIP_DATA_URL:-}}"

if [[ -z "$SOURCE" ]]; then
  cat >&2 <<'EOF'
Usage:
  bash download_data.sh /path/to/SuperShip-public-data.tar.zst
  SUPERSHIP_DATA_URL='<THU_CLOUD_DIRECT_DOWNLOAD_URL>' bash download_data.sh
EOF
  exit 2
fi

if [[ "$SOURCE" =~ ^https?:// ]]; then
  command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
  mkdir -p .cache
  ARCHIVE=".cache/$ARCHIVE_NAME"
  curl --fail --location --continue-at - --output "$ARCHIVE" "$SOURCE"
else
  ARCHIVE="$SOURCE"
fi

[[ -f "$ARCHIVE" ]] || { echo "Archive not found: $ARCHIVE" >&2; exit 1; }

if command -v sha256sum >/dev/null; then
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
else
  ACTUAL_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
fi

if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "SHA-256 mismatch for $ARCHIVE" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 1
fi

command -v zstd >/dev/null || {
  echo "zstd is required to extract $ARCHIVE_NAME" >&2
  exit 1
}

# macOS-created archives may contain AppleDouble sidecars, not dataset images.
tar --use-compress-program='zstd -d' --exclude='._*' \
  --warning=no-unknown-keyword -xf "$ARCHIVE" -C "$ROOT"
"$PYTHON_BIN" supership/validate_release.py
echo "SuperShip public data installed under data/SuperShip/."
