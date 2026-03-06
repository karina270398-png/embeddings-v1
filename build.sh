#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
VENV="$ROOT_DIR/.venv/bin/python"

CSV_INPUT=${1:-"/Users/datanomica/Downloads/data-20260301T1303-structure-20180402T1704.csv"}
OUT_DIR=${2:-"$ROOT_DIR/data"}

$VENV -m pip install -r "$ROOT_DIR/requirements.txt" >/dev/null 2>&1 || true

exec "$VENV" "$ROOT_DIR/src/build_index.py" --csv "$CSV_INPUT" --out "$OUT_DIR" --batch 64
