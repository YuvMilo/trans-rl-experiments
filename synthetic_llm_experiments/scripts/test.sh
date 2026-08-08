#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${RUNTIME_DIR:-$BUNDLE/.runtime}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPERIMENTS=(
  hard-op20
  uniform-op10-15
  uniform-op10-17
  uniform-op10-20
  uniform-op10-15-20
  uniform-op10-13-17-20
  uniform-op17-20
  uniform-op18-20
  uniform-op5-10-15-20
)

bash -n "$BUNDLE/bootstrap.sh" "$BUNDLE/run.sh" "$BUNDLE/scripts/"*.sh
"$PYTHON_BIN" -m py_compile "$BUNDLE/scripts/"*.py
RUNTIME_DIR="$RUNTIME" "$PYTHON_BIN" "$BUNDLE/scripts/validate.py" "$@"

for experiment in "${EXPERIMENTS[@]}"; do
  "$BUNDLE/run.sh" "$experiment" --dry-run
done

echo "[ok] source, configs, assets, and launchers"
