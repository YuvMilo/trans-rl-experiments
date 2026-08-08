#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

build_one() {
  local op="$1"
  local target="$2"
  local heldout="${3:-$BUNDLE/data/composition/heldout/op${op}-50k.jsonl}"
  local label="$target"
  [[ "$target" == "500000" ]] && label="500k"

  "$PYTHON_BIN" "$BUNDLE/scripts/build_dataset.py" \
    --op "$op" \
    --target "$target" \
    --heldout "$heldout" \
    --output "$BUNDLE/data/composition/heldout/op${op}-${label}.jsonl" \
    --cache-dir "$BUNDLE/cache/composition" \
    --max-shards 2
}

for op in 5 10 13 15 17 18 19; do
  build_one "$op" 137500
done

build_one 20 500000
build_one 20 137500 "$BUNDLE/data/composition/heldout/op20-500k.jsonl"

"$PYTHON_BIN" "$BUNDLE/scripts/validate.py"
echo "[ok] generated datasets ready"
