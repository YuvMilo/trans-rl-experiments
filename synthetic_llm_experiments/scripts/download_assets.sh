#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_BIN="${HF_BIN:-hf}"
DATA_REVISION="a09d5c14c02bfa339143fb00a93274d1a84aa31d"
MODEL_REVISION="4861bd030e6fb92d94be3a1cecab89c2fac4b94a"

includes=()
for op in 5 {10..20}; do
  includes+=("heldout/op${op}-50k.jsonl")
done
for op in {10..20}; do
  includes+=("val/op${op}-200.jsonl")
done

"$HF_BIN" download Interplay-LM-Reasoning/composition \
  --repo-type dataset \
  --revision "$DATA_REVISION" \
  --include "${includes[@]}" \
  --local-dir "$BUNDLE/data/composition"

"$HF_BIN" download Interplay-LM-Reasoning/extrapolation_rl \
  --revision "$MODEL_REVISION" \
  --include "id2-10_0.2easy_0.3medium_0.5hard/base/*" \
  --local-dir "$BUNDLE/models/extrapolation_rl"

echo "[ok] pinned source data and model downloaded"
