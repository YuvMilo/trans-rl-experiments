#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$BUNDLE/.runtime"
REPOSITORY="https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning.git"
REVISION="40b9d29f0ea4f2dfb2f961fe754f4c7202873f3c"

if [[ ! -d "$RUNTIME/.git" ]]; then
  git clone "$REPOSITORY" "$RUNTIME"
fi

git -C "$RUNTIME" fetch origin "$REVISION"
git -C "$RUNTIME" checkout --detach "$REVISION"

if [[ -e "$RUNTIME/new_version" && ! -L "$RUNTIME/new_version" ]]; then
  echo "Cannot create runtime link: $RUNTIME/new_version already exists" >&2
  exit 2
fi
ln -sfn .. "$RUNTIME/new_version"

test -x "$RUNTIME/scripts/meta_run.sh"
test -f "$RUNTIME/verl/dataset.py"
test -d "$RUNTIME/LLaMA-Factory"
echo "[ok] pinned upstream runtime is ready"
