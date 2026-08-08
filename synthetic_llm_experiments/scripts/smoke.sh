#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -eq 1 ]] || { echo "Usage: ./scripts/smoke.sh EXPERIMENT" >&2; exit 2; }

if ps -eo cmd | grep -q '[v]erl.trainer.main_ppo'; then
  echo "Another verl job is active on this host; smoke test not started." >&2
  exit 4
fi

RUN_MODE=smoke "$BUNDLE/run.sh" "$1"
