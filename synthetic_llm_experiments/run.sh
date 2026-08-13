#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${RUNTIME_DIR:-$BUNDLE/.runtime}"
EXPERIMENTS=(
  hard-op20
  uniform-op10-17
  uniform-op10-20
)

usage() {
  echo "Usage: ./run.sh EXPERIMENT [--dry-run]"
  printf '  %s\n' "${EXPERIMENTS[@]}"
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
NAME="$1"
shift
if [[ ! " ${EXPERIMENTS[*]} " =~ " ${NAME} " ]]; then
  echo "Unknown experiment: $NAME" >&2
  usage >&2
  exit 2
fi

DRY_RUN=0
DRY_ARG=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  DRY_ARG=(--dry-run)
  shift
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
[[ -x "$RUNTIME/scripts/meta_run.sh" ]] || {
  echo "Runtime missing; run ./bootstrap.sh first" >&2
  exit 2
}

export PYTHONPATH="$RUNTIME:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

CONFIG="$BUNDLE/configs/$NAME.yaml"
EXPERIMENT_NAME="$NAME"
CHECKPOINT_DIR="$BUNDLE/checkpoints/$NAME"
MODEL_DIR="$BUNDLE/models/extrapolation_rl/id2-10_0.2easy_0.3medium_0.5hard/base"
OVERRIDES="actor_rollout_ref.model.path=$MODEL_DIR trainer.experiment_name=$EXPERIMENT_NAME trainer.default_local_dir=$CHECKPOINT_DIR"

if [[ "${RUN_MODE:-}" == "smoke" ]]; then
  EXPERIMENT_NAME="$NAME-smoke"
  CHECKPOINT_DIR="$BUNDLE/smoke/$NAME"
  OVERRIDES="actor_rollout_ref.model.path=$MODEL_DIR trainer.experiment_name=$EXPERIMENT_NAME trainer.default_local_dir=$CHECKPOINT_DIR trainer.total_training_steps=1 trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 trainer.logger=[console]"
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  OVERRIDES="$OVERRIDES $EXTRA_OVERRIDES"
fi

if [[ "$DRY_RUN" -eq 0 && "${ALLOW_EXISTING:-0}" != "1" ]] &&
   [[ -d "$CHECKPOINT_DIR" && -n "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]]; then
  echo "Checkpoint directory is nonempty: $CHECKPOINT_DIR" >&2
  echo "Set ALLOW_EXISTING=1 only to resume intentionally." >&2
  exit 3
fi

mkdir -p "$BUNDLE/logs"
cd "$RUNTIME"

LLAMA_CONFIG="$BUNDLE/configs/pretrain-naming.yaml" \
VERL_CONFIG="$CONFIG" \
VERL_EXTRA_ARGS="$OVERRIDES" \
  "$RUNTIME/scripts/meta_run.sh" --skip-pretrain --skip-eval "${DRY_ARG[@]}" \
  2>&1 | tee "$BUNDLE/logs/$EXPERIMENT_NAME.log"
