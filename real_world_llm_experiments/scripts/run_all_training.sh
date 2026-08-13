#!/usr/bin/env bash
# Run every training configuration for a model family, one after another.
#
# If a run fails, the remaining runs are NOT launched, so a broken environment
# doesn't waste hours of GPU time.
#
# Usage:
#   ./scripts/run_all_training.sh qwen
#   ./scripts/run_all_training.sh llama
#   ./scripts/run_all_training.sh all
#
# Environment overrides are forwarded to scripts/train.sh (NUM_PROCS,
# CUDA_VISIBLE_DEVICES, ACCEL_CFG, HF_TOKEN).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FAMILY="${1:-all}"

QWEN_CONFIGS=(
    configs/qwen-10-uniform.yaml
    configs/qwen-15-uniform.yaml
    configs/qwen-15-hard.yaml
)

LLAMA_CONFIGS=(
    configs/llama-10-uniform.yaml
    configs/llama-15-uniform.yaml
    configs/llama-15-hard.yaml
)

case "${FAMILY}" in
    qwen)  CONFIGS=("${QWEN_CONFIGS[@]}") ;;
    llama) CONFIGS=("${LLAMA_CONFIGS[@]}") ;;
    all)   CONFIGS=("${QWEN_CONFIGS[@]}" "${LLAMA_CONFIGS[@]}") ;;
    *)
        echo "usage: $0 [qwen|llama|all]" >&2
        exit 2
        ;;
esac

echo "[info] ${FAMILY}: ${#CONFIGS[@]} training runs queued"

for cfg in "${CONFIGS[@]}"; do
    echo
    echo "############################################################"
    echo "[queue] starting ${cfg}"
    echo "############################################################"
    ./scripts/train.sh "${cfg}"
done

echo
echo "[done] all ${FAMILY} runs finished cleanly at $(date '+%F %T')"
