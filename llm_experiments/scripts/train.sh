#!/usr/bin/env bash
# Launch a single GRPO training run with colocated vLLM.
#
# All 8 GPUs train; each rank spawns its own vLLM worker on its own GPU, so no
# separate vLLM server is needed.
#
# Usage:
#   ./scripts/train.sh configs/qwen-15-uniform.yaml
#
# Environment overrides:
#   NUM_PROCS             number of training processes   (default 8)
#   CUDA_VISIBLE_DEVICES  GPUs to use                    (default 0-7)
#   ACCEL_CFG             accelerate config              (default deepspeed_zero3)
#   HF_TOKEN              required for gated models (Llama)
#
# The run log is written to logs/<config-name>-<timestamp>.log.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <config.yaml>" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="$1"
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
ACCEL_CFG="${ACCEL_CFG:-accelerate_configs/deepspeed_zero3.yaml}"
NUM_PROCS="${NUM_PROCS:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

for f in "${CONFIG}" "${ACCEL_CFG}" run_r1_grpo_dag_v2.py; do
    if [[ ! -f "${f}" ]]; then
        echo "[error] required file missing: ${f}" >&2
        exit 1
    fi
done

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${CONFIG_NAME}-$(date +%Y%m%d_%H%M%S).log"

MODEL="$(sed -n 's/^model_name_or_path:[[:space:]]*//p' "${CONFIG}" | head -n 1)"

# Pre-download the weights on a single process. Without this, all ranks race to
# populate the HF cache simultaneously and the run can die mid-download.
echo "[info] warming HF cache for ${MODEL}..."
MODEL="${MODEL}" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
model = os.environ["MODEL"]
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
print(f"[info] cache warmed for {model}")
PY

echo "============================================================"
echo "[train] $(date '+%F %T')  config=${CONFIG}"
echo "[train] model=${MODEL}"
echo "[train] gpus=${GPUS}  num_processes=${NUM_PROCS}"
echo "[train] log -> ${LOG}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPUS}" \
accelerate launch \
    --config_file "${ACCEL_CFG}" \
    --num_processes "${NUM_PROCS}" \
    run_r1_grpo_dag_v2.py \
    --config "${CONFIG}" 2>&1 | tee "${LOG}"

status="${PIPESTATUS[0]}"
echo "[train] $(date '+%F %T')  config=${CONFIG} exit=${status}"
exit "${status}"
