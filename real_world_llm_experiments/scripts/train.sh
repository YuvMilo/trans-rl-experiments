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
#   HF_HUB_OFFLINE        set to 0 to let the ranks reach the Hub (default 1)
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

# The conda toolchain ships a newer libstdc++ than the system one. Unless it is
# loaded first, ICU aborts the vLLM import looking for CXXABI_1.3.15.
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

# Pre-download the weights on a single process and resolve the repo to the local
# snapshot it landed in. Without this, all ranks race to populate the HF cache
# simultaneously and the run can die mid-download.
if [[ -d "${MODEL}" ]]; then
    MODEL_PATH="${MODEL}"
    echo "[info] ${MODEL} is a local directory, skipping cache warm"
else
    echo "[info] warming HF cache for ${MODEL}..."
    MODEL_PATH="$(MODEL="${MODEL}" python -c '
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import cached_file
model = os.environ["MODEL"]
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
print(os.path.dirname(cached_file(model, "config.json")))
' | tail -n 1)"
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "[error] could not resolve a local snapshot for ${MODEL}" >&2
        exit 1
    fi
fi

# Hand the ranks the snapshot path rather than the repo id, and keep them off the
# Hub entirely. Each rank otherwise re-resolves the repo on startup, and one
# throttled reply out of eight is enough to kill the run. A path also skips the
# snapshot completeness check vLLM runs when offline, which demands every file in
# the repo -- including Llama's original/*.pth duplicate of the weights.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "============================================================"
echo "[train] $(date '+%F %T')  config=${CONFIG}"
echo "[train] model=${MODEL}"
echo "[train] model_path=${MODEL_PATH}"
echo "[train] gpus=${GPUS}  num_processes=${NUM_PROCS}"
echo "[train] log -> ${LOG}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPUS}" \
accelerate launch \
    --config_file "${ACCEL_CFG}" \
    --num_processes "${NUM_PROCS}" \
    run_r1_grpo_dag_v2.py \
    --config "${CONFIG}" \
    --model_name_or_path "${MODEL_PATH}" 2>&1 | tee "${LOG}"

status="${PIPESTATUS[0]}"
echo "[train] $(date '+%F %T')  config=${CONFIG} exit=${status}"
exit "${status}"
