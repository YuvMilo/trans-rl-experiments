#!/usr/bin/env bash
# Evaluate every checkpoint of a model family against every test distribution.
#
# Matrix (4 checkpoints x 4 distributions):
#   Checkpoints:   5-uniform, 10-uniform, 15-uniform, 15-hard
#   Distributions: 5-uniform, 10-uniform, 15-uniform, 15-hard
#
# By convention:
#   n-uniform => min_ancestor_depth=1, max_ancestor_depth=n
#   n-hard    => min_ancestor_depth=n, max_ancestor_depth=n
#
# The dataset flags below must stay identical to the ones in configs/*.yaml,
# otherwise the test distribution won't match the training distribution.
#
# Usage:
#   ./scripts/run_eval_matrix.sh qwen
#   ./scripts/run_eval_matrix.sh llama
#   N_SAMPLES=1000 BATCH_SIZE=16 ./scripts/run_eval_matrix.sh qwen
#   SAVE_SAMPLES=1 ./scripts/run_eval_matrix.sh qwen
#
# Environment overrides:
#   RUNS_DIR         where checkpoints live       (default runs)
#   N_SAMPLES        examples per cell            (default 1000)
#   BATCH_SIZE       generation batch size        (default 32)
#   MAX_NEW_TOKENS   completion cap               (default 2048)
#   SEED             dataset seed                 (default 111)
#   SAVE_SAMPLES     1 => dump per-sample outputs (default 0)
#   BASE_OUTPUT_DIR  results root                 (default evaluation_results/<family>_matrix_<ts>)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FAMILY="${1:-}"
case "${FAMILY}" in
    qwen|llama) ;;
    *)
        echo "usage: $0 [qwen|llama]" >&2
        exit 2
        ;;
esac

if [[ ! -f "evaluate_simple.py" ]]; then
    echo "[error] evaluate_simple.py not found in ${REPO_ROOT}" >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs/${FAMILY}_matrix_eval_${TS}"
mkdir -p "${LOG_DIR}"

RUNS_DIR="${RUNS_DIR:-runs}"
N_SAMPLES="${N_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-111}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-evaluation_results/${FAMILY}_matrix_${TS}}"
mkdir -p "${BASE_OUTPUT_DIR}"

# Checkpoints, evaluated in this order. Paths follow the output_dir convention
# in configs/*.yaml: runs/<config-name>.
MODEL_KEYS=(
    "15_hard"
    "10_uniform"
    "15_uniform"
    "5_uniform"
)

# Test distributions. Format: "<dist_name>:<n_nodes>:<mode>", mode in {uniform, hard}.
EVAL_DISTS=(
    "15_hard:15:hard"
    "10_uniform:10:uniform"
    "15_uniform:15:uniform"
    "5_uniform:5:uniform"
)

model_path_for() {
    # 5_uniform -> runs/<family>-5-uniform
    local key="$1"
    echo "${RUNS_DIR}/${FAMILY}-${key//_/-}"
}

run_one_eval() {
    local model_key="$1"
    local model_path="$2"
    local dist_name="$3"
    local n_nodes="$4"
    local mode="$5"

    local min_depth max_depth
    if [[ "${mode}" == "hard" ]]; then
        min_depth="${n_nodes}"
        max_depth="${n_nodes}"
    else
        min_depth="1"
        max_depth="${n_nodes}"
    fi

    local out_dir="${BASE_OUTPUT_DIR}/${model_key}__on__${dist_name}"
    local log_file="${LOG_DIR}/${model_key}__on__${dist_name}.log"
    mkdir -p "${out_dir}"

    echo "============================================================"
    echo "[matrix] $(date '+%F %T') model=${FAMILY}/${model_key} dist=${dist_name}"
    echo "[matrix] model_path=${model_path}"
    echo "[matrix] n_nodes=${n_nodes} mode=${mode} min_depth=${min_depth} max_depth=${max_depth}"
    echo "[matrix] output_dir=${out_dir}"
    echo "[matrix] log=${log_file}"
    echo "============================================================"

    local -a cmd=(
        python evaluate_simple.py
        --model_path "${model_path}"
        --n_samples "${N_SAMPLES}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --max_prompt_length "${MAX_PROMPT_LENGTH}"
        --output_dir "${out_dir}"
        --seed "${SEED}"
        --graph_type dag
        --min_nodes "${n_nodes}" --max_nodes "${n_nodes}"
        --max_in_degree 2 --edge_probability 0.2
        --max_degree 1 --max_coefficient 1 --max_terms 2
        --probabilistic_pairwise_terms
        --single_variable_term_probability 1.0
        --pairwise_product_term_probability 0.0
        --constant_term_probability 0.6666
        --min_constant 0 --max_constant 1
        --target_sink_only
        --max_value 5000 --modulus 2
        --min_ancestor_depth "${min_depth}" --max_ancestor_depth "${max_depth}"
        --min_start_value 0 --max_start_value 1
        --max_label_value "${n_nodes}" --no_fixed_label_set
        --sampling_strategy uniform_size
    )

    if [[ "${SAVE_SAMPLES}" == "1" ]]; then
        cmd+=(--save_samples)
    fi

    "${cmd[@]}" 2>&1 | tee "${log_file}"
    local status="${PIPESTATUS[0]}"
    echo "[matrix] finished model=${model_key} dist=${dist_name} exit=${status}"
    return "${status}"
}

for model_key in "${MODEL_KEYS[@]}"; do
    model_path="$(model_path_for "${model_key}")"
    if [[ ! -d "${model_path}" ]]; then
        echo "[warn] skipping ${model_key}: missing directory ${model_path}" >&2
        continue
    fi

    for dist_spec in "${EVAL_DISTS[@]}"; do
        IFS=":" read -r dist_name n_nodes mode <<< "${dist_spec}"
        run_one_eval "${model_key}" "${model_path}" "${dist_name}" "${n_nodes}" "${mode}"
    done
done

echo
echo "[done] ${FAMILY} matrix evaluation complete"
echo "[done] logs: ${LOG_DIR}"
echo "[done] outputs: ${BASE_OUTPUT_DIR}"
