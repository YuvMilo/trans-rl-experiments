#!/usr/bin/env bash
# Plot training-time accuracy on complex examples for two runs, side by side.
#
# This produces the "training on simple examples helps solve complex examples"
# figure: accuracy on maximal-depth examples throughout training, for a model
# trained on 15-Uniform versus one trained on 15-Hard.
#
# Both inputs are completion-sample logs written by run_r1_grpo_dag_v2.py to
# completion_samples/completion_samples_dag_<timestamp>.txt. The timestamp is
# the training start time, so match it against the run's log in logs/.
#
# Usage:
#   ./scripts/plot_uniform_vs_hard.sh <uniform_log.txt> <hard_log.txt> [output.png]
#
# Environment overrides:
#   DEPTH        ancestor depth (in EDGES) to filter both files to (default 14)
#   MAX_STEPS    x-axis extent, must match max_steps in the config (default 800)
#   SMOOTHING    sliding window for the trend line (default 300)
#   LABEL1       legend label for the first file  (default "15-Uniform")
#   LABEL2       legend label for the second file (default "15-Hard")
#
# On DEPTH: the configs express depth in NODES along the longest root-to-target
# path (min/max_ancestor_depth: 15), while the analysis script counts EDGES.
# A 15-node path is 14 edges, hence the default of 14.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <uniform_log.txt> <hard_log.txt> [output.png]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

UNIFORM_LOG="$1"
HARD_LOG="$2"
OUTPUT="${3:-figures/uniform_vs_hard.png}"

DEPTH="${DEPTH:-14}"
MAX_STEPS="${MAX_STEPS:-800}"
SMOOTHING="${SMOOTHING:-300}"
LABEL1="${LABEL1:-15-Uniform}"
LABEL2="${LABEL2:-15-Hard}"

for f in "${UNIFORM_LOG}" "${HARD_LOG}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[error] completion log not found: ${f}" >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${OUTPUT}")"

echo "[plot] uniform=${UNIFORM_LOG}"
echo "[plot] hard=${HARD_LOG}"
echo "[plot] filtering both to ancestor depth ${DEPTH} (edges)"
echo "[plot] output=${OUTPUT}"

python analyze_sliding_accuracy_compare.py \
    "${UNIFORM_LOG}" "${HARD_LOG}" \
    --ancestor-depth "${DEPTH}" \
    --max-steps "${MAX_STEPS}" \
    --smoothing "${SMOOTHING}" \
    --label1 "${LABEL1}" \
    --label2 "${LABEL2}" \
    --y-label "Accuracy on complex examples" \
    --output "${OUTPUT}"

echo "[done] figure written to ${OUTPUT}"
