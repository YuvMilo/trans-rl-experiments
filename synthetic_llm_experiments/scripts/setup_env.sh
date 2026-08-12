#!/usr/bin/env bash
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${RUNTIME_DIR:-$BUNDLE/.runtime}"
ENV_NAME="${ENV_NAME:-interplay-rl}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

[[ -d "$RUNTIME/verl" ]] || { echo "Run ./bootstrap.sh first" >&2; exit 2; }
[[ -n "$CONDA_BIN" ]] || { echo "conda not found; set CONDA_EXE" >&2; exit 2; }
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  echo "This lock targets Linux x86_64 with CUDA 12.6" >&2
  exit 2
}

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$BUNDLE/environment.yml"
fi

PY="$("$CONDA_BIN" run -n "$ENV_NAME" which python)"
"$PY" -m pip install \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip install -r "$BUNDLE/requirements-runtime.txt"
"$PY" -m pip install -e "$RUNTIME/LLaMA-Factory"
"$PY" -m pip install -e "$RUNTIME/verl[vllm]"
"$PY" -m pip install \
  huggingface_hub==0.36.2 transformers==4.53.3 vllm==0.9.1 ray==2.52.1
"$PY" -m pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
"$PY" -m pip uninstall -y uvloop

RUNTIME_DIR="$RUNTIME" "$PY" "$BUNDLE/scripts/verify_environment.py"
echo "Activate with: conda activate $ENV_NAME"
