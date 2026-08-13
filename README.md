Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data".

## Environments

Three sets of experiments, three conda environments. Create the ones you need.

```bash
# 1. Theoretically analysed experiments
conda create -n trans_rl python=3.9.23
conda activate trans_rl
pip install -r requirements.txt
```

```bash
# 2. Real-world LLM experiments
conda create -n real_world_llm python=3.11
conda activate real_world_llm
pip install torch==2.10.0
pip install numpy==2.2.6 psutil packaging ninja
pip install -r requirements_real_world_llm.txt --no-build-isolation
```

```bash
# 3. Synthetic LLM experiments
cd synthetic_llm_experiments
./bootstrap.sh
./scripts/setup_env.sh
conda activate interplay-rl
```

---

## 1. Theoretically analysed experiments

Run from the repository root in `trans_rl`. Each script accepts `--linear`
(default) or `--softmax`.

Run one experiment at a time:

```bash
# Emergence of efficient reasoning
python theoretically_inspired/exp_train_on_D.py --linear

# Out-of-distribution generalization
python theoretically_inspired/exp_train_on_D_m.py --linear

# Complex tasks require simple tasks
python theoretically_inspired/exp_vanish_grad.py --linear
```

Results go to `result/theoretically_inspired/<script>_linear/` for `--linear`
and `result/theoretically_inspired/<script>/` for `--softmax`.

---

## 2. Real-world LLM experiments

Needs 8 GPUs with at least 48GB each. Run from `real_world_llm_experiments/` in
`real_world_llm`. The Llama weights are gated:

```bash
export HF_TOKEN=hf_...
cd real_world_llm_experiments
```

### Train

All configs of one model family, sequentially:

```bash
./scripts/run_all_training.sh qwen
./scripts/run_all_training.sh llama
```

Or one config at a time:

```bash
./scripts/train.sh configs/qwen-15-uniform.yaml
```

| Config | Model | Nodes | Training distribution |
| --- | --- | --- | --- |
| `configs/qwen-10-uniform.yaml` | Qwen2.5-3B-Instruct | 10 | 10-uniform |
| `configs/qwen-15-uniform.yaml` | Qwen2.5-3B-Instruct | 15 | 15-uniform |
| `configs/qwen-15-hard.yaml` | Qwen2.5-3B-Instruct | 15 | 15-hard |
| `configs/llama-10-uniform.yaml` | Llama-3.2-3B-Instruct | 10 | 10-uniform |
| `configs/llama-15-uniform.yaml` | Llama-3.2-3B-Instruct | 15 | 15-uniform |
| `configs/llama-15-hard.yaml` | Llama-3.2-3B-Instruct | 15 | 15-hard |

Checkpoints go to `runs/<config-name>/`, sampled generations to
`completion_samples/completion_samples_dag_<timestamp>.txt`, console output to
`logs/`. Override the defaults with `NUM_PROCS`, `CUDA_VISIBLE_DEVICES`, and
`ACCEL_CFG`.

### Evaluate

Every checkpoint against every test distribution, 1000 greedy samples per cell:

```bash
./scripts/run_eval_matrix.sh qwen
./scripts/run_eval_matrix.sh llama
```

Results go to `evaluation_results/<family>_matrix_<timestamp>/`.

### Plot

```bash
./scripts/plot_uniform_vs_hard.sh \
    completion_samples/completion_samples_dag_<uniform_ts>.txt \
    completion_samples/completion_samples_dag_<hard_ts>.txt \
    figures/uniform_vs_hard.png
```

---

## 3. Synthetic LLM experiments

Needs Linux x86_64, 4 GPUs, an NVIDIA driver compatible with CUDA 12.6, and 35GB
of free disk. Run from `synthetic_llm_experiments/` in `interplay-rl`.

Download the pinned dataset and model, build the training files, and
authenticate to the WandB project `composition-10B-op-RL`:

```bash
cd synthetic_llm_experiments
HF_BIN="$(command -v hf)" ./scripts/download_assets.sh
./scripts/build_datasets.sh
wandb login
./scripts/test.sh
```

Run one experiment at a time:

```bash
./run.sh uniform-op10-20
```

| Experiment | Operation counts | Training rows |
| --- | --- | --- |
| `hard-op20` | 20 | 500,000 |
| `uniform-op10-17` | 10-17 | 400,000 |
| `uniform-op10-20` | 10-20 | 550,000 |

Logs go to `logs/` and checkpoints to `checkpoints/<experiment>/`. GPUs 0-3 are
used by default; override with `CUDA_VISIBLE_DEVICES=0,2,4,6`. The launcher
refuses a nonempty checkpoint directory unless you pass `ALLOW_EXISTING=1`.

---

## Credits

The real-world LLM experiments started from Phil Schmid's
[Mini-R1: Reproduce Deepseek R1 "aha moment"](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)
tutorial and the
[`run_r1_grpo.py`](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py)
script it ships.
The synthetic LLM experiments build on
[Interplay-LM-Reasoning](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning).
`bootstrap.sh` checks out revision `40b9d29`, and the runs use their
[composition](https://huggingface.co/datasets/Interplay-LM-Reasoning/composition)
dataset and
[extrapolation_rl](https://huggingface.co/Interplay-LM-Reasoning/extrapolation_rl)
base model. All three pinned revisions are recorded in
`synthetic_llm_experiments/manifests/sources.json`.
