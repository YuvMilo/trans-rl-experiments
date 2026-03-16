Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data".

## Setup

```bash
conda create -n trans_rl python=3.9.23
conda activate trans_rl
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Reproducing the Figures

All commands should be run from the root of the repository.

---

## Theoretically Analysed Setting

The experiments below correspond to the theoretically analysed setting in the paper.
Each experiment can be run with either **softmax** or **linear** attention by passing
`--softmax` or `--linear` (linear is the default).

### Emergence of efficient reasoning (Fig X and Y)

Trains the model on graphs with 2 chains of different sizes (D = 4, 8, 12) and evaluates
standard accuracy and exact chain-traversal accuracy, averaged over 3 seeds.

```bash
# Linear attention (default)
python theoretically_inspired/exp_train_on_D.py --linear

# Softmax attention
python theoretically_inspired/exp_train_on_D.py --softmax
```

Output: `result/theoretically_inspired/exp_train_on_D_linear/` and
`result/theoretically_inspired/exp_train_on_D/`

---

### Out-of-Distribution Generalization (Fig X and Y)

Trains on a fixed graph with small chains (train chain size = 4), then tests the same model
on chain sizes 4, 8, and 12, averaged over 3 seeds.

```bash
# Linear attention (default)
python theoretically_inspired/exp_train_on_D_m.py --linear

# Softmax attention
python theoretically_inspired/exp_train_on_D_m.py --softmax
```

Output: `result/theoretically_inspired/exp_train_on_D_m_linear/` and
`result/theoretically_inspired/exp_train_on_D_m/`

---

### Solving Complex Tasks Requires Training On Simple Tasks (Fig X and Y)

Trains models with changing task difficulty and generates loss plots.

```bash
# Linear attention (default)
python theoretically_inspired/exp_vanish_grad.py --linear

# Softmax attention
python theoretically_inspired/exp_vanish_grad.py --softmax
```

Output: `result/theoretically_inspired/exp_vanish_grad_linear/` and
`result/theoretically_inspired/exp_vanish_grad/`

---

## LLM Experiments

The experiments below correspond to the real-world LLM setting in the paper.
A Qwen 2.5 3B Instruct model is fine-tuned using GRPO (Group Relative Policy Optimization)
on systems of affine equations, where the model must trace a dependency chain to compute
the answer.

The task presents the model with a system of affine equations in natural language:

> x_15 = x_5 + 3. x_5 = x_27 + 2. x_8 = 3. x_4 = x_8 - 2. Find x_15.

The model outputs reasoning within `<think>` tags and a final answer within `<answer>` tags.
Complexity is controlled by the number of reasoning steps (distance from the constant to the
target variable).

### Training Distributions

The paper defines training distributions by the maximum dependency length L:


| Config     | Distribution                       | Reasoning steps   |
| ---------- | ---------------------------------- | ----------------- |
| 5-Uniform  | `min_distance=1, max_distance=4`   | Uniform over 1–4  |
| 10-Uniform | `min_distance=1, max_distance=9`   | Uniform over 1–9  |
| 15-Uniform | `min_distance=1, max_distance=14`  | Uniform over 1–14 |
| 15-Hard    | `min_distance=14, max_distance=14` | Only 14 steps     |


### Running Training (Fig X and Y)

Training uses DeepSpeed ZeRO-3 and vLLM for generation. Adjust `--num_processes`  
according to the number of available GPUs (Should be 1 less than the total GPU count available).

```bash
# 15-Uniform
accelerate launch --num_processes <N> \
    --config_file llm_experiments/qwen_run/deepspeed_zero3.yaml \
    llm_experiments/run_r1_grpo_dag.py \
    --config llm_experiments/configs/15-uniform.yaml

# 10-Uniform
accelerate launch --num_processes <N> \
    --config_file llm_experiments/qwen_run/deepspeed_zero3.yaml \
    llm_experiments/run_r1_grpo_dag.py \
    --config llm_experiments/configs/10-uniform.yaml

# 5-Uniform
accelerate launch --num_processes <N> \
    --config_file llm_experiments/qwen_run/deepspeed_zero3.yaml \
    llm_experiments/run_r1_grpo_dag.py \
    --config llm_experiments/configs/5-uniform.yaml

# 15-Hard
accelerate launch --num_processes <N> \
    --config_file llm_experiments/qwen_run/deepspeed_zero3.yaml \
    llm_experiments/run_r1_grpo_dag.py \
    --config llm_experiments/configs/15-hard.yaml
```

Output: model checkpoints in `runs/`, completion samples in `completion_samples/`

---

### Evaluation

Evaluates a trained model on test samples across varying path lengths and chain sizes
using greedy decoding.

Below is an example of the evaluation command for the 15-Uniform model on a path length of 14.

```bash
python llm_experiments/evaluate_model.py \
    --model_path runs/<checkpoint_dir> \
    --chain_sizes "30" \
    --max_label_value 30 \
    --min_path_length 14  \
    --max_path_length 14  \
    --training_min_chain_size 15  \
    --training_max_chain_size 15  \
    --training_min_path 1   \
    --training_max_path 14   \
    --samples_per_config 1000 \
    --save_samples
```

Output: `evaluation_results/`

Key arguments:


| Argument                    | Description                                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--model_path`              | Path to trained model checkpoint                                                                                  |
| `--samples_per_config`      | Test samples per (chain_size, path_length) config                                                                 |
| `--min_path_length`         | Minimum path length to test                                                                                       |
| `--max_path_length`         | Maximum path length to test                                                                                       |
| `--training_min_path`       | Min path length seen during training (for ID/OOD labels)                                                          |
| `--training_max_path`       | Max path length seen during training (for ID/OOD labels)                                                          |
| `--training_min_chain_size` | Min chain size seen during training (for ID/OOD labels)                                                           |
| `--training_max_chain_size` | Max chain size seen during training (for ID/OOD labels)                                                           |
| `--chain_sizes`             | A comma seperated string of all the graph sizes/variable counts (total node/variable count) to run evaluation on. |
| `--max_label_value`         | Maximum value given to any label (Variable name) in the system                                                    |
| `--save_samples`            | Save sample outputs to file                                                                                       |


---

### Plotting Training Curves

```bash
# Compare two runs
python llm_experiments/analyze_sliding_accuracy_compare.py \
    completion_samples/<file1>.txt \
    completion_samples/<file2>.txt \
    --path-lengths "14" \
    --label1 "Uniform" \
    --label2 "Hard Only" \
    --strict-format \
    -o comparison.png \
    --raw-smoothing 1 \
    --raw-window-size 50 \
    --y-label "Test Accuracy (%)"
```

---

## Directory Structure

```
.
├── theoretically_inspired/          # Theoretically analysed experiments
│   ├── exp_train_on_D.py
│   ├── exp_train_on_D_m.py
│   ├── exp_vanish_grad.py
│   ├── run_exp_util.py
│   ├── models/
│   ├── dag_datasets/
│   └── utils/
├── llm_experiments/                 # LLM (GRPO) experiments
│   ├── run_r1_grpo_dag.py           # Main GRPO training script
│   ├── dag_dataset_simplified.py    # Dataset generation
│   ├── evaluate_model.py            # Model evaluation
│   ├── analyze_sliding_accuracy_compare.py  # Plotting training curves
│   ├── configs/                     # Pre-configured experiments
│   │   ├── 5-uniform.yaml
│   │   ├── 10-uniform.yaml
│   │   ├── 15-uniform.yaml
│   │   └── 15-hard.yaml
│   └── qwen_run/
│       └── deepspeed_zero3.yaml     # DeepSpeed ZeRO-3 config
├── result/                          # Outputs from theoretically analysed experiments
│   └── theoretically_inspired/
├── requirements.txt
└── README.md
```

