Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data".

## Setup

The two sets of experiments need separate environments; their dependency pins are
incompatible. The environment below covers the theoretically analysed experiments:

```bash
conda create -n trans_rl python=3.9.23
conda activate trans_rl
pip install -r requirements.txt
```

The LLM experiments use their own Python 3.11 environment, built from
`requirements_llm.txt` and described in [LLM Experiments](#llm-experiments) below.

## Reproducing the Figures

Both environments are installed from the root of the repository. The
theoretically analysed experiments are also run from there; the LLM experiments
are run from `llm_experiments/`, as noted in that section.

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

GRPO fine-tuning of pretrained LLMs on synthetic systems of equations whose
dependency structure is a random **directed acyclic graph**. These are the
experiments reported in the "real-world" section of the paper.

### The task

Each example is a shuffled list of equations over variables `x_1 ... x_n`. Some
variables are assigned constants; the rest are affine combinations (mod 2) of
other variables. The model is asked for the value of one target variable:

```
x_10 = 0. x_7 = (x_2 + x_8 - 1) mod 2. x_13 = (x_2 + x_12) mod 2.
x_9 = (x_2 + 1) mod 2. x_14 = (x_6) mod 2. x_15 = (x_2 + x_10 - 1) mod 2.
x_11 = (x_4 + x_10 + 1) mod 2. x_6 = 0. x_4 = 1. x_8 = (x_6 + x_10 - 1) mod 2.
x_1 = (x_6 + 1) mod 2. x_12 = (x_4 + x_6 - 1) mod 2. x_3 = (x_7 + x_10 + 1) mod 2.
x_2 = (x_4 + x_12 + 1) mod 2. x_5 = (x_3 + x_11) mod 2. Find x_5 (mod 2).
```

The variables the target actually depends on form its **ancestor subgraph**;
the remaining variables are distractors. Task complexity is the ancestor
subgraph's depth, i.e. the length of the longest path from a constant
assignment down to the target.

The model is prompted to reason inside `<think>` tags and answer inside
`<answer>` tags. It is given no instructions on *how* to reason, and no
supervision on intermediate steps. Two rewards are used, both binary: one for
adhering to the output format, one for the final answer being exactly right.

#### Distribution naming

Depth is controlled by partitioning the nodes into buckets and only allowing
edges that strictly descend bucket levels, which fixes the target's ancestor
depth exactly. Two families of training distribution are used:

| Name | Meaning | Config field |
| --- | --- | --- |
| `n-uniform` | depth sampled uniformly in `1..n` | `min_ancestor_depth: 1`, `max_ancestor_depth: n` |
| `n-hard` | every example at maximal depth `n` | `min_ancestor_depth: n`, `max_ancestor_depth: n` |
| mixes | depth drawn from a sparse set | `ancestor_depths: [5, 10, 15]` |

Note the two conventions for measuring depth. The configs and
`evaluate_simple.py` count **nodes** on the longest root-to-target path, so
`ancestor_depth: 15` means 15 nodes. `analyze_sliding_accuracy_compare.py` and
`count_difficulty.py` count **edges**, so the same examples appear there as
depth 14.

### Environment

The LLM experiments need their own Python 3.11 environment, separate from the
`trans_rl` environment used above. Create it from the repository root:

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install torch==2.10.0
pip install -r requirements_llm.txt --no-build-isolation
```

The Llama weights are gated, so export a token with access before training or
evaluating them:

```bash
export HF_TOKEN=hf_...
```

Training assumes 8 GPUs with at least 48GB each. vLLM is **colocated**: every
rank trains and hosts its own vLLM worker on the same GPU, so no separate
inference server is needed.

Every remaining command in this section is run from `llm_experiments/`, which is
also where the scripts write their checkpoints, logs and figures:

```bash
cd llm_experiments
```

### Experiments

| Config | Model | Nodes | Training distribution |
| --- | --- | --- | --- |
| `configs/qwen-5-uniform.yaml` | Qwen2.5-3B-Instruct | 5 | 5-uniform |
| `configs/qwen-10-uniform.yaml` | Qwen2.5-3B-Instruct | 10 | 10-uniform |
| `configs/qwen-15-uniform.yaml` | Qwen2.5-3B-Instruct | 15 | 15-uniform |
| `configs/qwen-15-hard.yaml` | Qwen2.5-3B-Instruct | 15 | 15-hard |
| `configs/qwen-15-mix-13-15.yaml` | Qwen2.5-3B-Instruct | 15 | depths 13-15 |
| `configs/qwen-15-mix-5-10-15.yaml` | Qwen2.5-3B-Instruct | 15 | depths {5, 10, 15} |
| `configs/llama-5-uniform.yaml` | Llama-3.2-3B-Instruct | 5 | 5-uniform |
| `configs/llama-10-uniform.yaml` | Llama-3.2-3B-Instruct | 10 | 10-uniform |
| `configs/llama-15-uniform.yaml` | Llama-3.2-3B-Instruct | 15 | 15-uniform |
| `configs/llama-15-hard.yaml` | Llama-3.2-3B-Instruct | 15 | 15-hard |

The `5/10/15-uniform` configs support the out-of-distribution generalization
result: train on each, evaluate all of them on 15-hard. The `15-uniform` versus
`15-hard` pair, and the two 15-node mixes, support the result that solving
complex examples requires simple examples in the training data.

Every config shares the same GRPO hyperparameters: 800 steps, 8 generations per
prompt, KL coefficient 1e-4, learning rate 5e-7 with a cosine schedule and 0.03
warmup ratio, 1024-token completion cap, and a 512-token prompt cap that filters
out longer examples. They differ only in model, node count, and depth range.

### Training

One run:

```bash
./scripts/train.sh configs/qwen-15-uniform.yaml
```

A whole model family, sequentially, stopping on the first failure:

```bash
./scripts/run_all_training.sh qwen
./scripts/run_all_training.sh llama
```

Checkpoints land in `runs/<config-name>/`. Sampled generations are appended to
`completion_samples/completion_samples_dag_<timestamp>.txt` during training,
where `<timestamp>` is the moment the run started; these are the inputs to the
analysis scripts. Console output is teed to `logs/`.

`NUM_PROCS`, `CUDA_VISIBLE_DEVICES`, and `ACCEL_CFG` override the defaults (8
processes, GPUs 0-7, DeepSpeed ZeRO-3).

### Evaluation

The full 4x4 matrix of checkpoints against test distributions, at 1000 greedily
decoded samples per cell:

```bash
./scripts/run_eval_matrix.sh qwen
./scripts/run_eval_matrix.sh llama
```

Results are written per cell to
`evaluation_results/<family>_matrix_<timestamp>/<checkpoint>__on__<distribution>/`.
Each cell reports accuracy, format accuracy, and *efficient rate*: the fraction
of correct answers that never mention a distractor variable, which measures
whether the model backtraces from the target rather than evaluating the whole
system.

A single configuration can also be evaluated directly. The dataset flags must
match the training config, otherwise the test distribution won't be the one you
think it is:

```bash
python evaluate_simple.py \
    --model_path runs/qwen-15-uniform \
    --n_samples 1000 --graph_type dag \
    --min_nodes 15 --max_nodes 15 \
    --min_ancestor_depth 15 --max_ancestor_depth 15 \
    --max_in_degree 2 --edge_probability 0.2 \
    --max_degree 1 --max_coefficient 1 --max_terms 2 \
    --probabilistic_pairwise_terms \
    --single_variable_term_probability 1.0 \
    --pairwise_product_term_probability 0.0 \
    --constant_term_probability 0.6666 \
    --min_constant 0 --max_constant 1 \
    --target_sink_only --max_value 5000 --modulus 2 \
    --min_start_value 0 --max_start_value 1 \
    --max_label_value 15 --no_fixed_label_set \
    --sampling_strategy uniform_size
```

### Analysis

Accuracy on complex examples over the course of training, for two runs on one
plot:

```bash
./scripts/plot_uniform_vs_hard.sh \
    completion_samples/completion_samples_dag_<uniform_ts>.txt \
    completion_samples/completion_samples_dag_<hard_ts>.txt \
    figures/uniform_vs_hard.png
```

The script filters both files to maximal-depth examples so the two curves are
comparable. `analyze_sliding_accuracy_compare.py` can be called directly for
other slices: `--ancestor-depth 2,3,4` or `--num-ancestors 3,5,8` to select
subsets, `--strict-format` to count an answer correct only when the output
format is also valid, `--window-size` and `--smoothing` to control the sliding
window.

To see how many examples of each difficulty a completion log actually contains:

```bash
python count_difficulty.py completion_samples/completion_samples_dag_<ts>.txt \
    --metric ancestor_depth
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
├── llm_experiments/                 # LLM (GRPO) experiments, separate Python 3.11 env
│   ├── run_r1_grpo_dag_v2.py        # GRPO training entry point; builds the dataset,
│   │                                #   defines both rewards, runs GRPOTrainer
│   ├── dag_dataset_simplified.py    # Graph and equation-system generator shared by
│   │                                #   training and evaluation
│   ├── evaluate_simple.py           # Evaluates one checkpoint on one generated distribution
│   ├── analyze_sliding_accuracy_compare.py  # Sliding-window accuracy curves for two
│   │                                #   completion logs
│   ├── count_difficulty.py          # Difficulty histogram of a completion log
│   ├── configs/                     # Ten configs: qwen-* and llama-*
│   ├── scripts/                     # Training, evaluation and plotting drivers
│   │   ├── train.sh
│   │   ├── run_all_training.sh
│   │   ├── run_eval_matrix.sh
│   │   └── plot_uniform_vs_hard.sh
│   └── accelerate_configs/
│       └── deepspeed_zero3.yaml     # DeepSpeed ZeRO-3 launch config
├── result/                          # Outputs from theoretically analysed experiments
│   └── theoretically_inspired/
├── requirements.txt                 # Python 3.9, theoretically analysed experiments
├── requirements_llm.txt             # Python 3.11 / torch 2.10, LLM experiments
└── README.md
```

