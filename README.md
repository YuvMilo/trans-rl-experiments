Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data".

## Setup

The three sets of experiments need three separate environments; their dependency
pins are mutually incompatible. The environment below covers the theoretically
analysed experiments:

```bash
conda create -n trans_rl python=3.9.23
conda activate trans_rl
pip install -r requirements.txt
```

The [real-world LLM experiments](#real-world-llm-experiments) use a Python 3.11
virtualenv built from `requirements_real_world_llm.txt`. The
[synthetic LLM experiments](#synthetic-llm-experiments) use a conda environment
that their own `scripts/setup_env.sh` builds, because that install also needs a
pinned upstream checkout, a CUDA-specific torch build, and a flash-attn wheel.
Both are described in their own sections below.

## Reproducing the Figures

The theoretically analysed experiments are run from the root of the repository.
Each set of LLM experiments is run from its own directory; each section states
where its commands belong.

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

## Real-World LLM Experiments

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
pip install -r requirements_real_world_llm.txt --no-build-isolation
```

The Llama weights are gated, so export a token with access before training or
evaluating them:

```bash
export HF_TOKEN=hf_...
```

Training assumes 8 GPUs with at least 48GB each. vLLM is **colocated**: every
rank trains and hosts its own vLLM worker on the same GPU, so no separate
inference server is needed.

Every remaining command in this section is run from `real_world_llm_experiments/`,
which is also where the scripts write their checkpoints, logs and figures:

```bash
cd real_world_llm_experiments
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

## Synthetic LLM Experiments

GRPO fine-tuning on the Interplay-LM-Reasoning composition task, where each
example is a chain of function applications and complexity is the number of
operations to compose. Nine training distributions over the operation count are
compared, all validated on the same ops 10-20 test set, so the effect of the
training mix is isolated from the evaluation.

Every run uses GRPO with batch size 1024 for 200 steps on a 10B base model.

### Hardware

- Linux x86_64
- An NVIDIA driver compatible with CUDA 12.6
- Four GPUs
- Conda, Git, and at least 35 GB of free disk space

### Environment

Every command in this section is run from `synthetic_llm_experiments/`:

```bash
cd synthetic_llm_experiments
```

`bootstrap.sh` clones the upstream code at the revision pinned in
`manifests/sources.json` into the ignored `.runtime/` directory;
`scripts/setup_env.sh` then builds the conda environment on top of it, including
the CUDA 12.6 torch build, the pins in `requirements-runtime.txt`, and a
flash-attn wheel:

```bash
./bootstrap.sh
./scripts/setup_env.sh
conda activate interplay-rl
```

### Data and model

Download the pinned base dataset and model, then generate the enlarged training
files. Generation enforces the requested difficulty, removes duplicates, and
checks the results against the checksums in `manifests/generated-datasets.json`:

```bash
HF_BIN="$(command -v hf)" ./scripts/download_assets.sh
./scripts/build_datasets.sh
```

Runs are logged to the WandB project `composition-10B-op-RL`, so authenticate
first. This keeps credentials out of the repository:

```bash
wandb login
```

Large artifacts (datasets, models, caches, logs, and checkpoints) are all
generated locally and excluded from Git.

### Verification

```bash
./scripts/verify_environment.py
./scripts/test.sh
```

`./scripts/test.sh --skip-assets` is a quicker check that can be run before the
downloads have finished.

### Experiments

| Experiment | Operation counts | Training rows |
| --- | --- | --- |
| `hard-op20` | 20 | 500,000 |
| `uniform-op10-15` | 10-15 | 300,000 |
| `uniform-op10-17` | 10-17 | 400,000 |
| `uniform-op10-20` | 10-20 | 550,000 |
| `uniform-op10-15-20` | 10, 15, 20 | 412,500 |
| `uniform-op10-13-17-20` | 10, 13, 17, 20 | 550,000 |
| `uniform-op17-20` | 17-20 | 550,000 |
| `uniform-op18-20` | 18-20 | 412,500 |
| `uniform-op5-10-15-20` | 5, 10, 15, 20 | 550,000 |

Each name is also the config filename under `configs/`, which layers the
experiment's training files over the shared settings in `configs/base.yaml`.

### Running

One experiment per invocation, on four GPUs (0-3 by default):

```bash
./run.sh uniform-op10-13-17-20
```

Choose another set of GPUs when needed:

```bash
CUDA_VISIBLE_DEVICES=0,2,4,6 ./run.sh uniform-op17-20
```

A one-step smoke test, which refuses to start if another verl job is already
running on the host:

```bash
./scripts/smoke.sh uniform-op10-13-17-20
```

For a long remote run, use `screen`. Detach with `Ctrl+A` then `D`, and reattach
with `screen -r interplay`:

```bash
screen -S interplay
conda activate interplay-rl
./run.sh uniform-op10-13-17-20
```

Logs are written to `logs/` and checkpoints to `checkpoints/<experiment>/`. The
launcher refuses to reuse a nonempty checkpoint directory; override that only
when resuming deliberately:

```bash
ALLOW_EXISTING=1 ./run.sh uniform-op10-13-17-20
```

If an interrupted run leaves Ray processes behind, clear them with
`ray stop --force`. Only do this when no other Ray job from your account is
active, since it kills every Ray process you own.

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
├── real_world_llm_experiments/      # Equation-system GRPO, separate Python 3.11 env
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
├── synthetic_llm_experiments/       # Composition/op-count GRPO, conda env built by script
│   ├── bootstrap.sh                 # Clones the pinned upstream runtime into .runtime/
│   ├── run.sh                       # Launcher; one experiment per invocation
│   ├── configs/                     # base.yaml plus the nine experiment configs
│   ├── scripts/                     # Env setup, asset download, dataset build, tests
│   │   ├── setup_env.sh
│   │   ├── download_assets.sh
│   │   ├── build_datasets.sh
│   │   ├── build_dataset.py
│   │   ├── validate.py
│   │   ├── verify_environment.py
│   │   ├── test.sh
│   │   └── smoke.sh
│   ├── manifests/                   # Pinned upstream revisions and dataset checksums
│   ├── environment.yml              # Conda env (Python 3.12)
│   └── requirements-runtime.txt     # Pip pins installed by scripts/setup_env.sh
├── result/                          # Outputs from theoretically analysed experiments
│   └── theoretically_inspired/
├── requirements.txt                 # Python 3.9, theoretically analysed experiments
├── requirements_real_world_llm.txt  # Python 3.11 / torch 2.10, real-world LLM experiments
└── README.md
```

