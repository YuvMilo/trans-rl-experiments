Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Dat".

## Setup

```bash
conda create -n trans_rl python=3.11
conda activate trans_rl
pip install -r requirements.txt
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

Trains models with chainging task difficulty and generates loss plots.

```bash
# Linear attention (default)
python theoretically_inspired/exp_vanish_grad.py --linear

# Softmax attention
python theoretically_inspired/exp_vanish_grad.py --softmax
```

Output: `result/theoretically_inspired/exp_vanish_grad_linear/` and
`result/theoretically_inspired/exp_vanish_grad/`

## Directory Structure

```
trans-rl-experiments/
├── theoretically_inspired/      # All experiment code
│   ├── exp_train_on_D.py
│   ├── exp_train_on_D_m.py
│   ├── exp_vanish_grad.py
│   ├── run_exp_util.py
│   ├── models/
│   ├── dag_datasets/
│   └── utils/
├── result/
│   └── theoretically_inspired/  # All experiment outputs
│       ├── exp_train_on_D/
│       ├── exp_train_on_D_linear/
│       ├── exp_train_on_D_m/
│       ├── exp_train_on_D_m_linear/
│       ├── exp_vanish_grad/
│       └── exp_vanish_grad_linear/
├── requirements.txt
└── README.md
```
