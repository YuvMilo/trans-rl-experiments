# Linear Transformer for Permutation List Classification

This repository contains a clean implementation of a linear transformer model for DAG-based permutation list classification tasks with three main experiments.

## Structure

```
new_repo_code/
├── dag_datasets/           # Dataset generation and tokenization
│   ├── __init__.py
│   ├── dag_tokenizer.py
│   ├── list_classification_dataset.py
│   └── permutation_list_classification_dataset.py
├── models/                 # Model architectures
│   ├── __init__.py
│   └── linear_transformer.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── graph_utils.py
│   ├── misc.py
│   └── training_utils.py
├── visualization/          # Training visualization tools
│   ├── __init__.py
│   └── training_dynamics.py
├── run_exp_util.py        # Shared experiment utilities
├── train_on_D.py          # Experiment 1: Train on different chain sizes
├── exp_train_on_D_m.py    # Experiment 2: Fixed graph, varying test chains
├── plot_A_heatmap.py      # Experiment 3: Visualize A (Query) matrix
├── main_exp.py            # Original main experiment (for reference)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

### Experiment 1: Train on Different Chain Sizes (`train_on_D.py`)

Trains the model on graphs with 2 chains of varying sizes (10, 20, 30) and evaluates:
- Training accuracy
- Test accuracy  
- Chain traverse accuracy (exact path matching)

**Run:**
```bash
python train_on_D.py
```

**Output:** `results_train_on_D.csv` with columns:
- `chain_size`: Size of each chain (graph_size = chain_size × 2)
- `train_accuracy`: Training accuracy in %
- `test_accuracy`: Test accuracy in %
- `chain_traverse`: Percentage of exact chain traversals

**Configuration:**
- Graph sizes: 20, 40, 60 (2 chains of 10, 20, 30 each)
- Training samples: 1,000,000 (default)
- Validation samples: 1,000 (default)
- Epochs: 20
- Connectivity pattern: `[True]*(chain_size-1) + [False] + [True]*(chain_size-1)`

### Experiment 2: Fixed Graph with Varying Test Chains (`exp_train_on_D_m.py`)

Trains on a fixed graph size (30) with small chains (5), then tests generalization to different chain sizes (5, 10, 15).

**Run:**
```bash
python exp_train_on_D_m.py
```

**Output:** `results_train_on_D_m.csv` with columns:
- `chain_size`: Test chain size
- `train_accuracy`: Training accuracy (same for all, trained once)
- `test_accuracy`: Test accuracy on this chain size
- `chain_traverse`: Exact chain traversal accuracy

**Configuration:**
- Graph size: 30 (fixed)
- Training chain size: 5
- Test chain sizes: 5, 10, 15
- Training samples: 1,000,000 (default)
- Validation samples: 1,000 (default)
- Epochs: 20
- Training pattern: `[True]*4 + [False] + [True]*4 + [False]*20`

### Experiment 3: Visualize A (Query) Matrix (`plot_A_heatmap.py`)

Trains a model on chain_size=5 and visualizes the learned A (Query) matrix.

**Run:**
```bash
python plot_A_heatmap.py
```

**Output:**
- `A_matrix_heatmap.png`: Heatmap of the final A matrix
- `A_matrix_comparison.png`: Initial vs final A matrix comparison

**Configuration:**
- Chain size: 5 (graph size: 10)
- Training samples: 1,000,000 (default)
- Validation samples: 1,000 (default)
- Epochs: 20

## Model Architecture

### Linear Transformer Features

- **Linear Attention**: Uses linear attention mechanism (no softmax)
- **Fixed K Matrix**: K (Key) matrix is fixed to identity and frozen
- **Small Initialization**: Q (A) and V matrices initialized with small std (1e-4)
- **Masked Attention**: Edges attend to nothing, vertices attend only to edges
- **Vertex Masking**: Prevents outputting vertices that already appeared (`mask_out_last_vertex_as_output`)

### Training Configuration

Default parameters (can be overridden):
- Training samples: 1,000,000
- Validation samples: 1,000
- Epochs: 20
- Batch size: 2,000
- Learning rate: 1e-2 (Adam optimizer)
- Temperature: 1/20
- Max new tokens: Depends on chain size

## Shared Utilities (`run_exp_util.py`)

The `run_exp_util.py` module provides reusable functions:

- `init_linear_transformer()`: Initialize model with proper configuration
- `train_list_classifier()`: Main training loop with REINFORCE
- `compute_chain_traverse_accuracy()`: Compute exact path matching accuracy
- `sample_until_max_with_logprob_sums()`: Efficient batched sampling
- Loss and metric computation functions

## Key Features

1. **Never Sample Chain Ends**: Dataset ensures source vertex is never the last vertex in a chain
2. **Exact Chain Traversal**: Evaluates whether generated path exactly matches the gold chain
3. **REINFORCE Training**: Uses policy gradient with reward based on reaching correct sink
4. **Efficient Sampling**: Batched autoregressive sampling with early stopping on sink vertices
5. **Weight Tracking**: Optional weight snapshot saving for analysis

## Results Format

### CSV Files

All experiments save results as CSV with consistent format:
```csv
chain_size,train_accuracy,test_accuracy,chain_traverse
10,99.50,98.75,95.20
20,98.30,96.80,92.50
30,97.10,94.50,89.30
```

All accuracy values are rounded to 2 decimal places.

## Connectivity Patterns

The experiments use specific connectivity patterns:

1. **Two Chains**: `[True]*(n-1) + [False] + [True]*(n-1)`
   - Creates 2 chains of size n each
   
2. **Two Chains + Disconnected**: `[True]*(n-1) + [False] + [True]*(n-1) + [False]*(m)`
   - Creates 2 chains of size n, plus m disconnected vertices

## Citation

If you use this code in your research, please cite:

```bibtex
@software{linear_transformer_dag,
  title={Linear Transformer for DAG Permutation Classification},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```
