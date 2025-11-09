# Experiments Guide

This guide explains the three experiments and how to run them.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment 1: Train on different chain sizes
python train_on_D.py

# Run experiment 2: Fixed graph, varying test chains  
python exp_train_on_D_m.py

# Run experiment 3: Visualize A matrix
python plot_A_heatmap.py
```

## Experiment Details

### 1. Train on D (`train_on_D.py`)

**Goal**: Understand how model performance scales with chain size.

**Setup**:
- Train 3 separate models on chain sizes: 10, 20, 30
- Each graph has 2 chains of the specified size
- Graph sizes: 20, 40, 60 respectively

**Evaluation Metrics**:
1. **Training Accuracy**: Final epoch accuracy on training data
2. **Test Accuracy**: Accuracy on held-out test set (same distribution)
3. **Chain Traverse**: % of samples where model generates exact correct path

**Expected Runtime**: ~30-60 minutes per chain size (total: 1.5-3 hours)

**Output**: `results_train_on_D.csv`

**Example Results**:
```
chain_size,train_accuracy,test_accuracy,chain_traverse
10,99.50,98.75,95.20
20,98.30,96.80,92.50
30,97.10,94.50,89.30
```

---

### 2. Train on D_m (`exp_train_on_D_m.py`)

**Goal**: Test generalization from small chains to larger chains.

**Setup**:
- Fixed graph size: 30 vertices
- Train on chain_size=5 (2 chains of 5, rest disconnected)
- Test on chain_size=5, 10, 15 (using the same trained model)

**Why This Matters**: 
Shows whether model learns general chain-following behavior or just memorizes specific lengths.

**Evaluation Metrics**: Same as Experiment 1, but evaluated on 3 different test distributions.

**Expected Runtime**: ~30-45 minutes (train once, test 3 times)

**Output**: `results_train_on_D_m.csv`

**Example Results**:
```
chain_size,train_accuracy,test_accuracy,chain_traverse
5,99.20,98.50,94.80    # In-distribution
10,99.20,85.30,78.40   # Out-of-distribution (longer)
15,99.20,72.10,65.20   # Out-of-distribution (much longer)
```

**Expected Observation**: Performance degrades as test chain size increases beyond training size.

---

### 3. Plot A Heatmap (`plot_A_heatmap.py`)

**Goal**: Visualize what the model learns in its attention mechanism.

**Setup**:
- Train on chain_size=5
- Save weight snapshots during training
- Plot A (Query) matrix as heatmap

**Outputs**:
1. `A_matrix_heatmap.png`: Final A matrix visualization
2. `A_matrix_comparison.png`: Initial vs final A matrix

**Expected Runtime**: ~30-45 minutes

**What to Look For**:
- **Structure in A matrix**: Does it show patterns corresponding to graph structure?
- **Sparse vs Dense**: Are values concentrated in specific regions?
- **Evolution**: How does A change from initialization to final state?

---

## Understanding the Metrics

### Standard Accuracy
- Measures if model predicts correct sink vertex
- Binary: right or wrong final answer
- **Formula**: `correct_sinks / total_samples`

### Chain Traverse Accuracy
- Stricter metric: entire path must be correct
- Checks if `generated_path == gold_path` (exact match)
- **Formula**: `exact_matches / total_samples`
- Always ≤ Standard Accuracy

**Example**:
```
Gold path:     v2 → v5 → v8
Generated:     v2 → v5 → v8    ✓ Both metrics correct
Generated:     v2 → v7 → v8    ✓ Standard accuracy (reaches v8)
                               ✗ Chain traverse (wrong path)
```

---

## Key Parameters

### Model Configuration
```python
# All experiments use these defaults:
tmp = 1/20                           # Temperature for attention
layers = 1                           # Single attention layer
masked_some_attention = True         # Edges attend to nothing
mask_out_last_vertex_as_output = True  # No repeated vertices
```

### Training Configuration
```python
# Defaults (can be changed in run_exp_util.py):
train_samples = 1_000_000           # 1M training samples
val_samples = 1_000                 # 1K validation samples
num_epochs = 20                     # 20 training epochs
batch_size = 2_000                  # Batch size
learning_rate = 1e-2                # Adam learning rate
```

### Connectivity Patterns

**Experiment 1 & 3** (chain_size = n):
```python
[True]*(n-1) + [False] + [True]*(n-1)
# Example (n=5): [T, T, T, T, F, T, T, T, T]
# Creates: v0→v1→v2→v3→v4  (disconnected)  v5→v6→v7→v8→v9
```

**Experiment 2** (train_chain_size = n, graph_size = m):
```python
[True]*(n-1) + [False] + [True]*(n-1) + [False]*(m-2n)
# Example (n=5, m=30): [T]*4 + [F] + [T]*4 + [F]*20
# Creates: v0→v1→v2→v3→v4  (disconnected)  v5→v6→v7→v8→v9  (disconnected)  v10...v29
```

---

## Customization

### Change Training Size
Edit in each experiment file:
```python
train_samples = 100000  # Reduce for faster experiments
val_samples = 500       # Reduce for faster validation
```

### Change Chain Sizes
```python
# train_on_D.py
chain_sizes = [5, 10, 15]  # Try smaller sizes

# exp_train_on_D_m.py
train_chain_size = 3       # Train on smaller chains
test_chain_sizes = [3, 5, 7]  # Test on these sizes
```

### Change Epochs
Edit in `run_exp_util.py`:
```python
def train_list_classifier(..., num_epochs: int = 10, ...):  # Reduce to 10
```

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch_size in experiment files:
batch_size = 1000  # or 500
```

### Training Too Slow
```python
# Reduce dataset size:
train_samples = 100000
val_samples = 500

# Or reduce epochs:
num_epochs = 10
```

### Experiments Not Converging
```python
# Try increasing epochs:
num_epochs = 30

# Or adjusting learning rate in run_exp_util.py:
optim = torch.optim.Adam(model.parameters(), lr=5e-3)
```

---

## Expected Outputs Summary

| Experiment | CSV File | PNG Files | Runtime |
|-----------|----------|-----------|---------|
| train_on_D | results_train_on_D.csv | - | 1.5-3h |
| exp_train_on_D_m | results_train_on_D_m.csv | - | 30-45min |
| plot_A_heatmap | - | A_matrix_heatmap.png<br>A_matrix_comparison.png | 30-45min |

---

## Next Steps

After running experiments:

1. **Analyze CSV results**: Plot accuracy vs chain size
2. **Compare experiments**: Does training on diverse sizes help?
3. **Study A matrix**: What patterns emerge?
4. **Try variations**: Different connectivity patterns, graph sizes, etc.

