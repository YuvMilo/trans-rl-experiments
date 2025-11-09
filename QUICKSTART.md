# Quick Start Guide

## Installation

```bash
cd new_repo_code
pip install -r requirements.txt
```

## Run the Main Experiment

```bash
python main_exp.py
```

This will:
1. Generate a demo dataset with 5 samples
2. Create training dataset (500k samples) with fixed connectivity pattern
3. Create validation dataset (1k samples) with random graphs
4. Train a linear transformer for 10 epochs
5. Save training dynamics visualizations to `results_RLCoT/training_visualizations_permutation/`

## Key Configuration Options

Edit the following in `main_exp.py` (around line 638):

```python
train_samples = 500000          # Number of training samples
val_samples = 1000             # Number of validation samples
connectivity_pattern = [True,]*4+[False,]+[True,]*4  # Edge connectivity
starting_vertex = 3            # Source vertex selection (-1 for random)
layers_d = 1                   # Number of transformer layers
use_masked_attention = True    # Enable attention masking
batch_size = 2000             # Training batch size
num_epochs = 10               # Number of training epochs
```

## Expected Output

### During Training
- Progress bars with loss and accuracy metrics
- Rolling 50-step average accuracy
- Validation accuracy after each epoch
- Example predictions with input/output sequences

### After Training
- Weight matrix visualizations (V, Q, K)
- Training dynamics GIF showing weight evolution
- Static images at key training steps (start, middle, end)

### Output Directory
All visualizations are saved to:
```
results_RLCoT/training_visualizations_permutation/
├── linear_training_dynamics_evolution.gif
├── linear_training_dynamics_start_step_0.png
├── linear_training_dynamics_middle_step_XXX.png
└── linear_training_dynamics_end_step_XXX.png
```

## Understanding the Task

The model learns to:
1. Parse a graph structure encoded as edge tokens
2. Given a source vertex, navigate the graph
3. Predict the sink vertex (end of the chain containing the source)

### Example
Input: `[(0,1) (1,2) (3,4) v1]`
- Edges: 0→1, 1→2, 3→4
- Source: vertex 1
- Target: vertex 2 (end of chain containing vertex 1)

## Troubleshooting

### CUDA out of memory
Reduce `batch_size` in the training configuration (line 666).

### Training is slow
- Reduce `train_samples` (line 607)
- Reduce `dag_size` (line 612)
- Reduce `num_epochs` (line 665)

### Visualization fails
Ensure matplotlib and pillow are installed:
```bash
pip install matplotlib pillow
```

## Next Steps

1. **Experiment with connectivity patterns**: Modify `connectivity_pattern` to create different graph structures
2. **Try different attention mechanisms**: Toggle `use_masked_attention`
3. **Scale up**: Increase `dag_size` for larger graphs
4. **Analyze results**: Examine weight matrices to understand what the model learned

