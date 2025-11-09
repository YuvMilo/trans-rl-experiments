# Repository Structure Summary

This document describes what was included in the clean repository and what was excluded.

## Included Files

### Main Experiment
- `main_exp.py` - Main experiment script based on `RLCoTLinearPer.py`
  - Cleaned up cell markers
  - Removed commented-out code sections
  - Kept all functional training and evaluation code

### dag_datasets Module
- `dag_tokenizer.py` - DAG tokenization with vertex and edge tokens
- `list_classification_dataset.py` - Random graph dataset generation
- `permutation_list_classification_dataset.py` - Fixed pattern dataset generation

### models Module
- `linear_transformer.py` - Linear attention transformer implementation
  - LinearTransformer model
  - LinearAttentionLayer
  - AttentionConfig and TransformerConfig
  - Weight and activation tracking classes

### utils Module
- `misc.py` - Utility functions (collate_fn, vocab helpers)
- `graph_utils.py` - Graph validation functions (is_valid_topo)
- `training_utils.py` - Training utilities (weight saving, SavedWeights class)

### visualization Module
- `training_dynamics.py` - Training visualization functions
  - visualize_linear_training_dynamics for creating GIFs and static images

### Documentation
- `README.md` - Repository overview and usage instructions
- `requirements.txt` - Python dependencies
- `STRUCTURE.md` - This file

## Excluded Files

The following were NOT included as they are not needed for the main experiment:

### From dag_datasets:
- `dag_dataset.py` - Not used in RLCoTLinearPer.py
- `dag_dataset_uniform_over_top.py` - Not used
- `dag_tokenizer_with_exit.py` - Not used
- `permutation_list_classification_dataset_with_exit.py` - Not used

### From models:
- `simplified_linear_transformer.py` - Not used in main experiment
- `softmax_transformer.py` - Not used in main experiment

### From visualization:
- `traning_stats.py` - Not used in main experiment
- `visualize_training_dynamics` (full version) - Only kept the linear version

### Other excluded:
- All experiment result directories
- All backup files
- All other experiment scripts (RLCoT.py, RLCoTLinear.py, etc.)
- __pycache__ directories
- Test files
- Documentation files not directly related to this code

## Import Structure

The clean __init__.py files only expose what's actually used:

```python
# dag_datasets/__init__.py
from .dag_tokenizer import DAGTokenizer
from .list_classification_dataset import ListClassificationDataset
from .permutation_list_classification_dataset import PermutationListClassificationDataset

# models/__init__.py
from .linear_transformer import (
    LinearTransformer,
    LinearAttentionLayer,
    LinearTransformerActivations,
    LinearAttentionActivation,
    LinearTransformerWeights,
    LinearAttentionLayerWeights,
    AttentionConfig,
    TransformerConfig
)

# utils/__init__.py
from .misc import collate_fn, get_vocab_labels_and_positions
from .training_utils import get_current_training_weights, SavedWeights
from .graph_utils import is_valid_topo

# visualization/__init__.py
from .training_dynamics import visualize_linear_training_dynamics
```

## Dependencies

The repository has minimal dependencies:
- torch (core ML framework)
- numpy (numerical operations)
- matplotlib (visualization)
- tqdm (progress bars)
- pillow (GIF creation)

All dependencies are listed in `requirements.txt`.

