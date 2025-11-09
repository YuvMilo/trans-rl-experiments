# %%[markdown]
"""
Experiment 3: Plot A (Query) Matrix Heatmap

This experiment trains a model on chain_size=5 and visualizes the learned
A (Q) matrix as a heatmap.

Results are saved to: A_matrix_heatmap.png

## Added: Also plots the V matrix as a heatmap for inspection (not saved to file)
"""
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import random
import numpy as np

import torch
import matplotlib.pyplot as plt

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import train_list_classifier, plot_training_curves, init_simplified_linear_transformer

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



# %%
# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

chain_size = 4
dag_size = chain_size * 2  # 2 chains
train_samples = 1000000
val_samples = 10000
num_epochs = 200
batch_size = 10000
starting_vertex = -1  # Random source selection

# Create results directory
RESULTS_DIR = "results/plot_A_heatmap"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "run_losses"), exist_ok=True)

# %%
# Training
print(f"\n{'='*80}")
print(f"Training on chain_size={chain_size} (graph_size={dag_size})")
print(f"{'='*80}")

# Create 2 chains of size chain_size
connectivity_pattern = [True,] * (chain_size - 1) + [False,] + [True,] * (chain_size - 1)

print(f"DAG size: {dag_size}")
print(f"Connectivity pattern: {connectivity_pattern}")

# Create tokenizer and datasets
tokenizer = DAGTokenizer(dag_size=dag_size)

train_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=train_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)

val_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=val_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=-1,
)

print(f"Training dataset: {len(train_ds)} samples")
print(f"Validation dataset: {len(val_ds)} samples")

# Train the model with weight saving
training_results = train_list_classifier(
    train_ds,
    val_ds,
    num_epochs=num_epochs,
    batch_size=batch_size,
    device=device,
    init_transformer_fn=init_simplified_linear_transformer,
    tmp=1/5,
    layers=1,
    max_new_tokens=chain_size+5,
    masked_some_attention=False,
    mask_out_last_vertex_as_output=True,  # Explicitly pass True
    verbose=True,
    save_weights=True,  # Save weights for visualization
)

model = training_results["model"]
saved_weights = training_results.get("saved_weights", [])

print(f"\nTraining completed!")
print(f"Saved {len(saved_weights)} weight snapshots")

# Plot training curves
plot_training_curves(training_results, save_path=os.path.join(RESULTS_DIR, f"training_curves_chain_{chain_size}.png"))

# Save individual val accuracy and val loss plots
val_hist = training_results["val_history"]
epochs_list = [h['epoch'] for h in val_hist]
val_acc_list = [h['accuracy'] for h in val_hist]
val_loss_list = [h['loss'] for h in val_hist]

# Val accuracy plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_list, val_acc_list, 'r-o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title(f'Validation Accuracy - Chain Size {chain_size}', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_acc_chain_{chain_size}.png"), dpi=150)
plt.close()

# Val loss plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_list, val_loss_list, 'b-o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title(f'Validation Loss - Chain Size {chain_size}', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_loss_chain_{chain_size}.png"), dpi=150)
plt.close()

# Extract A (Q) and V matrix from final weights
if saved_weights:
    final_weights = saved_weights[-1]
    
    # Get the actual learned A matrix (num_vertices x num_context) if available
    # Otherwise fall back to the full sparse Q matrix
    layer_weights = model.get_weights().attention_layers_weights[0]
    if hasattr(layer_weights, 'A_raw'):
        A_matrix = layer_weights.A_raw  # The actual learned matrix
        print(f"Using actual learned A matrix (num_vertices × num_context)")
    else:
        A_matrix = final_weights.Q_matrix  # Fall back to full Q
        print(f"Using embedded Q matrix (full vocab)")

    # Try to get the V matrix (for many RLCoT experiments this is exposed as .V_matrix)
    if hasattr(final_weights, "V_matrix"):
        V_matrix = final_weights.V_matrix
    elif hasattr(final_weights, "V") and isinstance(final_weights.V, np.ndarray):
        V_matrix = final_weights.V
    else:
        V_matrix = None
    
    print(f"\nA matrix shape: {A_matrix.shape}")
    print(f"A matrix statistics:")
    print(f"  Min: {A_matrix.min():.6f}")
    print(f"  Max: {A_matrix.max():.6f}")
    print(f"  Mean: {A_matrix.mean():.6f}")
    print(f"  Std: {A_matrix.std():.6f}")
    if V_matrix is not None:
        print(f"\nV matrix shape: {V_matrix.shape}")
        print(f"V matrix statistics:")
        print(f"  Min: {V_matrix.min():.6f}")
        print(f"  Max: {V_matrix.max():.6f}")
        print(f"  Mean: {V_matrix.mean():.6f}")
        print(f"  Std: {V_matrix.std():.6f}")
    else:
        print("No V matrix found! (V_matrix attribute missing in saved weights?)")
else:
    print("No saved weights found!")
    exit(1)

# Plot A matrix heatmap
print(f"\nCreating A matrix heatmap...")

# Transpose A for visualization (rows = vertices, cols = context tokens)
A_to_plot = A_matrix.T  # Now shape: (num_context, num_vertices)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(A_to_plot, cmap='RdBu_r', aspect='auto', interpolation='nearest')

ax.set_title(f'A (Query) Matrix Heatmap - Chain Size {chain_size}\n(Trained for {num_epochs} epochs)\nShape: {A_matrix.shape[0]} vertices × {A_matrix.shape[1]} context tokens', 
             fontsize=16, pad=20)
ax.set_xlabel('Vertex Queries', fontsize=14)
ax.set_ylabel('Context Tokens (Vertices + Edges)', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, label='A Matrix Value', shrink=0.8)
cbar.ax.tick_params(labelsize=12)

# Add grid for better readability
num_vertices_display = A_matrix.shape[0]
num_context_display = A_matrix.shape[1]
ax.set_xticks(np.arange(0, num_vertices_display, max(1, num_vertices_display // 10)), minor=False)
ax.set_yticks(np.arange(0, num_context_display, max(1, num_context_display // 10)), minor=False)
ax.grid(which='major', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)

# Add text annotation with matrix stats
stats_text = f"Shape: {A_matrix.shape}\nMin: {A_matrix.min():.4f}\nMax: {A_matrix.max():.4f}\nMean: {A_matrix.mean():.4f}"
ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_file = os.path.join(RESULTS_DIR, "A_matrix_heatmap.png")
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"Heatmap saved to: {output_file}")

plt.show()

# Plot V matrix heatmap (if available)
if V_matrix is not None:
    print("\nCreating V matrix heatmap...")
    max_v_display = 50
    v_display_size_0 = min(max_v_display, V_matrix.shape[0])
    v_display_size_1 = min(max_v_display, V_matrix.shape[1]) if V_matrix.ndim > 1 else 1
    # Use transpose for visualization consistency if matrix is square
    matrix_v_to_plot = V_matrix[:v_display_size_0, :v_display_size_1]
    if v_display_size_0 == v_display_size_1:
        matrix_v_to_plot = matrix_v_to_plot.T
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix_v_to_plot, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax.set_title(f'V Matrix Heatmap - Chain Size {chain_size} (not saved to file)', fontsize=16, pad=20)
    ax.set_xlabel('Input Token Dimension', fontsize=14)
    ax.set_ylabel('Value Dimension', fontsize=14)
    cbar = plt.colorbar(im, ax=ax, label='V Matrix Value', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xticks(np.arange(0, v_display_size_1, 5), minor=False)
    ax.set_yticks(np.arange(0, v_display_size_0, 5), minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    v_stats_text = f"Shape: {V_matrix.shape}\nMin: {V_matrix.min():.4f}\nMax: {V_matrix.max():.4f}\nMean: {V_matrix.mean():.4f}"
    ax.text(1.15, 0.5, v_stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.tight_layout()
    plt.show()

# Optional: Plot initial vs final A matrix comparison
if len(saved_weights) > 1:
    print(f"\nCreating initial vs final A matrix comparison...")
    
    # Get initial A matrix
    initial_weights = saved_weights[0]
    # For initial weights, we need to extract from the model at that point
    # For now, just skip the comparison or use final matrix
    print("Note: Initial vs final comparison requires storing model snapshots")
    print("Skipping comparison plot for simplified model")

print("\n" + "="*80)
print("Experiment completed successfully!")
print("="*80)

# %%

