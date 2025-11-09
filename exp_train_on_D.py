"""
Experiment 1: Train on Different Chain Sizes (D = 10, 20, 30)

This experiment trains the model on graphs with 2 chains of different sizes
and evaluates both standard accuracy and exact chain traversal accuracy.

Results are saved to: results_train_on_D.csv
"""
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import random
import numpy as np

import torch

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

chain_sizes = [4,6,8]
train_samples = 1000000
val_samples = 10000
num_epochs = 200
batch_size = 10000
starting_vertex = -1  # Random source selection

# Create results directory
RESULTS_DIR = "results/train_on_D"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "run_losses"), exist_ok=True)

results = []

# %%
for chain_size in chain_sizes:
    print(f"\n{'='*80}")
    print(f"Training on chain_size = {chain_size} (graph_size = {chain_size * 2})")
    print(f"{'='*80}")
    
    # Graph has 2 chains of size chain_size
    # connectivity_pattern: [True]*(chain_size-1) + [False] + [True]*(chain_size-1)
    dag_size = chain_size * 2
    connectivity_pattern = [True,] * (chain_size - 1) + [False,] + [True,] * (chain_size - 1)
    
    print(f"DAG size: {dag_size}")
    print(f"Connectivity pattern length: {len(connectivity_pattern)}")
    print(f"This creates 2 chains of size {chain_size} each")
    
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
        starting_vertex=starting_vertex,
    )
    
    print(f"Training dataset: {len(train_ds)} samples")
    print(f"Validation dataset: {len(val_ds)} samples")
    
    # Train the model
    training_results = train_list_classifier(
        train_ds,
        val_ds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        init_transformer_fn=init_simplified_linear_transformer,
        tmp=1,
        layers=1,
        max_new_tokens=chain_size + 5,
        masked_some_attention=True,
        mask_out_last_vertex_as_output=True,  # Explicitly pass True
        verbose=True,
        save_weights=False,
    )
    
    model = training_results["model"]
    train_history = training_results["train_history"]
    val_history = training_results["val_history"]
    
    # Plot training curves
    plot_training_curves(training_results, save_path=os.path.join(RESULTS_DIR, f"training_curves_chain_{chain_size}.png"))
    
    # Save individual val accuracy and val loss plots
    import matplotlib.pyplot as plt
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
    
    # Get final epoch accuracies
    final_train_acc = train_history[-1]['accuracy']
    final_val_acc = val_history[-1]['accuracy']
    
    # Get chain traverse accuracy from final epoch (already computed during validation)
    chain_traverse_acc = val_history[-1]['chain_traverse']
    
    print(f"\nResults for chain_size={chain_size}:")
    print(f"  Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Test Accuracy: {final_val_acc:.2f}%")
    print(f"  Chain Traverse Accuracy: {chain_traverse_acc:.2f}%")
    
    results.append({
        'chain_size': chain_size,
        'train_accuracy': round(final_train_acc, 2),
        'test_accuracy': round(final_val_acc, 2),
        'chain_traverse': round(chain_traverse_acc, 2),
    })

# %%
# Save results to CSV
output_file = os.path.join(RESULTS_DIR, "results_train_on_D.csv")
print(f"\n{'='*80}")
print(f"Saving results to {output_file}")
print(f"{'='*80}")

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['chain_size', 'train_accuracy', 'test_accuracy', 'chain_traverse']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\nResults saved successfully!")
print("\nSummary:")
print("-" * 60)
print(f"{'Chain Size':<15} {'Train Acc %':<15} {'Test Acc %':<15} {'Chain Traverse %':<15}")
print("-" * 60)
for row in results:
    print(f"{row['chain_size']:<15} {row['train_accuracy']:<15} {row['test_accuracy']:<15} {row['chain_traverse']:<15}")
print("-" * 60)

# %%

