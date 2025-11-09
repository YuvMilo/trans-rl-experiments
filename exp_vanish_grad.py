"""
Experiment 4: Vanishing Gradient Analysis

This experiment tests how the starting position in the chain affects learning.
Trains models with different starting positions (0 to chain_size-2) and plots
test accuracy across all runs.

Results are saved to: results/vanish_grad/
"""
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
import numpy as np
import pickle

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

chain_size = 6
dag_size = chain_size * 2  # 2 chains
train_samples = 1000000
val_samples = 10000
num_epochs = 200
batch_size = 10000

# Create results directory
RESULTS_DIR = "results/vanish_grad"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "run_losses"), exist_ok=True)

# Starting position configurations to test
# [0], [0,1], [0,1,2] (cumulative lists up to chain_size-2)
starting_position_configs = []
for i in range(chain_size - 1):
    config = list(range(i + 1))
    starting_position_configs.append(config)

print(f"\n{'='*80}")
print(f"Vanishing Gradient Experiment")
print(f"Chain size: {chain_size}, Graph size: {dag_size}")
print(f"Testing starting position configs: {starting_position_configs}")
print(f"  Config 1: {starting_position_configs[0]} (train on position 0 only)")
if len(starting_position_configs) > 1:
    print(f"  Config 2: {starting_position_configs[1]} (train on positions 0 or 1)")
if len(starting_position_configs) > 2:
    print(f"  Config 3: {starting_position_configs[2]} (train on positions 0, 1, or 2)")
print(f"{'='*80}")

# Store results for all runs
all_results = {}

# Create 2 chains of size chain_size (shared pattern)
connectivity_pattern = [True,] * (chain_size - 1) + [False,] + [True,] * (chain_size - 1)

# Create tokenizer (shared)
tokenizer = DAGTokenizer(dag_size=dag_size)

# Create SHARED validation dataset with starting_position=0
# All models will be evaluated on this same validation set
print(f"\nCreating shared validation dataset with starting_position=0...")
val_ds_shared = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=val_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=0,  # All models validate on starting_position=0
)
print(f"Shared validation dataset: {len(val_ds_shared)} samples (starting_position=0)")

# %%
for config_idx, starting_pos_config in enumerate(starting_position_configs):
    config_label = f"{config_idx+1}"  # Label as 1, 2, 3, ...
    config_str = str(starting_pos_config)
    
    print(f"\n{'='*80}")
    print(f"Training with starting_positions={config_str} (will be labeled as Config {config_label} in plots)")
    print(f"Validating on starting_position=0 (shared validation set)")
    print(f"{'='*80}")
    
    # Create training dataset with this starting position configuration
    # If it's a single-element list, unwrap it to an integer
    if len(starting_pos_config) == 1:
        train_starting_vertex = starting_pos_config[0]
    else:
        train_starting_vertex = starting_pos_config  # Pass list directly
    
    train_ds = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=dag_size,
        num_samples=train_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=train_starting_vertex,  # Can be int or list
    )
    
    print(f"Training dataset: {len(train_ds)} samples (starting_positions={config_str})")
    print(f"Validation dataset: {len(val_ds_shared)} samples (starting_position=0, SHARED)")
    
    # Train the model (validate on shared dataset)
    training_results = train_list_classifier(
        train_ds,
        val_ds_shared,  # Use shared validation dataset
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        init_transformer_fn=init_simplified_linear_transformer,
        tmp=1/5,
        layers=1,
        max_new_tokens=chain_size + 5,
        masked_some_attention=False,
        mask_out_last_vertex_as_output=True,
        verbose=True,
        save_weights=False,
    )
    
    model = training_results["model"]
    train_history = training_results["train_history"]
    val_history = training_results["val_history"]
    
    # Store results for this configuration
    all_results[config_idx] = {
        'train_history': train_history,
        'val_history': val_history,
        'starting_positions': starting_pos_config,
        'config_label': config_label,
    }
    
    # Plot training curves for this run
    plot_training_curves(training_results, save_path=os.path.join(RESULTS_DIR, f"training_curves_config_{config_label}.png"))
    
    # Save individual val accuracy and val loss plots
    epochs_list = [h['epoch'] for h in val_history]
    val_acc_list = [h['accuracy'] for h in val_history]
    val_loss_list = [h['loss'] for h in val_history]
    
    # Val accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_list, val_acc_list, 'r-o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Validation Accuracy - Config {config_label}: {config_str}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_acc_config_{config_label}.png"), dpi=150)
    plt.close()
    
    # Val loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_list, val_loss_list, 'b-o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title(f'Validation Loss - Config {config_label}: {config_str}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_loss_config_{config_label}.png"), dpi=150)
    plt.close()
    
    print(f"\nCompleted config {config_label}: {config_str}")
    print(f"Final val_acc: {val_history[-1]['accuracy']:.2f}%")

# %%
# Save all results as pickle
pickle_path = os.path.join(RESULTS_DIR, "all_results.pkl")
with open(pickle_path, 'wb') as f:
    pickle.dump(all_results, f)
print(f"\nAll results saved to: {pickle_path}")

# %%
# Plot all test accuracies on the same plot
print(f"\n{'='*80}")
print("Creating combined test accuracy plot...")
print(f"{'='*80}")

fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(starting_position_configs)))

for config_idx in range(len(starting_position_configs)):
    result = all_results[config_idx]
    val_history = result['val_history']
    epochs_list = [h['epoch'] for h in val_history]
    val_acc_list = [h['accuracy'] for h in val_history]
    starting_pos_config = result['starting_positions']
    
    # Create label showing the config
    label = f'Train on {starting_pos_config}'
    ax.plot(epochs_list, val_acc_list, '-o', markersize=2, linewidth=2, 
            label=label, color=colors[config_idx])

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title(f'Test Accuracy vs Training Starting Position\n(Chain Size {chain_size}, All Validated on Starting Position 0)', fontsize=16)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()

combined_plot_path = os.path.join(RESULTS_DIR, "combined_test_accuracy.png")
plt.savefig(combined_plot_path, dpi=200, bbox_inches='tight')
print(f"Combined test accuracy plot saved to: {combined_plot_path}")
plt.show()

# %%
# Plot all chain traverse accuracies on the same plot
print(f"\n{'='*80}")
print("Creating combined chain traverse accuracy plot...")
print(f"{'='*80}")

fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.plasma(np.linspace(0, 1, len(starting_position_configs)))

for config_idx in range(len(starting_position_configs)):
    result = all_results[config_idx]
    val_history = result['val_history']
    epochs_list = [h['epoch'] for h in val_history]
    val_chain_traverse = [h.get('chain_traverse', 0) for h in val_history]
    starting_pos_config = result['starting_positions']
    
    # Create label showing the config
    label = f'Train on {starting_pos_config}'
    ax.plot(epochs_list, val_chain_traverse, '-^', markersize=2, linewidth=2, 
            label=label, color=colors[config_idx])

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Chain Traverse Accuracy (%)', fontsize=14)
ax.set_title(f'Chain Traverse Accuracy vs Training Starting Position\n(Chain Size {chain_size}, All Validated on Starting Position 0)', fontsize=16)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()

combined_chain_plot_path = os.path.join(RESULTS_DIR, "combined_chain_traverse_accuracy.png")
plt.savefig(combined_chain_plot_path, dpi=200, bbox_inches='tight')
print(f"Combined chain traverse plot saved to: {combined_chain_plot_path}")
plt.show()

# %%
# Print summary
print(f"\n{'='*80}")
print("SUMMARY: Final Test Accuracies (All Validated on Starting Position 0)")
print(f"{'='*80}")
print(f"{'Training Config':<35} {'Test Acc %':<15} {'Chain Traverse %':<20}")
print("-" * 70)
for config_idx in range(len(starting_position_configs)):
    result = all_results[config_idx]
    val_history = result['val_history']
    starting_pos_config = result['starting_positions']
    final_acc = val_history[-1]['accuracy']
    final_chain_traverse = val_history[-1].get('chain_traverse', 0)
    print(f"Train on {starting_pos_config} / Validate on [0]{' '*10} {final_acc:.2f}%{' '*8} {final_chain_traverse:.2f}%")
print("-" * 70)

print(f"\nAll results saved in: {RESULTS_DIR}/")
print(f"  - all_results.pkl (loadable pickle)")
print(f"  - combined_test_accuracy.png (test accuracy curves)")
print(f"  - combined_chain_traverse_accuracy.png (chain traverse curves)")
print(f"  - run_losses/ (individual accuracy/loss plots)")
print(f"\nNote: All models trained on different starting positions but validated on starting_position=0")

# %%

