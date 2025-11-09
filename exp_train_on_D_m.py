"""
Experiment 2: Train on Fixed Graph Size with Small Chains, Test on Different Chain Sizes

This experiment:
- Trains on graph_size=30 with chain_size=5 (2 chains of size 5, rest disconnected)
- Tests on chain_size=5, 10, 15 using the same trained model

Results are saved to: results_train_on_D_m.csv
"""
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import random
import numpy as np

import torch

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import train_list_classifier, plot_training_curves, compute_batch_chain_traverse_accuracy, init_simplified_linear_transformer

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

graph_size = 30
train_chain_size = 4
test_chain_sizes = [4, 8, 12]
train_samples = 1000000
val_samples = 10000
num_epochs = 200
batch_size = 10000
starting_vertex = -1  # Random source selection

# Create results directory
RESULTS_DIR = "results/exp_train_on_D_m"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "run_losses"), exist_ok=True)

# %%
# Training
print(f"\n{'='*80}")
print(f"Training on graph_size={graph_size}, chain_size={train_chain_size}")
print(f"{'='*80}")

# connectivity_pattern: [True]*(chain_size-1) + [False] + [True]*(chain_size-1) + [False]*(rest)
# This creates 2 chains of size train_chain_size, rest disconnected
num_connected = 2 * train_chain_size  # Total vertices in the 2 chains
num_disconnected = graph_size - num_connected
connectivity_pattern = (
    [True,] * (train_chain_size - 1) +  # First chain
    [False,] +                           # Break
    [True,] * (train_chain_size - 1) +  # Second chain
    [False,] * num_disconnected          # Rest disconnected
)

print(f"DAG size: {graph_size}")
print(f"Connectivity pattern length: {len(connectivity_pattern)}")
print(f"Connected vertices: {num_connected} (2 chains of {train_chain_size})")
print(f"Disconnected vertices: {num_disconnected}")

# Create tokenizer and training dataset
tokenizer = DAGTokenizer(dag_size=graph_size)

train_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=graph_size,
    num_samples=train_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)

val_ds_train_size = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=graph_size,
    num_samples=val_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)

print(f"Training dataset: {len(train_ds)} samples")
print(f"Validation dataset (train chain size): {len(val_ds_train_size)} samples")

# Train the model
training_results = train_list_classifier(
    train_ds,
    val_ds_train_size,
    num_epochs=num_epochs,
    batch_size=batch_size,
    device=device,
    init_transformer_fn=init_simplified_linear_transformer,
    tmp=1/20,
    layers=1,
    max_new_tokens=20,  # Enough for largest test chain
    masked_some_attention=True,
    mask_out_last_vertex_as_output=True,  # Explicitly pass True
    verbose=True,
    save_weights=False,
)

model = training_results["model"]
train_history = training_results["train_history"]
val_history = training_results["val_history"]

final_train_acc = train_history[-1]['accuracy']

# Plot training curves
plot_training_curves(training_results, save_path=os.path.join(RESULTS_DIR, f"training_curves_train_chain_{train_chain_size}.png"))

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
ax.set_title(f'Validation Accuracy - Train Chain Size {train_chain_size}', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_acc_train_chain_{train_chain_size}.png"), dpi=150)
plt.close()

# Val loss plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs_list, val_loss_list, 'b-o', markersize=3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title(f'Validation Loss - Train Chain Size {train_chain_size}', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "run_losses", f"val_loss_train_chain_{train_chain_size}.png"), dpi=150)
plt.close()

print(f"\nTraining completed!")
print(f"Final training accuracy: {final_train_acc:.2f}%")

# %%
# Testing on different chain sizes
print(f"\n{'='*80}")
print(f"Testing on different chain sizes: {test_chain_sizes}")
print(f"{'='*80}")

results = []

for test_chain_size in test_chain_sizes:
    print(f"\nEvaluating on chain_size={test_chain_size}...")
    
    # Create test dataset with this chain size
    # Same structure: 2 chains of size test_chain_size, rest disconnected
    num_connected_test = 2 * test_chain_size
    num_disconnected_test = graph_size - num_connected_test
    
    if num_disconnected_test < 0:
        print(f"  WARNING: chain_size={test_chain_size} too large for graph_size={graph_size}, skipping")
        continue
    
    connectivity_pattern_test = (
        [True,] * (test_chain_size - 1) +  # First chain
        [False,] +                          # Break
        [True,] * (test_chain_size - 1) +  # Second chain
        [False,] * num_disconnected_test    # Rest disconnected
    )
    
    test_ds = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=graph_size,
        num_samples=val_samples,
        connectivity_pattern=connectivity_pattern_test,
        starting_vertex=starting_vertex,
    )
    
    # Evaluate standard accuracy
    from torch.utils.data import DataLoader
    from utils.misc import collate_fn
    from run_exp_util import sample_until_max_with_logprob_sums, build_sinks_mask_fast, compute_batch_metric_sink_0_1_with_mask
    
    def collate(batch):
        seqs = [b[0] for b in batch]
        tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        padded = collate_fn(seqs, tokenizer.pad_token_id)
        return padded, tgts, metas
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        chain_traverse_correct = 0
        chain_traverse_total = 0
        
        for tbatch in test_loader:
            tinp, ttgt, tmeta = tbatch
            tinp = tinp.to(device)
            ttgt = ttgt.to(device)
            sinks_ids_batch = [m.get('sinks', []) for m in tmeta]
            sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in tmeta]
            setattr(tinp, 'sinks_ids', sinks_ids_batch)
            tgen, _ = sample_until_max_with_logprob_sums(model, tinp, max_new_tokens=20, tokenizer=tokenizer)
            sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, tinp.device)
            
            batch_metric, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, tgen, ttgt, sinks_mask)
            acc = 1.0 - batch_metric.item() if num_valid.item() > 0 else 0.0
            correct += acc * int(num_valid.item())
            total += int(num_valid.item())
            
            # Compute chain traverse accuracy from already-generated sequences (FAST!)
            batch_chain_correct, batch_chain_total = compute_batch_chain_traverse_accuracy(
                tgen, tinp, tmeta, tokenizer
            )
            chain_traverse_correct += batch_chain_correct
            chain_traverse_total += batch_chain_total
    
    test_acc = 100.0 * correct / max(total, 1)
    chain_traverse_acc = 100.0 * chain_traverse_correct / max(chain_traverse_total, 1)
    
    print(f"  Test Accuracy: {test_acc:.2f}%")
    print(f"  Chain Traverse Accuracy: {chain_traverse_acc:.2f}%")
    
    # Print 3 example outputs from test dataset
    print(f"\n  Example outputs for test chain_size={test_chain_size}:")
    import random as _random
    for example_idx in range(3):
        rind = _random.randrange(len(test_ds))
        ex_item = test_ds[rind]
        if isinstance(ex_item, tuple) and len(ex_item) == 3:
            ex_seq_tensor, ex_tgt_id, ex_meta = ex_item
        else:
            ex_seq_tensor, ex_tgt_id = ex_item
            ex_meta = getattr(test_ds, 'samples', None)[rind][2] if hasattr(test_ds, 'samples') else {}
        
        ex_chains = ex_meta.get('chains')
        ex_source = ex_meta.get('source_vertex')
        ex_sinks = ex_meta.get('sinks', [])
        
        ex_decoded_inp = tokenizer.decode(ex_seq_tensor.tolist(), skip_special_tokens=False)
        ex_inp = collate_fn([ex_seq_tensor], tokenizer.pad_token_id).to(device)
        
        # Attach sinks for early stopping during generation
        setattr(ex_inp, 'sinks_ids', [ex_sinks])
        ex_gen, _ = sample_until_max_with_logprob_sums(model, ex_inp, max_new_tokens=20, tokenizer=tokenizer)
        ex_full_ids = ex_gen[0].tolist()
        
        # Extract generated span from after the input prefix to first SINK
        source_pos = len(ex_seq_tensor) - 1
        
        # Find first sink token after source
        sink_token_ids = [tokenizer.token_to_id.get(f'v{v}', tokenizer.unk_token_id) for v in ex_sinks]
        first_sink_pos = None
        for i, token_id in enumerate(ex_full_ids[source_pos + 1:], start=source_pos + 1):
            if token_id in sink_token_ids:
                first_sink_pos = i
                break
        
        # Build generated tail starting from the source vertex
        source_tok_id = ex_seq_tensor[-1].item()
        if first_sink_pos is not None:
            tail_ids = [source_tok_id] + ex_full_ids[source_pos + 1:first_sink_pos + 1]
            pred_tok = ex_full_ids[first_sink_pos]
        else:
            tail_ids = [source_tok_id] + ex_full_ids[source_pos + 1:]
            pred_tok = None
        
        ex_decoded_tail = tokenizer.decode(tail_ids, skip_special_tokens=False)
        
        # Build wanted chain starting from the source vertex
        wanted_chain_vertices = None
        if isinstance(ex_chains, list):
            for ch in ex_chains:
                if ex_source in ch:
                    try:
                        start_idx = ch.index(ex_source)
                    except ValueError:
                        start_idx = 0
                    wanted_chain_vertices = ch[start_idx:]
                    break
        
        if wanted_chain_vertices is not None:
            wanted_chain_ids = [tokenizer.token_to_id.get(f"v{v}", tokenizer.unk_token_id) for v in wanted_chain_vertices]
            wanted_chain_decoded = tokenizer.decode(wanted_chain_ids, skip_special_tokens=False)
        else:
            wanted_chain_decoded = "N/A"
        
        is_correct = (pred_tok == ex_tgt_id)
        print(f"    [{example_idx+1}] Input: {ex_decoded_inp}")
        print(f"        Wanted:  {wanted_chain_decoded}")
        print(f"        Got:     {ex_decoded_tail}")
        print(f"        Pred: {tokenizer.id_to_token.get(pred_tok, 'None')} | Target: {tokenizer.id_to_token[ex_tgt_id]} | {'✓' if is_correct else '✗'}")
    
    results.append({
        'chain_size': test_chain_size,
        'train_accuracy': round(final_train_acc, 2),  # Same for all (trained once)
        'test_accuracy': round(test_acc, 2),
        'chain_traverse': round(chain_traverse_acc, 2),
    })

# %%
# Save results to CSV
output_file = os.path.join(RESULTS_DIR, "results_train_on_D_m.csv")
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
print(f"Trained on graph_size={graph_size}, chain_size={train_chain_size}")
print("-" * 60)
print(f"{'Test Chain Size':<20} {'Train Acc %':<15} {'Test Acc %':<15} {'Chain Traverse %':<15}")
print("-" * 60)
for row in results:
    print(f"{row['chain_size']:<20} {row['train_accuracy']:<15} {row['test_accuracy']:<15} {row['chain_traverse']:<15}")
print("-" * 60)

# %%

