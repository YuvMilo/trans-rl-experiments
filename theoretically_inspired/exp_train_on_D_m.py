"""
Experiment: Out-of-Distribution Generalization

Trains on a fixed graph with small chains (train chain size = 4), then tests the same
model on chain sizes 4, 8, and 12.

Runs for 3 seeds and aggregates results.
Results are saved to: result/theoretically_inspired/exp_train_on_D_m/
"""
import argparse
import os
import csv
import random
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import (
    train_list_classifier, plot_training_curves, compute_batch_chain_traverse_accuracy,
    init_simplified_linear_transformer, generate_dist_for_starting_vertex,
    generate_until_max_greedy, build_sinks_mask_fast, compute_batch_metric_sink_0_1_with_mask,
)
from utils.misc import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Out-of-Distribution Generalization (train on small chains, test on different sizes)"
    )
    attn = parser.add_mutually_exclusive_group()
    attn.add_argument("--softmax", dest="use_softmax", action="store_true", help="Use softmax attention")
    attn.add_argument("--linear", dest="use_softmax", action="store_false", help="Use linear attention (default)")
    parser.set_defaults(use_softmax=False)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Save per-seed results in addition to aggregated CSV")
    return parser.parse_args()


def run_single_seed(seed, train_chain_size, test_chain_sizes, graph_size, train_samples,
                    val_samples, num_epochs, batch_size, amount_of_sub_batches,
                    device, USE_SOFTMAX, save_dir=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*80}")
    print(f"[Seed {seed}] Training on graph_size={graph_size}, chain_size={train_chain_size}")
    print(f"{'='*80}")

    num_connected = 2 * train_chain_size
    num_disconnected = graph_size - num_connected
    connectivity_pattern = (
        [True] * (train_chain_size - 1)
        + [False]
        + [True] * (train_chain_size - 1)
        + [False] * num_disconnected
    )

    starting_vertex_dist_train = generate_dist_for_starting_vertex(
        min_start_index=0, max_start_index=train_chain_size - 2
    )

    tokenizer = DAGTokenizer(dag_size=graph_size)

    train_ds = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=graph_size,
        num_samples=train_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=starting_vertex_dist_train,
        seed=seed,
    )

    val_ds_train_size = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=graph_size,
        num_samples=val_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=starting_vertex_dist_train,
        seed=seed + 1000,
    )

    training_results = train_list_classifier(
        train_ds,
        val_ds_train_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        init_transformer_fn=init_simplified_linear_transformer,
        tmp=1 / 20,
        layers=1,
        max_new_tokens=max(test_chain_sizes) + 1,
        masked_some_attention=False,
        mask_out_last_vertex_as_output=False,
        verbose=True,
        save_weights=False,
        use_softmax_attention=USE_SOFTMAX,
        amount_of_sub_batches=amount_of_sub_batches,
        early_stop_at_100_acc=True,
    )

    model = training_results["model"]
    train_history = training_results["train_history"]
    val_history = training_results["val_history"]
    final_train_acc = train_history[-1]['accuracy']

    if save_dir:
        plot_training_curves(
            training_results,
            save_path=os.path.join(save_dir, f"training_curves_train_chain_{train_chain_size}.png"),
        )

    print(f"\n{'='*80}")
    print(f"[Seed {seed}] Testing on different chain sizes: {test_chain_sizes}")
    print(f"{'='*80}")

    results = []
    all_data = {
        'train_chain_size': train_chain_size,
        'graph_size': graph_size,
        'train_history': train_history,
        'val_history': val_history,
        'test_results': {},
    }

    for test_chain_size in test_chain_sizes:
        print(f"\nEvaluating on chain_size={test_chain_size}...")

        num_connected_test = 2 * test_chain_size
        num_disconnected_test = graph_size - num_connected_test

        if num_disconnected_test < 0:
            print(f"  WARNING: chain_size={test_chain_size} too large for graph_size={graph_size}, skipping")
            continue

        connectivity_pattern_test = (
            [True] * (test_chain_size - 1)
            + [False]
            + [True] * (test_chain_size - 1)
            + [False] * num_disconnected_test
        )

        starting_vertex_dist_test = generate_dist_for_starting_vertex(
            min_start_index=0, max_start_index=test_chain_size - 2
        )

        test_ds = PermutationListClassificationDataset(
            dag_tokenizer=tokenizer,
            dag_size=graph_size,
            num_samples=val_samples,
            connectivity_pattern=connectivity_pattern_test,
            starting_vertex=starting_vertex_dist_test,
            seed=seed + 2000 + test_chain_size,
        )

        def collate(batch):
            seqs = [b[0] for b in batch]
            tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
            metas = [b[2] for b in batch]
            padded = collate_fn(seqs, tokenizer.pad_token_id)
            return padded, tgts, metas

        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False
        )

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

                tgen = generate_until_max_greedy(model, tinp, max_new_tokens=20, tokenizer=tokenizer)
                sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, tinp.device)

                batch_metric, num_valid = compute_batch_metric_sink_0_1_with_mask(
                    tokenizer, tgen, ttgt, sinks_mask
                )
                acc = 1.0 - batch_metric.item() if num_valid.item() > 0 else 0.0
                correct += acc * int(num_valid.item())
                total += int(num_valid.item())

                batch_chain_correct, batch_chain_total = compute_batch_chain_traverse_accuracy(
                    tgen, tinp, tmeta, tokenizer
                )
                chain_traverse_correct += batch_chain_correct
                chain_traverse_total += batch_chain_total

        test_acc = 100.0 * correct / max(total, 1)
        chain_traverse_acc = 100.0 * chain_traverse_correct / max(chain_traverse_total, 1)

        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Chain Traverse Accuracy: {chain_traverse_acc:.2f}%")

        results.append({
            'chain_size': test_chain_size,
            'train_accuracy': round(final_train_acc, 2),
            'test_accuracy': round(test_acc, 2),
            'chain_traverse': round(chain_traverse_acc, 2),
        })

        all_data['test_results'][test_chain_size] = {
            'test_accuracy': test_acc,
            'chain_traverse_accuracy': chain_traverse_acc,
        }

    if save_dir:
        with open(os.path.join(save_dir, "all_training_test_data.pkl"), 'wb') as f:
            pickle.dump(all_data, f)

        output_file = os.path.join(save_dir, "results_train_on_D_m.csv")
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['chain_size', 'train_accuracy', 'test_accuracy', 'chain_traverse']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    return results


def main():
    args = parse_args()
    USE_SOFTMAX = args.use_softmax
    verbose_saving = args.verbose

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Attention type: {'softmax' if USE_SOFTMAX else 'linear'}")

    train_chain_size = 4
    test_chain_sizes = [4, 8, 12]
    graph_size = max(test_chain_sizes) * 2

    if not USE_SOFTMAX:
        train_samples = 2000000
        val_samples = 1000
        num_epochs = 200
        batch_size = 200000
        amount_of_sub_batches = 80
    else:
        train_samples = 1000000
        val_samples = 1000
        num_epochs = 300
        batch_size = 50000
        amount_of_sub_batches = 30

    seeds = [42, 43, 44]

    dir_suffix = "" if USE_SOFTMAX else "_linear"
    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "result", "theoretically_inspired",
        f"exp_train_on_D_m{dir_suffix}",
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_seed_results = {}

    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"RUNNING SEED {seed}")
        print(f"{'#'*80}")

        if verbose_saving:
            seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            os.makedirs(os.path.join(seed_dir, "run_losses"), exist_ok=True)
        else:
            seed_dir = None

        all_seed_results[seed] = run_single_seed(
            seed, train_chain_size, test_chain_sizes, graph_size, train_samples,
            val_samples, num_epochs, batch_size, amount_of_sub_batches,
            device, USE_SOFTMAX, seed_dir,
        )

    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*80}")

    aggregated = {cs: {'train_accuracy': [], 'test_accuracy': [], 'chain_traverse': []}
                  for cs in test_chain_sizes}

    for seed, results in all_seed_results.items():
        for row in results:
            cs = row['chain_size']
            aggregated[cs]['train_accuracy'].append(row['train_accuracy'])
            aggregated[cs]['test_accuracy'].append(row['test_accuracy'])
            aggregated[cs]['chain_traverse'].append(row['chain_traverse'])

    final_results = []
    for cs in test_chain_sizes:
        final_results.append({
            'chain_size': cs,
            'train_accuracy_mean': round(np.mean(aggregated[cs]['train_accuracy']), 2),
            'train_accuracy_std': round(np.std(aggregated[cs]['train_accuracy']), 2),
            'test_accuracy_mean': round(np.mean(aggregated[cs]['test_accuracy']), 2),
            'test_accuracy_std': round(np.std(aggregated[cs]['test_accuracy']), 2),
            'chain_traverse_mean': round(np.mean(aggregated[cs]['chain_traverse']), 2),
            'chain_traverse_std': round(np.std(aggregated[cs]['chain_traverse']), 2),
        })

    output_file = os.path.join(RESULTS_DIR, "results_aggregated.csv")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'chain_size', 'train_accuracy_mean', 'train_accuracy_std',
            'test_accuracy_mean', 'test_accuracy_std',
            'chain_traverse_mean', 'chain_traverse_std',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_results:
            writer.writerow(row)

    print(f"\nAggregated results saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY (averaged over {len(seeds)} seeds)")
    print(f"Trained on graph_size={graph_size}, chain_size={train_chain_size}")
    print(f"{'='*80}")
    print(f"{'Test Chain Size':<20} {'Train Acc %':<20} {'Test Acc %':<20} {'Chain Traverse %':<20}")
    print("-" * 80)
    for row in final_results:
        train_str = f"{row['train_accuracy_mean']:.2f} ± {row['train_accuracy_std']:.2f}"
        test_str = f"{row['test_accuracy_mean']:.2f} ± {row['test_accuracy_std']:.2f}"
        chain_str = f"{row['chain_traverse_mean']:.2f} ± {row['chain_traverse_std']:.2f}"
        print(f"{row['chain_size']:<20} {train_str:<20} {test_str:<20} {chain_str:<20}")
    print("-" * 80)
    print(f"\nAll results saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
