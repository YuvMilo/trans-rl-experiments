"""
Experiment: Emergence of Efficient Reasoning

Trains the model on graphs with 2 chains of different sizes (D = 4, 8, 12) and evaluates
both standard accuracy and exact chain traversal accuracy.

Runs for 3 seeds and aggregates results.
Results are saved to: result/theoretically_inspired/exp_train_on_D/
"""
import argparse
import os
import csv
import random
import numpy as np
import pickle

import torch

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import (
    train_list_classifier, plot_training_curves,
    init_simplified_linear_transformer, generate_dist_for_starting_vertex,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Emergence of Efficient Reasoning (train on different chain sizes)")
    attn = parser.add_mutually_exclusive_group()
    attn.add_argument("--softmax", dest="use_softmax", action="store_true", help="Use softmax attention")
    attn.add_argument("--linear", dest="use_softmax", action="store_false", help="Use linear attention (default)")
    parser.set_defaults(use_softmax=False)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Save per-seed results in addition to aggregated CSV")
    return parser.parse_args()


def run_single_seed(seed, chain_sizes, train_samples, val_samples, num_epochs,
                    batch_size, amount_of_sub_batches, device, USE_SOFTMAX, save_dir=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    results = []
    all_training_data = {}

    for chain_size in chain_sizes:
        print(f"\n{'='*80}")
        print(f"[Seed {seed}] Training on chain_size = {chain_size} (graph_size = {chain_size * 2})")
        print(f"{'='*80}")

        dag_size = chain_size * 2
        connectivity_pattern = [True] * (chain_size - 1) + [False] + [True] * (chain_size - 1)

        starting_vertex_dist = generate_dist_for_starting_vertex(
            min_start_index=0, max_start_index=chain_size - 2
        )

        tokenizer = DAGTokenizer(dag_size=dag_size)

        train_ds = PermutationListClassificationDataset(
            dag_tokenizer=tokenizer,
            dag_size=dag_size,
            num_samples=train_samples,
            connectivity_pattern=connectivity_pattern,
            starting_vertex=starting_vertex_dist,
            seed=seed,
        )

        val_ds = PermutationListClassificationDataset(
            dag_tokenizer=tokenizer,
            dag_size=dag_size,
            num_samples=val_samples,
            connectivity_pattern=connectivity_pattern,
            starting_vertex=starting_vertex_dist,
            seed=seed + 1000,
        )

        tmp = 1 / 20 if USE_SOFTMAX else 1 / 5
        training_results = train_list_classifier(
            train_ds,
            val_ds,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            init_transformer_fn=init_simplified_linear_transformer,
            tmp=tmp,
            layers=1,
            max_new_tokens=chain_size + 1,
            masked_some_attention=False,
            mask_out_last_vertex_as_output=False,
            verbose=True,
            save_weights=False,
            use_softmax_attention=USE_SOFTMAX,
            amount_of_sub_batches=amount_of_sub_batches,
            early_stop_at_100_acc=True,
        )

        train_history = training_results["train_history"]
        val_history = training_results["val_history"]

        final_train_acc = train_history[-1]['accuracy']
        final_val_acc = val_history[-1]['accuracy']
        chain_traverse_acc = val_history[-1]['chain_traverse']

        results.append({
            'chain_size': chain_size,
            'train_accuracy': round(final_train_acc, 2),
            'test_accuracy': round(final_val_acc, 2),
            'chain_traverse': round(chain_traverse_acc, 2),
        })

        all_training_data[chain_size] = {
            'train_history': train_history,
            'val_history': val_history,
            'chain_size': chain_size,
        }

        if save_dir:
            plot_training_curves(
                training_results,
                save_path=os.path.join(save_dir, f"training_curves_chain_{chain_size}.png"),
            )

    if save_dir:
        with open(os.path.join(save_dir, "all_training_data.pkl"), 'wb') as f:
            pickle.dump(all_training_data, f)

        output_file = os.path.join(save_dir, "results_train_on_D.csv")
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

    chain_sizes = [4, 8, 12][::-1]
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
        f"exp_train_on_D{dir_suffix}",
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
            seed, chain_sizes, train_samples, val_samples, num_epochs,
            batch_size, amount_of_sub_batches, device, USE_SOFTMAX, seed_dir,
        )

    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print(f"{'='*80}")

    aggregated = {cs: {'train_accuracy': [], 'test_accuracy': [], 'chain_traverse': []}
                  for cs in chain_sizes}

    for seed, results in all_seed_results.items():
        for row in results:
            cs = row['chain_size']
            aggregated[cs]['train_accuracy'].append(row['train_accuracy'])
            aggregated[cs]['test_accuracy'].append(row['test_accuracy'])
            aggregated[cs]['chain_traverse'].append(row['chain_traverse'])

    final_results = []
    for cs in chain_sizes:
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
    print(f"{'='*80}")
    print(f"{'Chain Size':<15} {'Train Acc %':<20} {'Test Acc %':<20} {'Chain Traverse %':<20}")
    print("-" * 75)
    for row in final_results:
        train_str = f"{row['train_accuracy_mean']:.2f} ± {row['train_accuracy_std']:.2f}"
        test_str = f"{row['test_accuracy_mean']:.2f} ± {row['test_accuracy_std']:.2f}"
        chain_str = f"{row['chain_traverse_mean']:.2f} ± {row['chain_traverse_std']:.2f}"
        print(f"{row['chain_size']:<15} {train_str:<20} {test_str:<20} {chain_str:<20}")
    print("-" * 75)
    print(f"\nAll results saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
