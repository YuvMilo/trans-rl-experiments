"""
Experiment: Solving Complex Tasks Requires Training On Simple Tasks

Trains models with changing task difficulty (cumulative starting-position configurations)
and generates loss plots showing how training on simpler tasks enables solving harder ones.

Results are saved to: result/theoretically_inspired/exp_vanish_grad/
"""
import argparse
import os
import random
import numpy as np
import pickle

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import (
    train_list_classifier, plot_training_curves,
    init_simplified_linear_transformer, generate_dist_for_starting_vertex,
)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['mathtext.fontset'] = 'dejavusans'


def parse_args():
    parser = argparse.ArgumentParser(description="Solving Complex Tasks Requires Training On Simple Tasks")
    attn = parser.add_mutually_exclusive_group()
    attn.add_argument("--softmax", dest="use_softmax", action="store_true", help="Use softmax attention")
    attn.add_argument("--linear", dest="use_softmax", action="store_false", help="Use linear attention (default)")
    parser.set_defaults(use_softmax=False)
    return parser.parse_args()


def main():
    args = parse_args()
    USE_SOFTMAX = args.use_softmax

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Attention type: {'softmax' if USE_SOFTMAX else 'linear'}")

    chain_size = 5
    dag_size = chain_size * 2
    num_epochs = 1000

    if not USE_SOFTMAX:
        train_samples = 2000000
        val_samples = 10000
        batch_size = 200000
        amount_of_sub_batches = 2
    else:
        train_samples = 1000000
        val_samples = 10000
        batch_size = 50000
        amount_of_sub_batches = 1

    dir_suffix = "" if USE_SOFTMAX else "_linear"
    RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "result", "theoretically_inspired",
        f"exp_vanish_grad{dir_suffix}",
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "run_losses"), exist_ok=True)

    starting_position_configs = [list(range(i + 1)) for i in range(chain_size - 1)]

    print(f"\n{'='*80}")
    print(f"Solving Complex Tasks Requires Training On Simple Tasks")
    print(f"Chain size: {chain_size}, Graph size: {dag_size}")
    print(f"Testing starting position configs: {starting_position_configs}")
    print(f"{'='*80}")

    connectivity_pattern = [True] * (chain_size - 1) + [False] + [True] * (chain_size - 1)
    tokenizer = DAGTokenizer(dag_size=dag_size)

    print(f"\nCreating shared validation dataset with starting position 0 only...")
    val_ds_shared = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=dag_size,
        num_samples=val_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=0,
    )
    print(f"Shared validation dataset: {len(val_ds_shared)} samples (starting position 0 only)")

    all_results = {}

    for config_idx, starting_pos_config in enumerate(starting_position_configs):
        config_label = f"{config_idx + 1}"
        config_str = str(starting_pos_config)

        print(f"\n{'='*80}")
        print(f"Training with starting_positions={config_str} (Config {config_label})")
        print(f"Validating on starting position 0 only (shared validation set)")
        print(f"{'='*80}")

        train_starting_vertex = generate_dist_for_starting_vertex(
            min_start_index=starting_pos_config[0],
            max_start_index=starting_pos_config[-1],
        )

        train_ds = PermutationListClassificationDataset(
            dag_tokenizer=tokenizer,
            dag_size=dag_size,
            num_samples=train_samples,
            connectivity_pattern=connectivity_pattern,
            starting_vertex=train_starting_vertex,
        )

        tmp = 1 / 20 if USE_SOFTMAX else 1 / 5
        training_results = train_list_classifier(
            train_ds,
            val_ds_shared,
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
            early_stop_at_100_acc=False,
        )

        train_history = training_results["train_history"]
        val_history = training_results["val_history"]

        all_results[config_idx] = {
            'train_history': train_history,
            'val_history': val_history,
            'starting_positions': starting_pos_config,
            'config_label': config_label,
        }

        plot_training_curves(
            training_results,
            save_path=os.path.join(RESULTS_DIR, f"training_curves_config_{config_label}.png"),
        )

        epochs_list = [h['epoch'] for h in val_history]
        val_acc_list = [h['accuracy'] for h in val_history]
        val_loss_list = [h['loss'] for h in val_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_list, val_acc_list, 'r-o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'Validation Accuracy - Config {config_label}: {config_str}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, "run_losses", f"val_acc_config_{config_label}.png"), dpi=150
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_list, val_loss_list, 'b-o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title(f'Validation Loss - Config {config_label}: {config_str}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, "run_losses", f"val_loss_config_{config_label}.png"), dpi=150
        )
        plt.close()

        print(f"\nCompleted config {config_label}: {config_str}")
        print(f"Final val_acc: {val_history[-1]['accuracy']:.2f}%")

    with open(os.path.join(RESULTS_DIR, "all_results.pkl"), 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: {os.path.join(RESULTS_DIR, 'all_results.pkl')}")

    # Publication-quality plots
    print(f"\n{'='*80}")
    print("Creating publication-quality plots...")
    print(f"{'='*80}")

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    linestyles = [':', '-.', '--', '-', ':', '-.', '--', '-', ':', '--']

    if len(all_results) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for config_idx in sorted(all_results.keys()):
        result = all_results[config_idx]
        epochs_list = [h['epoch'] for h in result['val_history']]
        val_acc_list = [h['accuracy'] for h in result['val_history']]
        k_value = len(result['starting_positions'])
        ax.plot(epochs_list, val_acc_list,
                linestyle=linestyles[config_idx % len(linestyles)],
                linewidth=2.5, label=f'$k={k_value}$',
                color=colors[config_idx], alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=22, fontweight='normal')
    ax.set_ylabel('Test Accuracy (%)', fontsize=22, fontweight='normal')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18, loc='best', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "test_accuracy_paper.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for config_idx in sorted(all_results.keys()):
        result = all_results[config_idx]
        epochs_list = [h['epoch'] for h in result['val_history']]
        val_chain_traverse = [h.get('chain_traverse', 0) for h in result['val_history']]
        k_value = len(result['starting_positions'])
        ax.plot(epochs_list, val_chain_traverse,
                linestyle=linestyles[config_idx % len(linestyles)],
                linewidth=2.5, label=f'$k={k_value}$',
                color=colors[config_idx], alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=22, fontweight='normal')
    ax.set_ylabel('Chain Traverse Accuracy (%)', fontsize=22, fontweight='normal')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=18, loc='best', frameon=True, fancybox=False,
              edgecolor='black', framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='-')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "chain_traverse_paper.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Chain traverse plot saved to: {output_path}")
    plt.close()

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
        print(f"Train on {starting_pos_config} / Validate on pos 0{' '*10} {final_acc:.2f}%{' '*8} {final_chain_traverse:.2f}%")
    print("-" * 70)
    print(f"\nAll results saved in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
