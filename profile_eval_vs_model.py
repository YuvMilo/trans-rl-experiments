# %%
"""
Profile per-batch time split between:
- Sampling (autoregressive generation)
- Validation loss computation (REINFORCE objective)
- Validation metric computation (sink 0/1 metric)
- Chain traverse accuracy computation

Usage:
    python profile_eval_vs_model.py
    python profile_eval_vs_model.py --device cuda --batches 10 --batch-size 2000 --chain-size 4 --train-samples 20000 --val-samples 2000
"""
import os
import argparse
import time
from contextlib import contextmanager

import torch

from dag_datasets import DAGTokenizer, PermutationListClassificationDataset
from run_exp_util import (
    sample_until_max_with_logprob_sums,
    build_sinks_mask_fast,
    compute_reinforce_loss_batched_with_mask,
    compute_batch_metric_sink_0_1_with_mask,
    compute_batch_chain_traverse_accuracy,
    init_linear_transformer,
)
from utils.misc import collate_fn
from torch.utils.data import DataLoader


@contextmanager
def cuda_timer():
    """Accurate GPU timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"{end - start:.6f}", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--chain-size", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=40000)
    parser.add_argument("--val-samples", type=int, default=4000)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--tmp", type=float, default=1.0/20.0)
    parser.add_argument("--max-new-tokens", type=int, default=9)
    parser.add_argument("--masked-some-attention", action="store_true", default=False)
    parser.add_argument("--mask-out-last-vertex", action="store_true", default=True)
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # Dataset config
    chain_size = args.chain_size
    dag_size = chain_size * 2
    connectivity_pattern = [True] * (chain_size - 1) + [False] + [True] * (chain_size - 1)

    # Tokenizer and datasets
    tokenizer = DAGTokenizer(dag_size=dag_size)
    train_ds = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=dag_size,
        num_samples=args.train_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=-1,
    )
    val_ds = PermutationListClassificationDataset(
        dag_tokenizer=tokenizer,
        dag_size=dag_size,
        num_samples=args.val_samples,
        connectivity_pattern=connectivity_pattern,
        starting_vertex=-1,
    )

    def collate(batch):
        seqs = [b[0] for b in batch]
        tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        padded = collate_fn(seqs, tokenizer.pad_token_id)
        return padded, tgts, metas

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate, num_workers=0)

    # Model
    model = init_linear_transformer(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        device=device,
        tmp=args.tmp,
        layers=args.layers,
        masked_some_attention=args.masked_some_attention,
        mask_out_last_vertex_as_output=args.mask_out_last_vertex,
    )
    model.eval()

    # Timers
    t_sampling = 0.0
    t_loss = 0.0
    t_metric = 0.0
    t_chain = 0.0

    batches_done = 0
    print("\nProfiling per-batch timings (seconds):")
    print("batch_idx | sampling | loss | metric | chain")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break
        vinp, vtgt, vmeta = batch
        vinp = vinp.to(device)
        vtgt = vtgt.to(device)

        # Attach sinks_ids for early stopping and precompute sinks_token_ids
        sinks_ids_batch = [m.get('sinks', []) for m in vmeta]
        sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in vmeta]
        setattr(vinp, 'sinks_ids', sinks_ids_batch)

        # Sampling
        start = time.perf_counter()
        vgen, vpath_logprob_sum = sample_until_max_with_logprob_sums(model, vinp, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_sampling += time.perf_counter() - start

        # Build sinks mask (fast)
        sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, vinp.device)

        # Loss
        start = time.perf_counter()
        _ = compute_reinforce_loss_batched_with_mask(tokenizer, vgen, vpath_logprob_sum, vtgt, sinks_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_loss += time.perf_counter() - start

        # Metric (sink 0/1)
        start = time.perf_counter()
        _metric_loss, _num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, vgen, vtgt, sinks_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_metric += time.perf_counter() - start

        # Chain traverse accuracy
        start = time.perf_counter()
        _ = compute_batch_chain_traverse_accuracy(vgen, vinp, vmeta, tokenizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_chain += time.perf_counter() - start

        print(f"{batch_idx:9d} | {t_sampling:8.4f} | {t_loss:6.4f} | {t_metric:6.4f} | {t_chain:6.4f}")
        batches_done += 1

    if batches_done == 0:
        print("\nNo batches profiled (increase --val-samples or decrease --batch-size).")
        return

    # Report averages and proportions
    avg_sampling = t_sampling / batches_done
    avg_loss = t_loss / batches_done
    avg_metric = t_metric / batches_done
    avg_chain = t_chain / batches_done
    total = avg_sampling + avg_loss + avg_metric + avg_chain
    print("\nAverages per batch (seconds):")
    print(f"Sampling: {avg_sampling:.6f}  ({100.0 * avg_sampling / total:.1f}%)")
    print(f"Loss:     {avg_loss:.6f}  ({100.0 * avg_loss / total:.1f}%)")
    print(f"Metric:   {avg_metric:.6f}  ({100.0 * avg_metric / total:.1f}%)")
    print(f"Chain:    {avg_chain:.6f}  ({100.0 * avg_chain / total:.1f}%)")
    print(f"Total:    {total:.6f}  (100.0%)")


if __name__ == "__main__":
    main()



# %%
