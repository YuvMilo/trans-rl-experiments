"""
Shared utilities for running experiments on linear transformers.
"""
import os
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dag_datasets import DAGTokenizer
from models import LinearTransformer, AttentionConfig, TransformerConfig, SimplifiedLinearTransformer
from utils.misc import collate_fn
from utils.training_utils import get_current_training_weights, SavedWeights


def init_linear_transformer(
    vocab_size: int, 
    tokenizer, 
    device: str, 
    *, 
    tmp: float = 1/20, 
    layers: int = 1, 
    masked_some_attention: bool = True,
    mask_out_last_vertex_as_output: bool = True
) -> LinearTransformer:
    """
    Initialize a linear-attention transformer with configurable number of layers.
    
    Args:
        vocab_size: Size of the vocabulary
        tokenizer: The tokenizer to use
        device: Device to place the model on
        tmp: Temperature parameter for attention
        layers: Number of transformer layers (default: 1)
        masked_some_attention: If True, edges attend to nothing, vertices attend only to edges
        mask_out_last_vertex_as_output: If True, prevent outputting vertices that already appeared
    """
    model = LinearTransformer(
        config=TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=vocab_size * 2 + 16,
            num_layers=layers,
            tokenizer=tokenizer,
            tmp=tmp,
            attention_config=AttentionConfig(),
            values_for_V=True,
            use_cumsum=False,  # Skip cumsum operation for pure attention-based output
            masked_some_attention=masked_some_attention,
            mask_past_verticies=mask_out_last_vertex_as_output,  # Prevent repeating vertices
        )
    ).to(device)
    # Near-zero initialization to encourage stable early training
    small_std = 1e-8
    for layer in model.attention_layers:
        # Set K matrix to identity and freeze it
        with torch.no_grad():
            eye = torch.eye(layer.W_k.weight.data.shape[0], device=device)
            layer.W_k.weight.data.copy_(eye)
        layer.W_k.weight.requires_grad_(False)
        
        # Initialize Q (A) and V with small std
        for name, param in layer.named_parameters():
            if param is not None and param.data is not None and 'W_k' not in name:
                param.data.normal_(mean=0.0, std=small_std)
    return model


def init_simplified_linear_transformer(
    vocab_size: int, 
    tokenizer, 
    device: str, 
    *, 
    tmp: float = 1/20, 
    layers: int = 1, 
    masked_some_attention: bool = True,  # kept for API symmetry; ignored here
    mask_out_last_vertex_as_output: bool = True
):
    """
    Initialize the simplified transformer:
      - K assumed identity (not a parameter)
      - Queries defined only for vertex positions (A: num_vertices x vocab_size)
      - Values map context tokens to vertex logits (vocab_size x num_vertices)
      - Output logits expanded to full vocab with -inf for non-vertex tokens
    """
    model = SimplifiedLinearTransformer(
        config=TransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=vocab_size * 2 + 16,
            num_layers=layers,
            tokenizer=tokenizer,
            tmp=tmp,
            attention_config=AttentionConfig(),
            values_for_V=True,
            use_cumsum=False,
            masked_some_attention=False,
            mask_past_verticies=mask_out_last_vertex_as_output,
        )
    ).to(device)
    return model

def build_sinks_mask_fast(sinks_token_ids_batch: List[List[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Optimized version: takes pre-computed token IDs, no string formatting or dict lookups."""
    batch_size = len(sinks_token_ids_batch)
    sinks_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for tok_id in sinks_token_ids_batch[b]:
            sinks_mask[b, tok_id] = True
    return sinks_mask


def sample_until_max_with_logprob_sums(model, input_ids: torch.Tensor, max_new_tokens: int, tokenizer: DAGTokenizer):
    """
    Optimized batched autoregressive sampling using fixed-size tensors.
    Returns:
        generated: (batch, input_len + max_new_tokens) - fixed size, padded
        path_logprob_sum: (batch,) sum of log-probs up to first stop (sink/EOS)
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Build sink mask once - major optimization
    sinks_ids = getattr(input_ids, 'sinks_ids', None)
    sinks_mask = None
    if sinks_ids is not None:
        sinks_mask = build_sinks_mask(sinks_ids, tokenizer, device)

    # Fixed-size tensor approach - eliminates per-step padding
    input_lens = (input_ids != pad_id).sum(dim=1)  # (B,) - keep on GPU
    max_total_len = input_ids.size(1) + max_new_tokens
    generated = torch.full((batch_size, max_total_len), pad_id, dtype=torch.long, device=device)
    generated[:, :input_ids.size(1)] = input_ids  # copy input
    
    current_pos = input_lens.clone()  # (B,) current write position for each sample
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    path_logprob_sum = torch.zeros(batch_size, device=device)
    batch_idx = torch.arange(batch_size, device=device)

    for _ in range(max_new_tokens):
        # Compute logits only at current positions - more efficient
        all_logits = model(generated).logits  # (B, T, V)
        last_pos = (current_pos - 1).clamp(min=0)  # ensure valid indexing
        logits = all_logits[batch_idx, last_pos, :]  # (B, V)
        
        # Sample and get log-prob in one pass
        dist = torch.distributions.Categorical(logits=logits)
        next_ids = dist.sample()  # (B,)
        step_logprobs = dist.log_prob(next_ids)  # (B,)

        # Accumulate logprobs only for unfinished samples
        active_mask = (~finished).float()
        path_logprob_sum = path_logprob_sum + step_logprobs * active_mask

        # Write next tokens to their positions (vectorized)
        write_mask = ~finished & (current_pos < max_total_len)
        generated[batch_idx[write_mask], current_pos[write_mask]] = next_ids[write_mask]
        current_pos = current_pos + write_mask.long()  # increment position for active samples

        # Check stopping conditions (vectorized, stays on GPU)
        stop_on_eos = (next_ids == eos_id)
        stop_on_sink = torch.zeros_like(stop_on_eos)
        if sinks_mask is not None:
            stop_on_sink = sinks_mask[batch_idx, next_ids]
        finished = finished | stop_on_eos | stop_on_sink

        if finished.all():
            break

    return generated, path_logprob_sum


def build_sinks_mask(sinks_ids_batch: List[List[int]], tokenizer: DAGTokenizer, device: torch.device) -> torch.Tensor:
    """Helper: Build sink mask once for reuse across functions."""
    batch_size = len(sinks_ids_batch)
    sinks_mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for v in sinks_ids_batch[b]:
            tok_id = tokenizer.token_to_id.get(f'v{v}', tokenizer.unk_token_id)
            sinks_mask[b, tok_id] = True
    return sinks_mask


def generate_until_max_greedy(model, input_ids: torch.Tensor, max_new_tokens: int, tokenizer: DAGTokenizer):
    """
    Deterministic (argmax) generation until first sink/EOS or max_new_tokens.
    Used for validation metrics (val_acc and chain traverse).
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    
    # Build sink mask once
    sinks_ids = getattr(input_ids, 'sinks_ids', None)
    sinks_mask = None
    if sinks_ids is not None:
        sinks_mask = build_sinks_mask(sinks_ids, tokenizer, device)
    
    input_lens = (input_ids != pad_id).sum(dim=1)  # (B,)
    max_total_len = input_ids.size(1) + max_new_tokens
    generated = torch.full((batch_size, max_total_len), pad_id, dtype=torch.long, device=device)
    generated[:, :input_ids.size(1)] = input_ids
    
    current_pos = input_lens.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    batch_idx = torch.arange(batch_size, device=device)
    
    for _ in range(max_new_tokens):
        all_logits = model(generated).logits  # (B, T, V)
        last_pos = (current_pos - 1).clamp(min=0)
        logits = all_logits[batch_idx, last_pos, :]  # (B, V)
        
        next_ids = torch.argmax(logits, dim=-1)  # greedy
        
        write_mask = ~finished & (current_pos < max_total_len)
        generated[batch_idx[write_mask], current_pos[write_mask]] = next_ids[write_mask]
        current_pos = current_pos + write_mask.long()
        
        stop_on_eos = (next_ids == eos_id)
        stop_on_sink = torch.zeros_like(stop_on_eos)
        if sinks_mask is not None:
            stop_on_sink = sinks_mask[batch_idx, next_ids]
        finished = finished | stop_on_eos | stop_on_sink
        
        if finished.all():
            break
    
    return generated


def compute_rewards_for_sequences_with_mask(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized version that takes pre-built sinks_mask.
    
    Rewards scheme:
      +1.0 if first sink equals target
       0.0 if wrong sink OR no sink found
    """
    sink_flags = sinks_mask.gather(1, generated)
    any_sink = sink_flags.any(dim=1)
    csum = sink_flags.long().cumsum(dim=1)
    first_mask = csum.eq(1) & sink_flags
    first_pos = first_mask.float().argmax(dim=1)
    pred_tokens = generated.gather(1, first_pos.unsqueeze(1)).squeeze(1)
    correct = (pred_tokens == target_ids) & any_sink
    # +1 for correct, 0 for incorrect or no sink
    rewards = correct.float()
    return rewards, any_sink


def compute_reinforce_loss_batched_with_mask(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    path_logprob_sum: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_mask: torch.Tensor
) -> torch.Tensor:
    """Optimized version that takes pre-built sinks_mask.
    
    Reward scheme:
      +1 for correct; 0 for wrong or no-sink.
    Uses advantage: reward - mean(reward) for variance reduction.
    All samples contribute to gradients (no filtering by validity).
    """
    rewards, valid_mask = compute_rewards_for_sequences_with_mask(tokenizer, generated, target_ids, sinks_mask)
    batch_size = rewards.size(0)
    
    # Compute advantage: reward - baseline (mean reward)
    baseline = rewards.mean()
    advantages = rewards - baseline
    
    loss = -(path_logprob_sum * advantages).sum() / batch_size
    return loss


def compute_batch_metric_sink_0_1_with_mask(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized version that takes pre-built sinks_mask."""
    sink_flags = sinks_mask.gather(1, generated)
    any_sink = sink_flags.any(dim=1)
    csum = sink_flags.long().cumsum(dim=1)
    first_mask = csum.eq(1) & sink_flags
    first_pos = first_mask.float().argmax(dim=1)
    pred_tokens = generated.gather(1, first_pos.unsqueeze(1)).squeeze(1)
    correct = (pred_tokens == target_ids) & any_sink
    valid_mask_f = any_sink.float()
    num_valid = valid_mask_f.sum()
    if num_valid.item() == 0:
        return torch.tensor(0.0, device=generated.device), torch.tensor(0.0, device=generated.device)
    vals = (~correct).float()
    mean_loss = (vals * valid_mask_f).sum() / num_valid
    return mean_loss, num_valid


def compute_batch_chain_traverse_accuracy(
    generated: torch.Tensor,
    input_seqs: torch.Tensor,
    metas: List,
    tokenizer: DAGTokenizer
) -> Tuple[int, int]:
    """
    Compute chain traverse accuracy from already-generated sequences (batched).
    Returns (num_correct, num_total) for this batch.
    
    This is FAST because it uses already-generated sequences and just compares paths.
    """
    batch_size = generated.size(0)
    correct_traversals = 0
    total_valid = 0
    pad_id = tokenizer.pad_token_id
    
    for b in range(batch_size):
        meta = metas[b]
        ex_chains = meta.get('chains')
        ex_source = meta.get('source_vertex')
        ex_sinks = meta.get('sinks', [])
        
        # Get input and generated sequences
        input_seq = input_seqs[b]
        gen_seq = generated[b]
        
        # Find actual input length (before padding)
        input_len = (input_seq != pad_id).sum().item()
        source_pos = input_len - 1  # position of source vertex token
        
        # Find first sink token after source in generated sequence
        sink_token_ids = [tokenizer.token_to_id.get(f'v{v}', tokenizer.unk_token_id) for v in ex_sinks]
        gen_ids = gen_seq.tolist()
        first_sink_pos = None
        for i in range(source_pos + 1, len(gen_ids)):
            if gen_ids[i] in sink_token_ids:
                first_sink_pos = i
                break
        
        if first_sink_pos is None:
            continue  # Skip if no sink found
        
        # Build generated tail (from source to first sink, inclusive)
        source_tok_id = input_seq[source_pos].item()
        tail_ids = [source_tok_id] + gen_ids[source_pos + 1:first_sink_pos + 1]
        
        # Build wanted chain from metadata
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
            
            # Check exact match
            if tail_ids == wanted_chain_ids:
                correct_traversals += 1
            total_valid += 1
    
    return correct_traversals, total_valid


def train_list_classifier(
    train_ds,
    val_ds,
    *,
    init_transformer_fn: Callable = init_linear_transformer,
    num_epochs: int = 20,
    batch_size: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tmp: float = 1/20,
    layers: int = 1,
    num_save_steps: int = 50,
    max_new_tokens: int = 10,
    masked_some_attention: bool = True,
    mask_out_last_vertex_as_output: bool = True,
    verbose: bool = True,
    save_weights: bool = False,
):
    """
    Train a linear transformer on list classification task.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        init_transformer_fn: Function to initialize the transformer
        num_epochs: Number of training epochs (default: 20)
        batch_size: Training batch size
        device: Device to train on
        tmp: Temperature parameter
        layers: Number of transformer layers
        num_save_steps: Steps between weight snapshots
        max_new_tokens: Maximum tokens to generate
        masked_some_attention: Enable attention masking
        mask_out_last_vertex_as_output: Prevent repeating vertices (default: True)
        verbose: Print training progress
        save_weights: Save weight snapshots during training
        
    Returns:
        Dictionary with model, tokenizer, training history, and optionally saved weights
    """
    tokenizer = train_ds.tokenizer
    model = init_transformer_fn(
        tokenizer.vocab_size, 
        tokenizer, 
        device=device, 
        tmp=tmp, 
        layers=layers, 
        masked_some_attention=masked_some_attention,
        mask_out_last_vertex_as_output=mask_out_last_vertex_as_output
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    def collate(batch):
        seqs = [b[0] for b in batch]
        tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        padded = collate_fn(seqs, tokenizer.pad_token_id)
        return padded, tgts, metas

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True, num_workers=4, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False, num_workers=4, persistent_workers=False)

    saved_weights = [] if save_weights else None
    step_counter = 0
    fixed_sample_input = None
    
    train_history = []
    val_history = []

    # Save initial weights if requested
    if save_weights:
        weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
        saved_weights.append(weights_data)

    try:
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False, disable=not verbose)
            train_correct = 0.0
            train_total = 0
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in pbar:
                inp, tgt, metas = batch
                inp = inp.to(device)
                tgt = tgt.to(device)

                if save_weights and fixed_sample_input is None:
                    fixed_sample_input = inp[0:1]

                # Optimized: build sinks data once per batch using pre-computed token IDs
                sinks_ids_batch = [m.get('sinks', []) for m in metas]
                sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in metas]
                
                # Attach sinks to input for early stopping
                setattr(inp, 'sinks_ids', sinks_ids_batch)
                generated, path_logprob_sum = sample_until_max_with_logprob_sums(model, inp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                
                # Build sink mask once for this batch and reuse (fast version)
                sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, inp.device)
                # REINFORCE without advantage (reward in {0,1})
                objective = compute_reinforce_loss_batched_with_mask(tokenizer, generated, path_logprob_sum, tgt, sinks_mask)
                metric_loss, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, generated.detach(), tgt, sinks_mask)

                optim.zero_grad()
                objective.backward()
                optim.step()

                step_counter += 1
                batch_acc = 1.0 - metric_loss.item() if num_valid.item() > 0 else 0.0
                bs_valid = int(num_valid.item())
                train_correct += batch_acc * bs_valid
                train_total += bs_valid
                epoch_loss += objective.item()
                epoch_batches += 1
                
                pbar.set_postfix({"loss": f"{metric_loss.item():.3f}", "acc": f"{batch_acc:.3f}"})

                if save_weights and step_counter % num_save_steps == 0:
                    weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
                    saved_weights.append(weights_data)
            
            train_acc = 100.0 * train_correct / max(train_total, 1)
            train_loss = epoch_loss / max(epoch_batches, 1)
            train_history.append({'epoch': epoch + 1, 'accuracy': train_acc, 'loss': train_loss})

            # Validation
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss_sum = 0.0
                val_batches = 0
                chain_traverse_correct = 0
                chain_traverse_total = 0
                
                for vbatch in val_loader:
                    vinp, vtgt, vmeta = vbatch
                    vinp = vinp.to(device)
                    vtgt = vtgt.to(device)
                    sinks_ids_batch = [m.get('sinks', []) for m in vmeta]
                    sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in vmeta]
                    setattr(vinp, 'sinks_ids', sinks_ids_batch)
                    # For validation: use sampling for loss, but greedy for metrics
                    vgen_sampled, vpath_logprob_sum = sample_until_max_with_logprob_sums(model, vinp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    vgen_greedy = generate_until_max_greedy(model, vinp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, vinp.device)
                    
                    # Validation loss with simple reward (no advantage)
                    val_objective = compute_reinforce_loss_batched_with_mask(tokenizer, vgen_sampled, vpath_logprob_sum, vtgt, sinks_mask)
                    val_loss_sum += val_objective.item()
                    val_batches += 1
                    
                    # Metrics on deterministic (argmax) generations
                    batch_metric, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, vgen_greedy, vtgt, sinks_mask)
                    acc = 1.0 - batch_metric.item() if num_valid.item() > 0 else 0.0
                    correct += acc * int(num_valid.item())
                    total += int(num_valid.item())
                    
                    # Compute chain traverse accuracy from already-generated sequences (FAST!)
                    batch_chain_correct, batch_chain_total = compute_batch_chain_traverse_accuracy(
                        vgen_greedy, vinp, vmeta, tokenizer
                    )
                    chain_traverse_correct += batch_chain_correct
                    chain_traverse_total += batch_chain_total
                
                val_acc = 100.0 * correct / max(total, 1)
                val_loss = val_loss_sum / max(val_batches, 1)
                chain_traverse_acc = 100.0 * chain_traverse_correct / max(chain_traverse_total, 1)
                
                val_history.append({
                    'epoch': epoch + 1, 
                    'accuracy': val_acc, 
                    'loss': val_loss,
                    'chain_traverse': chain_traverse_acc
                })
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%, chain_traverse={chain_traverse_acc:.2f}%, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                    
                    # Print 3 example chains from validation set
                    import random as _random
                    print(f"\n  Example chains from validation:")
                    for example_idx in range(3):
                        rind = _random.randrange(len(val_ds))
                        ex_item = val_ds[rind]
                        if isinstance(ex_item, tuple) and len(ex_item) == 3:
                            ex_seq_tensor, ex_tgt_id, ex_meta = ex_item
                        else:
                            ex_seq_tensor, ex_tgt_id = ex_item
                            ex_meta = getattr(val_ds, 'samples', None)[rind][2] if hasattr(val_ds, 'samples') else {}
                        
                        ex_chains = ex_meta.get('chains')
                        ex_source = ex_meta.get('source_vertex')
                        ex_sinks = ex_meta.get('sinks', [])
                        ex_permutation = ex_meta.get('permutation', [])
                        
                        ex_decoded_inp = tokenizer.decode(ex_seq_tensor.tolist(), skip_special_tokens=False)
                        ex_inp = collate_fn([ex_seq_tensor], tokenizer.pad_token_id).to(device)
                        
                        # Attach sinks for early stopping during generation
                        setattr(ex_inp, 'sinks_ids', [ex_sinks])
                        # Use greedy generation for example outputs to match validation metrics
                        ex_gen = generate_until_max_greedy(model, ex_inp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
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

    except (KeyboardInterrupt, FileNotFoundError, BrokenPipeError) as e:
        if verbose:
            print(f"\nTraining interrupted ({type(e).__name__}). Returning results so far...")

    # Save final weights if requested
    if save_weights:
        weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
        saved_weights.append(weights_data)

    result = {
        "model": model,
        "tokenizer": tokenizer,
        "train_history": train_history,
        "val_history": val_history,
    }
    
    if save_weights:
        result["saved_weights"] = saved_weights
    
    return result


def plot_training_curves(results, save_path="training_curves.png"):
    """
    Plot training curves including loss and chain traverse accuracy.
    
    Args:
        results: Dictionary returned by train_list_classifier with train_history and val_history
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    train_history = results.get("train_history", [])
    val_history = results.get("val_history", [])
    
    if not train_history or not val_history:
        print("No training history to plot!")
        return
    
    epochs = [h['epoch'] for h in train_history]
    train_acc = [h['accuracy'] for h in train_history]
    train_loss = [h['loss'] for h in train_history]
    
    val_acc = [h['accuracy'] for h in val_history]
    val_loss = [h['loss'] for h in val_history]
    val_chain_traverse = [h.get('chain_traverse', 0) for h in val_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training and Validation Accuracy
    axes[0, 0].plot(epochs, train_acc, 'b-o', label='Train Accuracy', markersize=4)
    axes[0, 0].plot(epochs, val_acc, 'r-s', label='Val Accuracy', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Training and Validation Accuracy', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Loss
    axes[0, 1].plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    axes[0, 1].plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Training and Validation Loss', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Chain Traverse Accuracy
    axes[1, 0].plot(epochs, val_chain_traverse, 'g-^', label='Chain Traverse Acc', markersize=4, linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Chain Traverse Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Validation Chain Traverse Accuracy', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Val Loss vs Chain Traverse (combined)
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=4)
    line2 = ax2.plot(epochs, val_chain_traverse, 'g-^', label='Chain Traverse Acc', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12, color='r')
    ax2.set_ylabel('Chain Traverse Accuracy (%)', fontsize=12, color='g')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='g')
    ax1.set_title('Val Loss vs Chain Traverse Accuracy', fontsize=14)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    plt.close()
    
    return fig

