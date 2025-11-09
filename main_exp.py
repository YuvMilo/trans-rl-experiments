# %%
"""
Main experiment script for training a linear transformer on permutation list classification tasks.
"""
# Standard libraries
import os
import random
from typing import List, Tuple

# Third-party libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from dag_datasets import DAGTokenizer, PermutationListClassificationDataset, ListClassificationDataset
from models import LinearTransformer, AttentionConfig, TransformerConfig
from utils.misc import collate_fn
from utils.training_utils import get_current_training_weights, SavedWeights
from visualization.training_dynamics import visualize_linear_training_dynamics


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

# ============================================================================
# Configuration
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dag_size = 5
connectivity_pattern = [True, True, False, True]  # Connects: 0->1, 1->2, 2 disconnected from 3, 3->4
starting_vertex = 2  # -1 for random selection, or permutation index (0 to dag_size-1) to use permutation[index] as source
demo_num_samples = 5
layers_d = 1  # Use single layer for linear transformer
use_masked_attention = False  # Set to True to enable attention masking: edges attend to nothing, vertices attend only to edges

print(f"Using connectivity pattern: {connectivity_pattern}")
print(f"Starting vertex: {starting_vertex} ({'random' if starting_vertex == -1 else f'use permutation[{starting_vertex}] as source'})")
print(f"This creates the following structure: 0->1->2  3->4 (with permutations)")

# ============================================================================
# Demo Dataset Generation
# ============================================================================
print("Generating PermutationListClassificationDataset demo...")
tokenizer = DAGTokenizer(dag_size=dag_size)
demo_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=demo_num_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)
print(f"Generated demo dataset with {len(demo_ds)} samples")
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
for i in range(min(demo_num_samples, 5)):
    demo_item = demo_ds[i]
    if isinstance(demo_item, tuple) and len(demo_item) == 3:
        seq, tgt, meta = demo_item
    else:
        seq, tgt = demo_item
        meta = demo_ds.samples[i][2] if hasattr(demo_ds, 'samples') else {}
    decoded = tokenizer.decode(seq.tolist(), skip_special_tokens=False)
    edges = meta.get('edges')
    chains = meta.get('chains')
    source_vertex = meta.get('source_vertex')
    target_vertex = meta.get('target_vertex')
    permutation = meta.get('permutation')
    print(f"Sample {i}: tokens={decoded}")
    print(f"  permutation={permutation}")
    print(f"  edges={edges}\n chains={chains}\n length={len(chains)}\n source=v{source_vertex}\n target=v{target_vertex}\n tgt_token={tokenizer.id_to_token[tgt]}")


# ============================================================================
# Model Initialization
# ============================================================================
def init_linear_transformer(vocab_size: int, tokenizer, device: str, *, tmp: float = 1/20, layers: int = layers_d, masked_some_attention: bool = True) -> LinearTransformer:
    """
    Initialize a linear-attention transformer with configurable number of layers.
    
    Args:
        vocab_size: Size of the vocabulary
        tokenizer: The tokenizer to use
        device: Device to place the model on
        tmp: Temperature parameter for attention
        layers: Number of transformer layers (default: 1)
        masked_some_attention: If True, edges attend to nothing, vertices attend only to edges
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
            masked_some_attention=masked_some_attention,  # Enable attention masking
        )
    ).to(device)
    # Near-zero initialization to encourage stable early training
    small_std = 1e-4
    for layer in model.attention_layers:
        # Set K matrix to identity and freeze it
        with torch.no_grad():
            eye = torch.eye(layer.W_k.weight.data.shape[0], device=device)
            layer.W_k.weight.data.copy_(eye)
        layer.W_k.weight.requires_grad_(False)
        
        # Initialize other parameters with small std
        for name, param in layer.named_parameters():
            if param is not None and param.data is not None and 'W_k' not in name:
                param.data.normal_(mean=0.0, std=small_std)
                param.data.abs_()
    return model


# ============================================================================
# Sampling Functions
# ============================================================================
def sample_until_eos_with_logprobs(model, input_ids: torch.Tensor, max_new_tokens: int, tokenizer: DAGTokenizer):
    """
    Autoregressively sample tokens; return sequences and log-probs of sampled tokens.
    Returns:
        sequences: (batch, seq_len_in + <=max_new_tokens)
        logprobs_list: list of length batch with lists of per-step log-probs (tensors)
    """
    generated = input_ids.clone()
    batch_size = input_ids.size(0)
    eos_id = tokenizer.eos_token_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    logprobs_list = [[] for _ in range(batch_size)]
    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        next_ids = torch.multinomial(log_probs.exp(), num_samples=1)  # (B,1)
        # collect logprobs for chosen ids
        step_logprobs = log_probs.gather(1, next_ids).squeeze(1)  # (B,)
        for b in range(batch_size):
            if not finished[b]:
                logprobs_list[b].append(step_logprobs[b])
        generated = torch.cat([generated, next_ids], dim=1)
        finished |= (next_ids.squeeze(1) == eos_id)
        if finished.all():
            break
    return generated, logprobs_list


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

    for _ in range(max_new_tokens):
        # Compute logits only at current positions - more efficient
        all_logits = model(generated).logits  # (B, T, V)
        batch_idx = torch.arange(batch_size, device=device)
        last_pos = (current_pos - 1).clamp(min=0)  # ensure valid indexing
        logits = all_logits[batch_idx, last_pos, :]  # (B, V)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        next_ids = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(1)  # (B,)
        step_logprobs = log_probs.gather(1, next_ids.unsqueeze(1)).squeeze(1)  # (B,)

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


def build_sinks_mask_fast(sinks_token_ids_batch: List[List[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Optimized version: takes pre-computed token IDs, no string formatting or dict lookups."""
    batch_size = len(sinks_token_ids_batch)
    sinks_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for tok_id in sinks_token_ids_batch[b]:
            sinks_mask[b, tok_id] = True
    return sinks_mask


# ============================================================================
# Metric and Loss Functions
# ============================================================================
def compute_batch_metric_sink_0_1(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_ids_batch: List[List[int]]
) -> torch.Tensor:
    """Return mean 0/1 metric: 0 if first sink equals target, else 1; 1 if no sink."""
    device = generated.device
    batch_size, seq_len = generated.shape
    # Use helper to build sinks mask
    sinks_mask = build_sinks_mask(sinks_ids_batch, tokenizer, device)
    # For each position, mark if token is sink
    sink_flags = sinks_mask.gather(1, generated)
    any_sink = sink_flags.any(dim=1)
    # first sink position via cumulative trick
    csum = sink_flags.long().cumsum(dim=1)
    first_mask = csum.eq(1) & sink_flags
    first_pos = first_mask.float().argmax(dim=1)
    pred_tokens = generated.gather(1, first_pos.unsqueeze(1)).squeeze(1)
    valid = any_sink
    correct = (pred_tokens == target_ids) & valid
    vals = (~correct).float()
    return vals.mean()


def compute_reinforce_loss(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    logprobs_list: List[List[torch.Tensor]],
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Simple REINFORCE loss: -reward * sum_t log p(a_t)
    reward = 1 if EOS exists and token before EOS equals target; else 0.
    """
    batch_size = generated.size(0)
    eos_id = tokenizer.eos_token_id
    losses: List[torch.Tensor] = []
    for b in range(batch_size):
        seq = generated[b].tolist()
        try:
            eos_pos = seq.index(eos_id)
        except ValueError:
            reward = 0.0
        else:
            if eos_pos == 0:
                reward = 0.0
            else:
                pred_id = seq[eos_pos - 1]
                reward = 1.0 if pred_id == target_ids[b].item() else 0.0
        if len(logprobs_list[b]) == 0:
            # no steps sampled; zero contribution
            losses.append(torch.tensor(0.0, device=generated.device))
        else:
            path_logprob = torch.stack(logprobs_list[b]).sum()
            losses.append(-path_logprob * reward)
    return torch.stack(losses).mean()


def compute_rewards_for_sequences(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_ids_batch: List[List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (rewards, valid_mask).
    rewards: 1 if first sink equals target; else 0. 0 if no sink.
    valid_mask: True if a sink was found (i.e., this sample is valid), else False.
    """
    device = generated.device
    # Use helper to build sinks mask
    sinks_mask = build_sinks_mask(sinks_ids_batch, tokenizer, device)
    sink_flags = sinks_mask.gather(1, generated)
    any_sink = sink_flags.any(dim=1)
    csum = sink_flags.long().cumsum(dim=1)
    first_mask = csum.eq(1) & sink_flags
    first_pos = first_mask.float().argmax(dim=1)
    pred_tokens = generated.gather(1, first_pos.unsqueeze(1)).squeeze(1)
    correct = (pred_tokens == target_ids) & any_sink
    rewards = correct.float()
    return rewards, any_sink


def compute_reinforce_loss_batched(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    path_logprob_sum: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_ids_batch: List[List[int]]
) -> torch.Tensor:
    rewards, valid_mask = compute_rewards_for_sequences(tokenizer, generated, target_ids, sinks_ids_batch)
    valid_mask_f = valid_mask.float()
    # Sum contributions only over valid samples, normalize by num_valid to avoid diluting gradients
    num_valid = valid_mask_f.sum()
    if num_valid.item() == 0:
        return torch.tensor(0.0, device=generated.device)
    loss = -(path_logprob_sum * rewards * valid_mask_f).sum() / num_valid
    return loss


def compute_reinforce_loss_batched_with_mask(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    path_logprob_sum: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_mask: torch.Tensor
) -> torch.Tensor:
    """Optimized version that takes pre-built sinks_mask."""
    rewards, valid_mask = compute_rewards_for_sequences_with_mask(tokenizer, generated, target_ids, sinks_mask)
    valid_mask_f = valid_mask.float()
    num_valid = valid_mask_f.sum()
    if num_valid.item() == 0:
        return torch.tensor(0.0, device=generated.device)
    loss = -(path_logprob_sum * rewards * valid_mask_f).sum() / num_valid
    return loss


def compute_rewards_for_sequences_with_mask(
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
    rewards = correct.float()
    return rewards, any_sink


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


def compute_batch_metric_sink_0_1_ignore_invalid(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_ids_batch: List[List[int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean 0/1 loss over valid samples, num_valid).
    A sample is valid if it produced any sink; invalid samples are ignored.
    """
    device = generated.device
    # Use helper to build sinks mask
    sinks_mask = build_sinks_mask(sinks_ids_batch, tokenizer, device)
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
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    vals = (~correct).float()
    mean_loss = (vals * valid_mask_f).sum() / num_valid
    return mean_loss, num_valid


# ============================================================================
# Training Function
# ============================================================================
def train_list_classifier(
    train_ds,
    val_ds,
    *,
    init_transformer_fn,
    num_epochs: int = 5,
    batch_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tmp: float = 1/20,
    layers: int = 1,
    num_save_steps: int = 200,
    max_new_tokens: int = 8,
    masked_some_attention: bool = False,
):
    tokenizer = train_ds.tokenizer
    model = init_transformer_fn(tokenizer.vocab_size, tokenizer, device=device, tmp=tmp, layers=layers, masked_some_attention=masked_some_attention)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    def collate(batch):
        seqs = [b[0] for b in batch]
        tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        padded = collate_fn(seqs, tokenizer.pad_token_id)
        return padded, tgts, metas

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False, num_workers=0, persistent_workers=False)

    saved_weights = []
    step_counter = 0
    fixed_sample_input = None

    # Save initial weights
    weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
    saved_weights.append(weights_data)

    try:
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
            train_correct = 0.0
            train_total = 0
            recent_accuracies = []  # Track last 50 accuracies for rolling average
            for batch in pbar:
                inp, tgt, metas = batch
                inp = inp.to(device)
                tgt = tgt.to(device)

                if fixed_sample_input is None:
                    fixed_sample_input = inp[0:1]

                # Optimized: build sinks data once per batch using pre-computed token IDs
                sinks_ids_batch = [m.get('sinks', []) for m in metas]  # for sampling (backward compat)
                sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in metas]  # for fast mask building
                
                # Attach sinks to input for early stopping
                setattr(inp, 'sinks_ids', sinks_ids_batch)
                generated, path_logprob_sum = sample_until_max_with_logprob_sums(model, inp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                
                # Build sink mask once for this batch and reuse (fast version)
                sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, inp.device)
                objective = compute_reinforce_loss_batched_with_mask(tokenizer, generated, path_logprob_sum, tgt, sinks_mask)
                metric_loss, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, generated.detach(), tgt, sinks_mask)

                optim.zero_grad()
                objective.backward()
                optim.step()

                step_counter += 1
                # batch and rolling accuracy (using 0/1 metric)
                batch_acc = 1.0 - metric_loss.item() if num_valid.item() > 0 else 0.0
                bs_valid = int(num_valid.item())
                train_correct += batch_acc * bs_valid
                train_total += bs_valid
                
                # Track last 50 accuracies for rolling average
                recent_accuracies.append(batch_acc)
                if len(recent_accuracies) > 50:
                    recent_accuracies.pop(0)
                rolling_avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                
                pbar.set_postfix({"loss": f"{metric_loss.item():.3f}", "acc": f"{batch_acc:.3f}", "avg50": f"{rolling_avg_acc:.3f}"})

                if step_counter % num_save_steps == 0:
                    weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
                    saved_weights.append(weights_data)

            # Validation per epoch (print one generated sample and correctness)
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss_sum = 0.0
                val_batches = 0
                for vbatch in val_loader:
                    vinp, vtgt, vmeta = vbatch
                    vinp = vinp.to(device)
                    vtgt = vtgt.to(device)
                    # Optimized validation: build sinks once per batch using pre-computed token IDs
                    sinks_ids_batch = [m.get('sinks', []) for m in vmeta]  # for sampling (backward compat)
                    sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in vmeta]  # for fast mask building
                    setattr(vinp, 'sinks_ids', sinks_ids_batch)
                    vgen, vpath_logprob_sum = sample_until_max_with_logprob_sums(model, vinp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, vinp.device)
                    
                    # Compute validation loss (same as training objective)
                    val_objective = compute_reinforce_loss_batched_with_mask(tokenizer, vgen, vpath_logprob_sum, vtgt, sinks_mask)
                    val_loss_sum += val_objective.item()
                    val_batches += 1
                    
                    batch_metric, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, vgen, vtgt, sinks_mask)
                    acc = 1.0 - batch_metric.item() if num_valid.item() > 0 else 0.0
                    correct += acc * int(num_valid.item())
                    total += int(num_valid.item())
                val_acc = correct / max(total, 1)
                val_loss = val_loss_sum / max(val_batches, 1)
                # Print detailed examples from the raw val dataset (with chains/meta)
                import random as _random
                print(f"Epoch {epoch+1}: val_acc={val_acc:.3f}, val_loss={val_loss:.4f}")
                for _ in range(3):  # Show fewer examples for cleaner output
                    rind = _random.randrange(len(val_ds))
                    ex_item = val_ds[rind]
                    if isinstance(ex_item, tuple) and len(ex_item) == 3:
                        ex_seq_tensor, ex_tgt_id, ex_meta = ex_item
                    else:
                        ex_seq_tensor, ex_tgt_id = ex_item
                        ex_meta = getattr(val_ds, 'samples', None)[rind][2] if hasattr(val_ds, 'samples') else {}
                    ex_edges = ex_meta.get('edges')
                    ex_chains = ex_meta.get('chains')
                    ex_source = ex_meta.get('source_vertex')
                    ex_target = ex_meta.get('target_vertex')
                    ex_permutation = ex_meta.get('permutation', [])
                    ex_decoded_inp = tokenizer.decode(ex_seq_tensor.tolist(), skip_special_tokens=False)
                    ex_inp = collate_fn([ex_seq_tensor], tokenizer.pad_token_id).to(device)
                    # Attach sinks for early stopping during generation
                    ex_sinks = ex_meta.get('sinks', [])
                    setattr(ex_inp, 'sinks_ids', [ex_sinks])
                    ex_gen, _ = sample_until_max_with_logprob_sums(model, ex_inp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    ex_full_ids = ex_gen[0].tolist()
                    
                    # Extract generated span from after the input prefix to first SINK (not EOS!)
                    source_pos = len(ex_seq_tensor) - 1  # position of source vertex token in the input prefix
                    
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
                        pred_tok = ex_full_ids[first_sink_pos]  # The sink token itself is the prediction
                    else:
                        tail_ids = [source_tok_id] + ex_full_ids[source_pos + 1:]
                        pred_tok = None
                    
                    ex_decoded_full = tokenizer.decode(ex_full_ids, skip_special_tokens=False)
                    ex_decoded_tail = tokenizer.decode(tail_ids, skip_special_tokens=False)
                    is_correct = (pred_tok == ex_tgt_id)
                    print(f"  Input seq: {ex_decoded_inp}")
                    print(f"  Permutation: {ex_permutation}")
                    print(f"  Generated full: {ex_decoded_full}")
                    print(f"  Generated tail: {ex_decoded_tail}")
                    # Build and print wanted chain starting from the source vertex (no SEP)
                    wanted_chain_vertices = None
                    if isinstance(ex_chains, list):
                        for ch in ex_chains:
                            if ex_source in ch:
                                # take subchain starting from the source vertex position
                                try:
                                    start_idx = ch.index(ex_source)
                                except ValueError:
                                    start_idx = 0
                                wanted_chain_vertices = ch[start_idx:]
                                break
                    if wanted_chain_vertices is not None:
                        wanted_chain_ids = [tokenizer.token_to_id.get(f"v{v}", tokenizer.unk_token_id) for v in wanted_chain_vertices]
                        wanted_chain_decoded = tokenizer.decode(wanted_chain_ids, skip_special_tokens=False)
                        print(f"  Wanted chain: {wanted_chain_decoded}")
                    else:
                        print(f"  Wanted chain: N/A")
                    print(f"  Pred: {tokenizer.id_to_token.get(pred_tok, 'None')}  Target: {tokenizer.id_to_token[ex_tgt_id]}  Correct={is_correct}")

    except (KeyboardInterrupt, FileNotFoundError, BrokenPipeError) as e:
        print(f"\nTraining interrupted ({type(e).__name__}). Returning results so far...")

    # Save final weights
    weights_data = get_current_training_weights(model, tokenizer, step_counter, fixed_sample_input)
    saved_weights.append(weights_data)

    result = {
        "model": model,
        "tokenizer": tokenizer,
        "saved_weights": saved_weights,
    }
    return result

# %%
# ============================================================================
# Main Training Execution
# ============================================================================

train_samples = 1000000  # Smaller dataset for faster experimentation
val_samples = 1000

chain_size = 5
connectivity_pattern = [True,]*chain_size+[False,]+[True,]*chain_size
dag_size = len(connectivity_pattern)+1
starting_vertex = -1  # -1 for random selection, or permutation index (0 to dag_size-1) to use permutation[index] as source
layers_d = 1  # Single layer for linear transformer
use_masked_attention = False

print(f"Using connectivity pattern: {connectivity_pattern}")
print(f"Starting vertex: {starting_vertex} ({'random' if starting_vertex == -1 else f'use permutation[{starting_vertex}] as source'})")
print(f"This creates chains: vertex_perm[0]->vertex_perm[1]->vertex_perm[2]->vertex_perm[3]  vertex_perm[4]->vertex_perm[5] (isolated)")

print("Generating PermutationListClassificationDataset for training...")
tokenizer = DAGTokenizer(dag_size=dag_size)
demo_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=5,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)
for i in range(5):
    demo_item = demo_ds[i]
    if isinstance(demo_item, tuple) and len(demo_item) == 3:
        seq, tgt, meta = demo_item
    else:
        seq, tgt = demo_item
        meta = demo_ds.samples[i][2] if hasattr(demo_ds, 'samples') else {}
    decoded = tokenizer.decode(seq.tolist(), skip_special_tokens=False)
    permutation = meta.get('permutation', [])
    print(f"Sample {i}: tokens={decoded}, target={tokenizer.id_to_token[tgt]}, perm={permutation}")

train_ds = PermutationListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=train_samples,
    connectivity_pattern=connectivity_pattern,
    starting_vertex=starting_vertex,
)
# Use general random graph dataset for validation to test generalization
val_ds = ListClassificationDataset(
    dag_tokenizer=tokenizer,
    dag_size=dag_size,
    num_samples=val_samples,
    p_link=0.7,  # Standard probability for random connections
)

print(f"\nDataset summary:")
print(f"Training: {len(train_ds)} samples with fixed connectivity pattern {connectivity_pattern}")
print(f"Validation: {len(val_ds)} samples with random graphs (p_link=0.7)")
print(f"This tests generalization from structured to random graphs.")

results = train_list_classifier(
    train_ds,
    val_ds,
    init_transformer_fn=init_linear_transformer,
    num_epochs=10,
    batch_size=2000,  # Adjusted for smaller dataset
    device=device,
    tmp=1/20,
    layers=layers_d,
    num_save_steps=1,
    max_new_tokens=10,
    masked_some_attention=use_masked_attention,
)
# %%
# Visualizations (weight evolution)
size1 = 50
size2 = 50
if results and "saved_weights" in results:
    print(f"Saved {len(results['saved_weights'])} weight snapshots during training")
    # Example: show first layer V and QK matrices at first and last snapshot
    first = results['saved_weights'][0]
    last = results['saved_weights'][-1]
    print("First snapshot step:", first.step, "V shape:", first.V_matrix.shape, "QK shape:", first.QK_matrix.shape)
    print("Last snapshot step:", last.step, "V shape:", last.V_matrix.shape, "QK shape:", last.QK_matrix.shape)

    # Plot heatmaps for V, K, Q at init and final (only first 40x40 indices)
    import matplotlib.pyplot as plt
    import numpy as np

    V_init = first.V_matrix
    V_final = last.V_matrix
    # Q and K are stored explicitly in SavedWeights now
    Q_init = first.Q_matrix.transpose(1, 0) if first.Q_matrix is not None else None
    Q_final = last.Q_matrix.transpose(1, 0) if last.Q_matrix is not None else None
    K_init = first.K_matrix if first.K_matrix is not None else None
    K_final = last.K_matrix if last.K_matrix is not None else None

    # Build figure dynamically based on availability of Q/K
    matrices = [("V", V_init, V_final)]
    if Q_init is not None and Q_final is not None:
        matrices.append(("Q", Q_init, Q_final))
    if K_init is not None and K_final is not None:
        matrices.append(("K", K_init, K_final))

    nrows = len(matrices)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    fig.suptitle('Weight Matrices: Initialization vs After Training', fontsize=14)

    for idx, (name, m_init, m_final) in enumerate(matrices):
        # Only plot the first 40x40 indices for each matrix
        m_init_crop = m_init[:size1, :size2]
        m_final_crop = m_final[:size1, :size2]

        im1 = axes[idx, 0].imshow(m_init_crop, cmap='RdBu_r', aspect='auto')
        axes[idx, 0].set_title(f'{name} - Init (first {size1}x{size2})')
        axes[idx, 0].set_xlabel('Output dim')
        axes[idx, 0].set_ylabel('Input dim')
        plt.colorbar(im1, ax=axes[idx, 0])

        im2 = axes[idx, 1].imshow(m_final_crop, cmap='RdBu_r', aspect='auto')
        axes[idx, 1].set_title(f'{name} - Final (first {size1}x{size2})')
        axes[idx, 1].set_xlabel('Output dim')
        axes[idx, 1].set_ylabel('Input dim')
        plt.colorbar(im2, ax=axes[idx, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# # Video visualization (training dynamics GIF)
# VIZ_DIR = os.path.join("results_RLCoT", "training_visualizations_permutation")
# os.makedirs(VIZ_DIR, exist_ok=True)
# if results and "saved_weights" in results:
#     print("Creating linear transformer training dynamics visualizations (GIF)...")
#     visualize_linear_training_dynamics(results, save_dir=VIZ_DIR)
#     print("Linear transformer training dynamics visualization saved to:", VIZ_DIR)
# else:
#     print("No weight evolution data found; skipping video visualization")

# print(tokenizer.id_to_token)


# %%
