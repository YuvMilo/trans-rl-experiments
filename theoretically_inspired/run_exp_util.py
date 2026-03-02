"""
Shared utilities for running experiments on linear transformers.
"""
from typing import List, Tuple, Callable, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dag_datasets import DAGTokenizer
from models import AttentionConfig, TransformerConfig, SimplifiedLinearTransformer
from utils.misc import collate_fn


def generate_dist_for_starting_vertex(min_start_index: int, max_start_index: int) -> Dict[int, float]:
    """Generate a uniform probability distribution over starting vertex indices."""
    if min_start_index < 0:
        raise ValueError(f"min_start_index must be >= 0, got {min_start_index}")
    if max_start_index < min_start_index:
        raise ValueError(f"max_start_index must be >= min_start_index")
    
    n = max_start_index - min_start_index + 1
    uniform_prob = 1.0 / n
    return {min_start_index + i: uniform_prob for i in range(n)}


def init_simplified_linear_transformer(
    vocab_size: int, 
    tokenizer, 
    device: str, 
    *, 
    tmp: float = 1/20, 
    layers: int = 1, 
    masked_some_attention: bool = True,
    mask_out_last_vertex_as_output: bool = True,
    use_softmax_attention: bool = False
):
    """Initialize the simplified transformer."""
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
            use_softmax_attention=use_softmax_attention,
        )
    ).to(device)
    return model


def build_sinks_mask_fast(sinks_token_ids_batch: List[List[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Build sink mask from pre-computed token IDs."""
    batch_size = len(sinks_token_ids_batch)
    sinks_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for tok_id in sinks_token_ids_batch[b]:
            sinks_mask[b, tok_id] = True
    return sinks_mask


def build_sinks_mask(sinks_ids_batch: List[List[int]], tokenizer: DAGTokenizer, device: torch.device) -> torch.Tensor:
    """Build sink mask from vertex IDs."""
    batch_size = len(sinks_ids_batch)
    sinks_mask = torch.zeros((batch_size, tokenizer.vocab_size), dtype=torch.bool, device=device)
    for b in range(batch_size):
        for v in sinks_ids_batch[b]:
            tok_id = tokenizer.token_to_id.get(f'v{v}', tokenizer.unk_token_id)
            sinks_mask[b, tok_id] = True
    return sinks_mask


def sample_until_max_with_logprob_sums(model, input_ids: torch.Tensor, max_new_tokens: int, tokenizer: DAGTokenizer):
    """Batched autoregressive sampling with log-prob accumulation."""
    device = input_ids.device
    batch_size = input_ids.size(0)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    sinks_ids = getattr(input_ids, 'sinks_ids', None)
    sinks_mask = None
    if sinks_ids is not None:
        sinks_mask = build_sinks_mask(sinks_ids, tokenizer, device)

    input_lens = (input_ids != pad_id).sum(dim=1)
    max_total_len = input_ids.size(1) + max_new_tokens
    generated = torch.full((batch_size, max_total_len), pad_id, dtype=torch.long, device=device)
    generated[:, :input_ids.size(1)] = input_ids
    
    current_pos = input_lens.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    path_logprob_sum = torch.zeros(batch_size, device=device)
    batch_idx = torch.arange(batch_size, device=device)

    for _ in range(max_new_tokens):
        all_logits = model(generated).logits
        last_pos = (current_pos - 1).clamp(min=0)
        logits = all_logits[batch_idx, last_pos, :]
        
        dist = torch.distributions.Categorical(logits=logits)
        next_ids = dist.sample()
        step_logprobs = dist.log_prob(next_ids)

        active_mask = (~finished).float()
        path_logprob_sum = path_logprob_sum + step_logprobs * active_mask

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

    return generated, path_logprob_sum


def generate_until_max_greedy(model, input_ids: torch.Tensor, max_new_tokens: int, tokenizer: DAGTokenizer):
    """Deterministic (argmax) generation until first sink/EOS or max_new_tokens."""
    device = input_ids.device
    batch_size = input_ids.size(0)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    
    sinks_ids = getattr(input_ids, 'sinks_ids', None)
    sinks_mask = None
    if sinks_ids is not None:
        sinks_mask = build_sinks_mask(sinks_ids, tokenizer, device)
    
    input_lens = (input_ids != pad_id).sum(dim=1)
    max_total_len = input_ids.size(1) + max_new_tokens
    generated = torch.full((batch_size, max_total_len), pad_id, dtype=torch.long, device=device)
    generated[:, :input_ids.size(1)] = input_ids
    
    current_pos = input_lens.clone()
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    batch_idx = torch.arange(batch_size, device=device)
    
    for _ in range(max_new_tokens):
        all_logits = model(generated).logits
        last_pos = (current_pos - 1).clamp(min=0)
        logits = all_logits[batch_idx, last_pos, :]
        
        next_ids = torch.argmax(logits, dim=-1)
        
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
    """Compute rewards: +1.0 if first sink equals target, 0.0 otherwise."""
    sink_flags = sinks_mask.gather(1, generated)
    any_sink = sink_flags.any(dim=1)
    csum = sink_flags.long().cumsum(dim=1)
    first_mask = csum.eq(1) & sink_flags
    first_pos = first_mask.float().argmax(dim=1)
    pred_tokens = generated.gather(1, first_pos.unsqueeze(1)).squeeze(1)
    correct = (pred_tokens == target_ids) & any_sink
    rewards = correct.float()
    return rewards, any_sink


def compute_reinforce_loss_batched_with_mask(
    tokenizer: DAGTokenizer,
    generated: torch.Tensor,
    path_logprob_sum: torch.Tensor,
    target_ids: torch.Tensor,
    sinks_mask: torch.Tensor
) -> torch.Tensor:
    """Compute REINFORCE loss with advantage baseline."""
    rewards, _ = compute_rewards_for_sequences_with_mask(tokenizer, generated, target_ids, sinks_mask)
    batch_size = rewards.size(0)
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
    """Compute batch accuracy metric."""
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
    """Compute chain traverse accuracy from generated sequences."""
    batch_size = generated.size(0)
    correct_traversals = 0
    total_valid = 0
    pad_id = tokenizer.pad_token_id
    
    for b in range(batch_size):
        meta = metas[b]
        ex_chains = meta.get('chains')
        ex_source = meta.get('source_vertex')
        ex_sinks = meta.get('sinks', [])
        
        input_seq = input_seqs[b]
        gen_seq = generated[b]
        
        input_len = (input_seq != pad_id).sum().item()
        source_pos = input_len - 1
        
        sink_token_ids = [tokenizer.token_to_id.get(f'v{v}', tokenizer.unk_token_id) for v in ex_sinks]
        gen_ids = gen_seq.tolist()
        first_sink_pos = None
        for i in range(source_pos + 1, len(gen_ids)):
            if gen_ids[i] in sink_token_ids:
                first_sink_pos = i
                break
        
        if first_sink_pos is None:
            continue
        
        source_tok_id = input_seq[source_pos].item()
        tail_ids = [source_tok_id] + gen_ids[source_pos + 1:first_sink_pos + 1]
        
        wanted_chain_vertices = None
        if isinstance(ex_chains, list):
            for ch in ex_chains:
                if ex_source in ch:
                    start_idx = ch.index(ex_source) if ex_source in ch else 0
                    wanted_chain_vertices = ch[start_idx:]
                    break
        
        if wanted_chain_vertices is not None:
            wanted_chain_ids = [tokenizer.token_to_id.get(f"v{v}", tokenizer.unk_token_id) for v in wanted_chain_vertices]
            if tail_ids == wanted_chain_ids:
                correct_traversals += 1
            total_valid += 1
    
    return correct_traversals, total_valid


def train_list_classifier(
    train_ds,
    val_ds,
    *,
    init_transformer_fn: Callable = init_simplified_linear_transformer,
    num_epochs: int = 20,
    batch_size: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tmp: float = 1/20,
    layers: int = 1,
    max_new_tokens: int = 10,
    masked_some_attention: bool = True,
    mask_out_last_vertex_as_output: bool = True,
    verbose: bool = True,
    use_softmax_attention: bool = False,
    amount_of_sub_batches: int = 1,
    early_stop_at_100_acc: bool = True,
    **kwargs
):
    """Train a linear transformer on list classification task."""
    tokenizer = train_ds.tokenizer
    model = init_transformer_fn(
        tokenizer.vocab_size, 
        tokenizer, 
        device=device, 
        tmp=tmp, 
        layers=layers, 
        masked_some_attention=masked_some_attention,
        mask_out_last_vertex_as_output=mask_out_last_vertex_as_output,
        use_softmax_attention=use_softmax_attention
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    def collate(batch):
        seqs = [b[0] for b in batch]
        tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas = [b[2] for b in batch]
        padded = collate_fn(seqs, tokenizer.pad_token_id)
        return padded, tgts, metas

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False, num_workers=4)

    train_history = []
    val_history = []

    try:
        for epoch in range(num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=not verbose)
            train_correct = 0.0
            train_total = 0
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in pbar:
                inp, tgt, metas = batch
                inp = inp.to(device)
                tgt = tgt.to(device)

                batch_size_actual = inp.shape[0]
                sub_batch_size = batch_size_actual // amount_of_sub_batches
                
                optim.zero_grad()
                
                total_objective = 0.0
                total_metric_loss = 0.0
                total_num_valid = 0
                
                for sub_batch_idx in range(amount_of_sub_batches):
                    start_idx = sub_batch_idx * sub_batch_size
                    end_idx = batch_size_actual if sub_batch_idx == amount_of_sub_batches - 1 else start_idx + sub_batch_size
                    
                    sub_inp = inp[start_idx:end_idx]
                    sub_tgt = tgt[start_idx:end_idx]
                    sub_metas = metas[start_idx:end_idx]
                    
                    sinks_ids_batch = [m.get('sinks', []) for m in sub_metas]
                    sinks_token_ids_batch = [m.get('sinks_token_ids', []) for m in sub_metas]
                    
                    setattr(sub_inp, 'sinks_ids', sinks_ids_batch)
                    generated, path_logprob_sum = sample_until_max_with_logprob_sums(model, sub_inp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    
                    sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, sub_inp.device)
                    objective = compute_reinforce_loss_batched_with_mask(tokenizer, generated, path_logprob_sum, sub_tgt, sinks_mask)
                    metric_loss, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, generated.detach(), sub_tgt, sinks_mask)
                    
                    scaled_objective = objective / amount_of_sub_batches
                    scaled_objective.backward()
                    
                    total_objective += objective.item()
                    total_metric_loss += metric_loss.item() * num_valid.item()
                    total_num_valid += num_valid.item()
                
                optim.step()

                batch_acc = 1.0 - (total_metric_loss / max(total_num_valid, 1)) if total_num_valid > 0 else 0.0
                train_correct += batch_acc * total_num_valid
                train_total += total_num_valid
                epoch_loss += total_objective / amount_of_sub_batches
                epoch_batches += 1
                
                pbar.set_postfix({"loss": f"{total_metric_loss / max(total_num_valid, 1):.3f}", "acc": f"{batch_acc:.3f}"})
            
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
                    
                    vgen_sampled, vpath_logprob_sum = sample_until_max_with_logprob_sums(model, vinp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    vgen_greedy = generate_until_max_greedy(model, vinp, max_new_tokens=max_new_tokens, tokenizer=tokenizer)
                    sinks_mask = build_sinks_mask_fast(sinks_token_ids_batch, tokenizer.vocab_size, vinp.device)
                    
                    val_objective = compute_reinforce_loss_batched_with_mask(tokenizer, vgen_sampled, vpath_logprob_sum, vtgt, sinks_mask)
                    val_loss_sum += val_objective.item()
                    val_batches += 1
                    
                    batch_metric, num_valid = compute_batch_metric_sink_0_1_with_mask(tokenizer, vgen_greedy, vtgt, sinks_mask)
                    acc = 1.0 - batch_metric.item() if num_valid.item() > 0 else 0.0
                    correct += acc * int(num_valid.item())
                    total += int(num_valid.item())
                    
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
                    print(f"Epoch {epoch+1}/{num_epochs}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%, chain_traverse={chain_traverse_acc:.2f}%")
                
                if early_stop_at_100_acc and train_acc >= 100.0 and val_acc >= 100.0:
                    if verbose:
                        print(f"Early stopping: Both train and val accuracy reached 100% at epoch {epoch+1}")
                    break

    except KeyboardInterrupt:
        if verbose:
            print("\nTraining interrupted. Returning results so far...")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_history": train_history,
        "val_history": val_history,
    }


def plot_training_curves(results, save_path="training_curves.png"):
    """Plot training curves."""
    import matplotlib.pyplot as plt
    
    train_history = results.get("train_history", [])
    val_history = results.get("val_history", [])
    
    if not train_history or not val_history:
        return
    
    epochs = [h['epoch'] for h in train_history]
    train_acc = [h['accuracy'] for h in train_history]
    train_loss = [h['loss'] for h in train_history]
    val_acc = [h['accuracy'] for h in val_history]
    val_loss = [h['loss'] for h in val_history]
    val_chain_traverse = [h.get('chain_traverse', 0) for h in val_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(epochs, train_acc, 'b-o', label='Train', markersize=4)
    axes[0, 0].plot(epochs, val_acc, 'r-s', label='Val', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_loss, 'b-o', label='Train', markersize=4)
    axes[0, 1].plot(epochs, val_loss, 'r-s', label='Val', markersize=4)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, val_chain_traverse, 'g-^', markersize=4, linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Chain Traverse Accuracy (%)')
    axes[1, 0].set_title('Chain Traverse Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    line1 = ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=4)
    line2 = ax2.plot(epochs, val_chain_traverse, 'g-^', label='Chain Traverse', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val Loss', color='r')
    ax2.set_ylabel('Chain Traverse (%)', color='g')
    ax1.legend(line1 + line2, ['Val Loss', 'Chain Traverse'], loc='center right')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig
