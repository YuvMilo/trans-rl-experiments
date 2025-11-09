"""
Miscellaneous utility functions.
"""
import torch
from typing import List, Tuple


def collate_fn(batch, pad_token_id):
    """
    Collate function for DataLoader that pads sequences to the same length.
    
    Args:
        batch: List of tensors
        pad_token_id: Token ID to use for padding
    
    Returns:
        Padded tensor batch
    """
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        padding_needed = max_len - len(seq)
        if padding_needed > 0:
            pad = seq.new_full((padding_needed,), pad_token_id)
            padded_seq = torch.cat([seq, pad])
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.stack(padded)


def collate_fn_uniform(batch, pad_token_id):
    """
    Collate function for DataLoader that handles tuples of (sequence, possible_tokens, uniform_targets).
    
    Args:
        batch: List of tuples (tensor, list of possible_tokens, dict of uniform_targets)
        pad_token_id: Token ID to use for padding
    
    Returns:
        Tuple of (padded_tensor_batch, list_of_possible_tokens, batched_uniform_targets)
    """
    sequences = [item[0] for item in batch]
    possible_tokens_lists = [item[1] for item in batch]
    
    # Handle both old format (2-tuple) and new format (3-tuple) for backward compatibility
    if len(batch[0]) == 3:
        uniform_targets_dicts = [item[2] for item in batch]
    else:
        # Create empty dicts for old format
        max_positions = 10
        uniform_targets_dicts = [{
            'positions': torch.full((max_positions,), -1, dtype=torch.long),
            'targets': torch.zeros((max_positions, len(batch[0][0])), dtype=torch.float32),
            'mask': torch.zeros(max_positions, dtype=torch.bool)
        } for _ in batch]
    
    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        padding_needed = max_len - len(seq)
        if padding_needed > 0:
            pad = seq.new_full((padding_needed,), pad_token_id)
            padded_seq = torch.cat([seq, pad])
        else:
            padded_seq = seq
        padded.append(padded_seq)
    
    # Pad possible_tokens lists with None to match sequence length
    padded_possible_tokens = []
    for possible_tokens in possible_tokens_lists:
        # Extend possible_tokens to match padded sequence length
        padded_possible = possible_tokens + [None] * (max_len - len(possible_tokens))
        padded_possible_tokens.append(padded_possible)
    
    # Stack uniform targets into batch tensors
    if uniform_targets_dicts:
        batch_uniform_targets = {
            'positions': torch.stack([d['positions'] for d in uniform_targets_dicts]),  # (batch_size, max_positions)
            'targets': torch.stack([d['targets'] for d in uniform_targets_dicts]),      # (batch_size, max_positions, vocab_size)
            'mask': torch.stack([d['mask'] for d in uniform_targets_dicts])             # (batch_size, max_positions)
        }
    else:
        batch_uniform_targets = None
    
    return torch.stack(padded), padded_possible_tokens, batch_uniform_targets


def get_vocab_labels_and_positions(tokenizer, dag_size: int) -> Tuple[List[str], List[int]]:
    """
    Generate vocabulary labels and positions for visualization from a tokenizer.
    
    Args:
        tokenizer: DAGTokenizer instance
        dag_size: Size of the DAG (number of vertices)
    
    Returns:
        Tuple of (vocab_labels, vocab_positions) lists
    """
    vocab_labels = []
    vocab_positions = []
    
    # Add all special tokens
    for i in range(5):  # PAD, BOS, EOS, UNK, SEP
        if i in tokenizer.id_to_token:
            vocab_labels.append(tokenizer.id_to_token[i])
            vocab_positions.append(i)
    
    # Add all vertex tokens
    for i in range(dag_size):
        vertex_token = f'v{i}'
        if vertex_token in tokenizer.token_to_id:
            token_id = tokenizer.token_to_id[vertex_token]
            vocab_labels.append(vertex_token)
            vocab_positions.append(token_id)
    
    # Add some edge tokens (every 1 to show all)
    edge_count = 0
    for token, token_id in tokenizer.token_to_id.items():
        if token.startswith('(') and token.endswith(')'):
            if edge_count % 1 == 0:  # Show every edge token
                vocab_labels.append(token)
                vocab_positions.append(token_id)
            edge_count += 1
    
    return vocab_labels, vocab_positions 