"""
Miscellaneous utility functions.
"""
import torch


def collate_fn(batch, pad_token_id):
    """Collate function for DataLoader that pads sequences to the same length."""
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
