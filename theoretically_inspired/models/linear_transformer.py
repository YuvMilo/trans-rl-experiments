import torch
from dataclasses import dataclass
from typing import List


@dataclass
class AttentionConfig:
    """Configuration for attention layers"""
    pass


@dataclass
class TransformerConfig:
    """Configuration for the transformer model"""
    vocab_size: int
    max_seq_len: int
    num_layers: int
    tokenizer: object
    tmp: float = 0.5
    attention_config: AttentionConfig = None
    mask_past_verticies: bool = False
    values_for_V: bool = True
    use_cumsum: bool = True
    masked_some_attention: bool = False
    EOS_in_vocab: bool = False
    use_softmax_attention: bool = False
    
    def __post_init__(self):
        if self.attention_config is None:
            self.attention_config = AttentionConfig()


@dataclass
class LinearAttentionActivation:
    """Stores activations from a linear attention layer forward pass"""
    QK_multiplied: torch.Tensor
    output: torch.Tensor


@dataclass
class LinearTransformerActivations:
    """Stores all activations from a forward pass"""
    embeddings: torch.Tensor
    attention_activations: List[LinearAttentionActivation]
    cumsum_output: torch.Tensor
    logits: torch.Tensor
