import torch
import torch.nn as nn
from typing import List

from .linear_transformer import (
    TransformerConfig,
    LinearAttentionActivation,
    LinearTransformerActivations,
)


class SimplifiedLinearTransformer(nn.Module):
    """
    Simplified linear/softmax attention transformer:
      - Outputs only vertex tokens
      - K assumed identity (no K params)
      - Query (A) defined per-vertex, attends over context tokens
      - Values project context tokens to vertex logits
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.tokenizer = config.tokenizer
        self.tmp = config.tmp
        self.mask_past_verticies = config.mask_past_verticies
        self.use_softmax_attention = config.use_softmax_attention

        # Build vertex and context token lists
        self.vertex_token_ids: List[int] = [idx for tok, idx in self.tokenizer.token_to_id.items() if tok.startswith('v')]
        self.num_vertices = len(self.vertex_token_ids)
        
        # Context = vertices + edges (exclude specials)
        self.context_token_ids: List[int] = [
            idx for tok, idx in self.tokenizer.token_to_id.items() 
            if tok.startswith('v') or tok.startswith('(')
        ]
        self.num_context = len(self.context_token_ids)

        # Build lookup buffers
        full_to_vertex = torch.full((self.vocab_size,), -1, dtype=torch.long)
        for i, tid in enumerate(self.vertex_token_ids):
            full_to_vertex[tid] = i
        self.register_buffer("full_to_vertex_idx", full_to_vertex, persistent=False)
        
        full_to_context = torch.full((self.vocab_size,), -1, dtype=torch.long)
        for j, tid in enumerate(self.context_token_ids):
            full_to_context[tid] = j
        self.register_buffer("full_to_context_idx", full_to_context, persistent=False)

        # Parameters: A (num_vertices x num_context), V (num_context x num_vertices)
        self.A = nn.Parameter(torch.empty(self.num_vertices, self.num_context))
        self.V_param = nn.Parameter(torch.empty(self.num_context, self.num_vertices))
        self.reset_parameters()

        self._last_activations: LinearTransformerActivations = None

    def reset_parameters(self):
        small_std = 1e-8
        nn.init.normal_(self.A, mean=0.0, std=small_std)
        nn.init.normal_(self.V_param, mean=0.0, std=small_std)

    def _one_hot(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        x = torch.zeros(bsz, seq_len, self.vocab_size, device=device)
        x.scatter_(2, input_ids.unsqueeze(2), 1.0)
        return x

    def forward(self, input_ids: torch.Tensor, save_activation: bool = False):
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        x_onehot_full = self._one_hot(input_ids)
        counts_full = torch.cumsum(x_onehot_full, dim=1)
        counts_ctx = counts_full[:, :, self.context_token_ids]

        vertex_idx = self.full_to_vertex_idx[input_ids]
        is_vertex = vertex_idx.ge(0)
        safe_vertex_idx = vertex_idx.clamp(min=0)

        A_gather = self.A.index_select(0, safe_vertex_idx.view(-1)).view(bsz, seq_len, self.num_context)
        A_gather = A_gather * is_vertex.unsqueeze(-1).float()

        if self.use_softmax_attention:
            context_idx = self.full_to_context_idx[input_ids]
            is_context = context_idx.ge(0)
            
            x_ctx = torch.zeros(bsz, seq_len, self.num_context, device=device)
            batch_indices = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seq_len)
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
            valid_mask = is_context
            safe_context_idx = context_idx.clamp(min=0)
            x_ctx[batch_indices[valid_mask], seq_indices[valid_mask], safe_context_idx[valid_mask]] = 1.0
            
            scores = torch.matmul(A_gather, x_ctx.transpose(1, 2))
            
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0), -1e9)
            
            non_context_mask = ~is_context.unsqueeze(1).expand(bsz, seq_len, seq_len)
            scores = scores.masked_fill(non_context_mask, -1e9)
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_ctx = torch.matmul(attn_weights, x_ctx)
        else:
            attn_ctx = counts_ctx * A_gather

        logits_vertex = torch.matmul(attn_ctx, self.V_param) / self.tmp

        logits_full = torch.full((bsz, seq_len, self.vocab_size), -1e9, device=device)
        logits_full[:, :, self.vertex_token_ids] = logits_vertex

        if self.mask_past_verticies:
            for i in range(1, seq_len):
                prev_vertices = is_vertex[:, :i]
                for b in range(bsz):
                    vertex_positions = torch.where(prev_vertices[b])[0]
                    if len(vertex_positions) > 0:
                        last_vertex_pos = vertex_positions[-1].item()
                        last_vertex_token_id = input_ids[b, last_vertex_pos].item()
                        logits_full[b, i, last_vertex_token_id] = -1e9

        if save_activation:
            attn_full = torch.zeros(bsz, seq_len, self.vocab_size, device=device)
            attn_full[:, :, self.context_token_ids] = attn_ctx
            attention_activations = [
                LinearAttentionActivation(
                    QK_multiplied=attn_full.detach().clone(),
                    output=attn_full.detach().clone()
                )
            ]
            self._last_activations = LinearTransformerActivations(
                embeddings=x_onehot_full.detach().clone(),
                attention_activations=attention_activations,
                cumsum_output=None,
                logits=logits_full.detach().clone()
            )

        return type('ModelOutput', (), {'logits': logits_full})()

    def get_last_activations(self):
        return self._last_activations
