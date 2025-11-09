import torch
import torch.nn as nn
from typing import List
import numpy as np

from .linear_transformer import (
    TransformerConfig,
    LinearAttentionActivation,
    LinearTransformerActivations,
)


class _SimplifiedLayerWeights:
    """Shim to provide Q, K, V for visualization compatibility."""
    def __init__(self, A_param: torch.Tensor, V_param: torch.Tensor, vertex_token_ids: List[int], context_token_ids: List[int], vocab_size: int, device: torch.device):
        self._A = A_param  # (num_vertices, vocab_size)
        self._V = V_param  # (vocab_size, num_vertices)
        self._vertex_token_ids = vertex_token_ids
        self._context_token_ids = context_token_ids
        self._vocab_size = vocab_size
        self._device = device

    @property
    def Q(self) -> np.ndarray:
        """Return a (vocab_size, vocab_size) matrix with A rows placed at vertex token rows."""
        with torch.no_grad():
            q_full = torch.zeros(self._vocab_size, self._vocab_size, device=self._device)
            for row_idx, tok_id in enumerate(self._vertex_token_ids):
                # Place context columns only
                q_full[tok_id, self._context_token_ids] = self._A[row_idx]
            return q_full.cpu().numpy()

    @property
    def K(self) -> np.ndarray:
        """Identity in simplified model."""
        with torch.no_grad():
            eye = torch.eye(self._vocab_size, device=self._device)
            return eye.cpu().numpy()

    @property
    def V(self) -> np.ndarray:
        """Return V as numpy (vocab_size, num_vertices)."""
        with torch.no_grad():
            return self._V.cpu().numpy()
    
    @property
    def A_raw(self) -> np.ndarray:
        """Return the actual learned A matrix (num_vertices, num_context)."""
        with torch.no_grad():
            return self._A.cpu().numpy()

    def get_QK_non_multiplied(self) -> np.ndarray:
        """Return Q^T @ K; with K=I and Q as above, equals Q^T."""
        Q = self.Q
        return Q.T


class _SimplifiedWeights:
    def __init__(self, A_param: torch.Tensor, V_param: torch.Tensor, vertex_token_ids: List[int], context_token_ids: List[int], vocab_size: int, device: torch.device):
        self.attention_layers_weights = [
            _SimplifiedLayerWeights(A_param, V_param, vertex_token_ids, context_token_ids, vocab_size, device)
        ]


class SimplifiedLinearTransformer(nn.Module):
    """
    Simplified linear attention:
      - Outputs only vertex tokens
      - K assumed identity; no K params or multiplies
      - Query (A) is defined per-vertex and attends over all context tokens
      - Values project context tokens to vertex logits
      - Internally computes reduced-dimension logits, then scatters to full vocab space
      - Provides activations compatible with visualization utilities
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.tokenizer = config.tokenizer
        self.tmp = config.tmp
        self.mask_past_verticies = config.mask_past_verticies

        # Build vertex and context (vertex+edge only, no specials) token lists and mappings
        self.vertex_token_ids: List[int] = [idx for tok, idx in self.tokenizer.token_to_id.items() if tok.startswith('v')]
        self.num_vertices = len(self.vertex_token_ids)
        # Context = vertices + edges (exclude BOS, EOS, PAD, UNK)
        self.context_token_ids: List[int] = [
            idx for tok, idx in self.tokenizer.token_to_id.items() 
            if tok.startswith('v') or tok.startswith('(')
        ]
        self.num_context = len(self.context_token_ids)

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

        # One-hot tokens and prefix counts (full vocab)
        x_onehot_full = self._one_hot(input_ids)  # (B, T, V)
        counts_full = torch.cumsum(x_onehot_full, dim=1)  # (B, T, V)
        # Slice to context features only
        counts_ctx = counts_full[:, :, self.context_token_ids]  # (B, T, C)

        # Determine current token vertex indices per position
        vertex_idx = self.full_to_vertex_idx[input_ids]  # (B, T)
        is_vertex = vertex_idx.ge(0)  # (B, T)
        safe_vertex_idx = vertex_idx.clamp(min=0)  # (B, T)

        # Gather per-position query vectors A[vertex_idx] over context columns
        A_gather = self.A.index_select(0, safe_vertex_idx.view(-1)).view(bsz, seq_len, self.num_context)  # (B, T, C)
        A_gather = A_gather * is_vertex.unsqueeze(-1).float()  # zero where not vertex position

        # Attention weights over context tokens (elementwise with counts)
        attn_ctx = counts_ctx * A_gather  # (B, T, C)

        # Project context to vertex logits
        logits_vertex = torch.matmul(attn_ctx, self.V_param)  # (B, T, num_vertices)
        logits_vertex = logits_vertex / self.tmp

        # Scatter to full-vocab logits (non-vertex tokens set to large negative)
        logits_full = torch.full((bsz, seq_len, self.vocab_size), -1e9, device=device)
        logits_full[:, :, self.vertex_token_ids] = logits_vertex

        # Optionally mask past vertices
        if self.mask_past_verticies:
            vertex_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            vertex_mask[self.vertex_token_ids] = True
            vertex_indicators = x_onehot_full.bool() & vertex_mask.unsqueeze(0).unsqueeze(0)  # (B, T, V)
            seen_before = torch.cumsum(vertex_indicators.long(), dim=1) > 0
            logits_full = logits_full.masked_fill(seen_before, -1e9)

        if save_activation:
            # Provide activations compatible with existing visualization flow
            # Embed context-only attention into full-vocab last-dim for visualization
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

    def get_weights(self):
        return _SimplifiedWeights(self.A, self.V_param, self.vertex_token_ids, self.context_token_ids, self.vocab_size, next(self.parameters()).device)


