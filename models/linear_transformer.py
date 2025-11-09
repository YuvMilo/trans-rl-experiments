import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class AttentionConfig:
    """Configuration for attention layers"""
    pass


@dataclass
class TransformerConfig:
    """Configuration for the LinearTransformer model"""
    vocab_size: int
    max_seq_len: int
    num_layers: int
    tokenizer: object  # DAGTokenizer instance
    tmp: float = 0.5
    attention_config: AttentionConfig = None
    mask_past_verticies: bool = False
    values_for_V: bool = True  # If False, zero out V values for vertex tokens
    use_cumsum: bool = True  # If False, skip cumsum operation (use last attention layer output directly)
    masked_some_attention: bool = False  # If True, edges attend to nothing, vertices attend only to edges
    EOS_in_vocab: bool = False  # If True, allow EOS token in output vocabulary
    
    def __post_init__(self):
        if self.attention_config is None:
            self.attention_config = AttentionConfig()


@dataclass
class LinearAttentionActivation:
    """Stores activations from a linear attention layer forward pass"""
    QK_multiplied: torch.Tensor  # Attention weights after applying inputs: xQK^Tx (batch_size, seq_len, seq_len)
    output: torch.Tensor  # Layer output (batch_size, seq_len, hidden_dim)


@dataclass
class LinearTransformerActivations:
    """Stores all activations from a LinearTransformer forward pass"""
    embeddings: torch.Tensor  # Input embeddings (batch_size, seq_len, vocab_size)
    attention_activations: List[LinearAttentionActivation]  # One per attention layer
    cumsum_output: torch.Tensor  # Output after cumsum (batch_size, seq_len, hidden_dim)
    logits: torch.Tensor  # Final logits (batch_size, seq_len, vocab_size)


class LinearAttentionLayerWeights:
    """Stores and provides access to linear attention layer weights"""
    
    def __init__(self, layer: 'LinearAttentionLayer'):
        self.layer = layer
    
    @property
    def Q(self) -> np.ndarray:
        """Get Q matrix as numpy array"""
        return self.layer.W_q.weight.data.cpu().numpy()
    
    @property
    def K(self) -> np.ndarray:
        """Get K matrix as numpy array"""
        return self.layer.W_k.weight.data.cpu().numpy()
    
    @property
    def V(self) -> np.ndarray:
        """Get V matrix as numpy array"""
        return self.layer.W_v.weight.data.cpu().numpy()
    
    def get_QK_non_multiplied(self) -> np.ndarray:
        """Get QK product matrix (Q^T @ K) - the actual weight matrices multiplied together
        
        This returns the matrix multiplication of the Q and K weight matrices,
        NOT the attention scores after applying inputs (which would be x @ Q^T @ K @ x^T).
        """
        Q = self.Q
        K = self.K
        return Q.T @ K


class LinearTransformerWeights:
    """Stores and provides access to all transformer weights"""
    
    def __init__(self, model: 'LinearTransformer'):
        self.model = model
        self.attention_layers_weights = [
            LinearAttentionLayerWeights(layer) 
            for layer in model.attention_layers
        ]
    
    def get_layer_weights(self, layer_idx: int) -> LinearAttentionLayerWeights:
        """Get weights for a specific attention layer"""
        return self.attention_layers_weights[layer_idx]


class LinearAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, config, tokenizer=None, masked_some_attention=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        assert input_dim == hidden_dim, "Input and hidden dimensions must match for diagonal matrix"
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)
        
        self.tokenizer = tokenizer
        self.masked_some_attention = masked_some_attention
        
        # Pre-compute vertex indices if masking is enabled
        self.vertex_indices = None
        if self.masked_some_attention and self.tokenizer:
            self.vertex_indices = [idx for token, idx in self.tokenizer.token_to_id.items() if token.startswith('v')]
        
        self.last_activation: Optional[LinearAttentionActivation] = None

    def forward(self, x, save_activation: bool = False):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # Ensure linear layers are on the same device as input
        if self.W_q.weight.device != device:
            self.W_q = self.W_q.to(device)
            self.W_k = self.W_k.to(device)
            self.W_v = self.W_v.to(device)
        
        # Ensure W_k stays identity and frozen (if it was set that way during init)
        if not self.W_k.weight.requires_grad:
            with torch.no_grad():
                eye = torch.eye(self.W_k.weight.data.shape[0], device=device)
                self.W_k.weight.data.copy_(eye)
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Apply attention masking if enabled
        if self.masked_some_attention and self.vertex_indices:
            # Get token indices for each position (assuming x is one-hot or has clear argmax)
            token_indices = torch.argmax(x, dim=-1)  # (batch_size, seq_len)
            
            # Create vertex mask: True where tokens are vertices
            vertex_mask = torch.zeros_like(token_indices, dtype=torch.bool)
            for v_idx in self.vertex_indices:
                vertex_mask |= (token_indices == v_idx)
            
            # Create edge mask: True where tokens are NOT vertices (i.e., edges)
            edge_mask = ~vertex_mask
            
            # Zero out Q for edge positions (edges attend to nothing)
            edge_positions = edge_mask.unsqueeze(-1).expand_as(Q)  # (batch_size, seq_len, hidden_dim)
            Q = Q * (~edge_positions).float()
            
            # For vertices: we need to mask their attention to other vertices
            # This is done by masking elements in the QK product later, but we can prepare here
            # Store masks for later use in QK computation
            self._vertex_mask = vertex_mask
            self._edge_mask = edge_mask
            
        QK = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        
        # Apply additional attention masking if enabled
        if self.masked_some_attention and hasattr(self, '_vertex_mask'):
            # For each vertex position (query), mask attention to other vertices (keys)
            # vertex_mask shape: (batch_size, seq_len) - True where query is vertex
            # We want to zero out QK[i,j] where i is vertex and j is vertex
            vertex_to_vertex_mask = self._vertex_mask.unsqueeze(2) & self._vertex_mask.unsqueeze(1)  # (batch_size, seq_len, seq_len)
            QK = QK.masked_fill(vertex_to_vertex_mask, 0.0)
        
        # Create and apply causal mask to QK
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        QK = QK.masked_fill(causal_mask, 0.0)
        
        # Compute masked attention with values
        output = torch.bmm(QK, V)
        
        # Save activation if requested
        if save_activation:
            self.last_activation = LinearAttentionActivation(
                QK_multiplied=QK.detach().clone(),
                output=output.detach().clone()
            )
        
        return output


class LinearTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers
        self.attention_config = config.attention_config
        self.input_dim = self.vocab_size 
        self.attention_layers = nn.ModuleList()
        self.tokenizer = config.tokenizer
        self.tmp = config.tmp
        self.mask_past_verticies = config.mask_past_verticies
        self.use_cumsum = config.use_cumsum
        self.masked_some_attention = config.masked_some_attention
        self.EOS_in_vocab = config.EOS_in_vocab

        # Create attention layers based on use_cumsum setting
        if self.use_cumsum:
            # Original behavior: create num_layers-1 attention layers (last layer is cumsum)
            num_attention_layers = self.num_layers - 1
        else:
            # New behavior: create num_layers attention layers (no cumsum)
            num_attention_layers = self.num_layers
            
        for i in range(num_attention_layers):
            self.attention_layers.append(
                LinearAttentionLayer(
                    self.input_dim, 
                    self.input_dim, 
                    config=self.attention_config, 
                    tokenizer=self.tokenizer,
                    masked_some_attention=self.masked_some_attention
                )
            )
        self.final_dim = self.input_dim 
        
        # Check if tokenizer has exit tokens (new tokenizer type)
        has_exit_tokens = hasattr(self.tokenizer, 'is_exit_token')
        
        # Get all non-vertex token indices (exclude EOS if EOS_in_vocab is True, and exit tokens if they exist)
        if has_exit_tokens:
            # New tokenizer with exit tokens: allow both vertex tokens AND exit tokens
            if self.EOS_in_vocab:
                non_vertex_indices = [
                    idx for token, idx in self.tokenizer.token_to_id.items()
                    if not token.startswith('v') and not token.startswith('E') and idx != self.tokenizer.eos_token_id
                ]
            else:
                non_vertex_indices = [
                    idx for token, idx in self.tokenizer.token_to_id.items()
                    if not token.startswith('v') and not token.startswith('E')
                ]
        else:
            # Old tokenizer without exit tokens: only allow vertex tokens
            if self.EOS_in_vocab:
                non_vertex_indices = [
                    idx for token, idx in self.tokenizer.token_to_id.items()
                    if not token.startswith('v') and idx != self.tokenizer.eos_token_id
                ]
            else:
                non_vertex_indices = [
                    idx for token, idx in self.tokenizer.token_to_id.items()
                    if not token.startswith('v')
                ]
        self.non_vertex_indices = non_vertex_indices
        
        # Get all vertex token indices for masking
        vertex_indices = [
            idx for token, idx in self.tokenizer.token_to_id.items()
            if token.startswith('v')
        ]
        self.vertex_indices = vertex_indices
        
        self.last_activation: Optional[LinearTransformerActivations] = None
       
    def _create_one_hot_embeddings(self, input_ids):
        """Create one-hot embeddings for input tokens"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create one-hot token embeddings
        token_onehot = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        token_onehot.scatter_(2, input_ids.unsqueeze(2), 1.0)

        embeddings = token_onehot
            
        return embeddings
    
    def forward(self, input_ids, attention_mask=None, debug=False, save_activation: bool = False):
        # Create one-hot embeddings
        x = self._create_one_hot_embeddings(input_ids)
        
        attention_activations = []

        # Apply attention layers
        for i, attention_layer in enumerate(self.attention_layers):  
            x = attention_layer(x, save_activation=save_activation)
            
            if save_activation and attention_layer.last_activation is not None:
                attention_activations.append(attention_layer.last_activation)

        # Apply cumsum if enabled, otherwise use last attention layer output directly
        if self.use_cumsum:
            cumsum_output = torch.cumsum(x, dim=1)
        else:
            cumsum_output = x

        logits = cumsum_output / self.tmp
        # Mask out non-vertex logits by setting their logits to a large negative value
        logits[..., self.non_vertex_indices] = -1e9

        # Mask past vertices if enabled (vectorized implementation)
        if self.mask_past_verticies:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Create indicator for vertex positions: (batch_size, seq_len, vocab_size)
            vertex_indicators = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.bool, device=device)
            
            # Use scatter to mark vertex positions efficiently
            vertex_indicators.scatter_(2, input_ids.unsqueeze(2), 1)
            
            # Only keep vertex tokens (mask out non-vertex positions)
            vertex_only_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
            vertex_only_mask[self.vertex_indices] = True
            vertex_indicators = vertex_indicators & vertex_only_mask.unsqueeze(0).unsqueeze(0)
            
            # Cumulative sum to get "seen so far including current position" mask
            cumsum_mask = torch.cumsum(vertex_indicators.long(), dim=1)
            
            # Convert to boolean mask (any count > 0 means seen at or before this position)
            seen_before_mask = cumsum_mask > 0
            
            # Apply mask to logits
            logits = logits.masked_fill(seen_before_mask, -1e9)

        # Save activation if requested
        if save_activation:
            self.last_activation = LinearTransformerActivations(
                embeddings=self._create_one_hot_embeddings(input_ids).detach().clone(),
                attention_activations=attention_activations,
                cumsum_output=cumsum_output.detach().clone(),
                logits=logits.detach().clone()
            )

        return type('ModelOutput', (), {'logits': logits})()
    
    def get_weights(self) -> LinearTransformerWeights:
        """Get structured access to all model weights"""
        return LinearTransformerWeights(self)
    
    def get_last_activations(self) -> Optional[LinearTransformerActivations]:
        """Get the last saved activations"""
        return self.last_activation
    
    def generate(self, input_ids, max_new_tokens=50, **kwargs):
        """Simple generation function"""
        self.eval()
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self(generated)
                next_token_logits = outputs.logits[:, -1, :]
                
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == 2:  # EOS token ID is now 2
                    break
                    
                # Stop if we exceed max length
                if generated.size(1) >= self.max_seq_len:
                    break
    
        return generated 