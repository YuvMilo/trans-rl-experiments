"""
Training utility functions and classes for DAG transformer training.
"""
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from utils.graph_utils import is_valid_topo
from models import LinearTransformer, AttentionConfig, TransformerConfig


@dataclass
class SavedWeights:
    """Container for saved model weights during training"""
    step: int
    V_matrix: np.ndarray
    QK_matrix: np.ndarray
    Q_matrix: Optional[np.ndarray] = None
    K_matrix: Optional[np.ndarray] = None
    vertex_outputs: Optional[np.ndarray] = None
    token_labels: Optional[List[str]] = None
    cumsum_output: Optional[np.ndarray] = None
    softmax_probs: Optional[np.ndarray] = None


@dataclass 
class ValidationResult:
    """Result of validating a single example"""
    hit_with_repeat: bool
    hit_no_repeat: bool
    wanted_output: str
    got_output_with_repeat: str
    got_output_no_repeat: str
    decoded_prefix: str


@dataclass
class ValidationResultUniform:
    """Result of validating a single example with possible next tokens"""
    hit_with_repeat: bool
    hit_no_repeat: bool
    wanted_output: str
    got_output_with_repeat: str
    got_output_no_repeat: str
    decoded_prefix: str
    possible_next_tokens: List[str]  # String representation of possible tokens at each position


def init_transformer(vocab_size: int, tokenizer, device: str, **kwargs) -> LinearTransformer:
    """
    Initialize a LinearTransformer model with specific configuration.
    
    Args:
        vocab_size: Size of the vocabulary
        tokenizer: DAG tokenizer instance
        device: Device to place the model on
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized LinearTransformer model
    """
    model = LinearTransformer(
        config=TransformerConfig(
            vocab_size=vocab_size, 
            max_seq_len=vocab_size*2+4,
            num_layers=2,
            tokenizer=tokenizer,
            tmp=1/20,
            attention_config=AttentionConfig()
        )
    ).to(device)
    
    # Make all params of layer start near zero
    sig_gen_layer_0 = 0.01
    sig_v_layer_0 = 0.001
    
    for param in model.attention_layers[0].parameters():
        param.data.normal_(mean=0.0, std=sig_gen_layer_0)
        param.data = param.data.abs()

    model.attention_layers[0].W_v.weight.data.normal_(mean=0.0, std=sig_v_layer_0)
    model.attention_layers[0].W_v.weight.data = model.attention_layers[0].W_v.weight.data.abs()

    # Force K of fist layer to be identity and not trainable
    model.attention_layers[0].W_k.weight.data = torch.eye(
        model.attention_layers[0].W_k.weight.data.shape[0], 
        device=device
    )
    model.attention_layers[0].W_k.weight.requires_grad = False

    # Ensure all parameters are on the correct device after manual initialization
    model = model.to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model


def init_transformer_non_repeating(vocab_size: int, tokenizer, device: str, **kwargs) -> LinearTransformer:
    """
    Initialize a LinearTransformer model with non-repeating vertex configuration.
    
    This is identical to init_transformer but enables mask_past_verticies=True
    to prevent the model from outputting vertex tokens that have already appeared.
    
    Args:
        vocab_size: Size of the vocabulary
        tokenizer: DAG tokenizer instance
        device: Device to place the model on
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized LinearTransformer model with non-repeating vertices enabled
    """
    print("tmp", kwargs.get('tmp', 1/20))
    model = LinearTransformer(
        config=TransformerConfig(
            vocab_size=vocab_size, 
            max_seq_len=vocab_size*2+4,
            num_layers=2,
            tokenizer=tokenizer,
            tmp=kwargs.get('tmp', 1/20),
            attention_config=AttentionConfig(),
            mask_past_verticies=True  # Enable non-repeating vertices
        )
    ).to(device)
    
    # Make all params of layer start near zero
    sig_gen_layer_0 = 0.01
    sig_v_layer_0 = 0.001
    
    for param in model.attention_layers[0].parameters():
        param.data.normal_(mean=0.0, std=sig_gen_layer_0)
        param.data = param.data.abs()

    model.attention_layers[0].W_v.weight.data.normal_(mean=0.0, std=sig_v_layer_0)
    model.attention_layers[0].W_v.weight.data = model.attention_layers[0].W_v.weight.data.abs()

    # Force K of fist layer to be identity and not trainable
    model.attention_layers[0].W_k.weight.data = torch.eye(
        model.attention_layers[0].W_k.weight.data.shape[0], 
        device=device
    )
    model.attention_layers[0].W_k.weight.requires_grad = False

    # Ensure all parameters are on the correct device after manual initialization
    model = model.to(device)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model


def generate_topological_sort_with_sampling(
    model, 
    prefix_ids: torch.Tensor, 
    tokenizer, 
    dag_size: int, 
    device: str,
    no_repeat: bool = False,
    temperature: float = 1.0
) -> List[int]:
    """
    Generate vertex sequence using sampling from the distribution.
    
    Args:
        model: The transformer model to use for generation
        prefix_ids: Input prefix tensor (1, L)
        tokenizer: DAG tokenizer
        dag_size: Number of vertices in the DAG
        device: Device to run computation on
        no_repeat: If True, prevent repeating vertex tokens
        temperature: Temperature for sampling (1.0 = no scaling)
        
    Returns:
        List of generated token IDs after the prefix
    """
    ids = prefix_ids.clone()
    used = set() if no_repeat else None
    
    # Find already used vertices in prefix if no_repeat is True
    if no_repeat:
        prefix_tokens = ids[0].tolist()
        for t in prefix_tokens:
            tok = tokenizer.id_to_token[t]
            if tok.startswith('v'):
                used.add(tok)
    
    for _ in range(dag_size):
        logits = model(ids).logits[:, -1]  # (1, V)
        
        if no_repeat:
            logits = logits.clone()
            # Mask out already used vertices
            for v in used:
                v_id = tokenizer.token_to_id[v]
                logits[0, v_id] = float('-inf')
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        next_token = tokenizer.id_to_token[next_id.item()]
        
        if next_id.item() in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
            
        if no_repeat and next_token.startswith('v'):
            used.add(next_token)
            
        ids = torch.cat([ids, next_id], dim=1)
    
    # Return list of token IDs after the prefix
    return ids[0, prefix_ids.size(1):].tolist()


def get_current_training_weights(
    model, 
    tokenizer, 
    step_counter: int, 
    fixed_sample_input: Optional[torch.Tensor]
) -> SavedWeights:
    """
    Extract current model weights and activations for saving.
    
    Args:
        model: The transformer model
        tokenizer: DAG tokenizer  
        step_counter: Current training step
        fixed_sample_input: Fixed sample for consistent activation computation
        
    Returns:
        SavedWeights object containing current model state
    """
    with torch.no_grad():
        weights = model.get_weights()
        layer_0_weights = weights.attention_layers_weights[0]
        V = layer_0_weights.V
        QK = layer_0_weights.get_QK_non_multiplied()
        Q = layer_0_weights.Q
        K = layer_0_weights.K
        
        # Compute first layer output for vertices using the fixed sample input
        if fixed_sample_input is not None:
            # Use the new activation saving functionality
            model(fixed_sample_input, save_activation=True)
            activations = model.get_last_activations()
            
            # Extract only vertex token outputs from first attention layer
            vertex_indices = [tid for token, tid in tokenizer.token_to_id.items() if token.startswith('v')]
            first_attention_output = activations.attention_activations[0].output[0]  # Shape: (seq_len, vocab_size)
            vertex_outputs = first_attention_output[:, vertex_indices].cpu().numpy()  # Shape: (seq_len, num_vertices)
            
            # Extract outputs compatible across models
            # Some models (linear) expose cumsum_output; softmax transformer may not
            if hasattr(activations, 'cumsum_output') and activations.cumsum_output is not None:
                cumsum_output = activations.cumsum_output[0].cpu().numpy()  # (seq_len, vocab_size)
            else:
                cumsum_output = None
            logits = activations.logits[0].cpu().numpy()  # (seq_len, vocab_size)
            
            # Compute softmax probabilities from logits
            # Apply softmax along the vocabulary dimension
            import torch.nn.functional as F
            softmax_probs = F.softmax(torch.tensor(logits), dim=-1).numpy()  # Shape: (seq_len, vocab_size)
            
            # Get token labels for the fixed sample
            token_ids = fixed_sample_input[0].cpu().tolist()
            token_labels = [tokenizer.id_to_token[tid] for tid in token_ids]
            
            return SavedWeights(
                step=step_counter,
                V_matrix=V.copy(),
                QK_matrix=QK.copy(),
                Q_matrix=Q.copy(),
                K_matrix=K.copy(),
                vertex_outputs=vertex_outputs.copy(),
                token_labels=token_labels.copy(),
                cumsum_output=cumsum_output.copy() if cumsum_output is not None else None,
                softmax_probs=softmax_probs.copy()
            )
        else:
            # Fallback if no fixed sample is available yet
            return SavedWeights(
                step=step_counter,
                V_matrix=V.copy(),
                QK_matrix=QK.copy(),
                Q_matrix=Q.copy(),
                K_matrix=K.copy()
            )


def validate_example(
    seq: List[int],
    model,
    tokenizer,
    dag_size: int,
    device: str,
    vertex_token_ids: List[int],
    pad_id: int
) -> Optional[ValidationResult]:
    """
    Validate a single example by generating topological sort and checking validity.
    
    Args:
        seq: Token sequence to validate
        model: The transformer model
        tokenizer: DAG tokenizer
        dag_size: Number of vertices in the DAG
        device: Device to run computation on
        vertex_token_ids: List of vertex token IDs
        pad_id: Padding token ID
        
    Returns:
        ValidationResult if successful, None if malformed example
    """
    # Strip padding / EOS on the right
    seq = seq.copy()
    while seq and seq[-1] in (pad_id, tokenizer.eos_token_id):
        seq.pop()

    # Find first vertex token to determine where edges end
    train_start_pos = -1  # Renamed from sep_idx
    for idx, token_id in enumerate(seq):
        if token_id in vertex_token_ids:
            train_start_pos = idx - 1  # Position before first vertex
            break
    
    if train_start_pos < 0:
        return None  # malformed (skip)

    prefix_ids = torch.tensor([seq[:train_start_pos+1]], device=device)
    edge_tokens = seq[1:train_start_pos+1]  # skip BOS, include up to train_start_pos
    edges = []
    for t in edge_tokens:
        tok = tokenizer.id_to_token[t]
        if tok.startswith('('):
            i_, j_ = map(int, tok[1:-1].split(','))
            edges.append((i_, j_))

    # Generate vertex sequences using sampling
    gen_ids = generate_topological_sort_with_sampling(
        model, prefix_ids, tokenizer, dag_size, device, no_repeat=False, temperature=1.0
    )
    gen_tokens = [tokenizer.id_to_token[t] for t in gen_ids
                  if t not in (pad_id, tokenizer.eos_token_id)]
    gen_vertices = [int(tok[1:]) for tok in gen_tokens if tok.startswith('v')]

    gen_ids_no_repeat = generate_topological_sort_with_sampling(
        model, prefix_ids, tokenizer, dag_size, device, no_repeat=True, temperature=1.0
    )
    gen_tokens_no_repeat = [tokenizer.id_to_token[t] for t in gen_ids_no_repeat
                           if t not in (pad_id, tokenizer.eos_token_id)]
    gen_vertices_no_repeat = [int(tok[1:]) for tok in gen_tokens_no_repeat if tok.startswith('v')]

    # Check if the generated vertices form a valid topological sort
    hit_with_repeat = is_valid_topo(gen_vertices, edges, dag_size)
    hit_no_repeat = is_valid_topo(gen_vertices_no_repeat, edges, dag_size)

    # Calculate the gold permutation as the list of vertex indices after train_start_pos
    good_res = []
    for t in seq[train_start_pos+1:]:
        tok = tokenizer.id_to_token[t]
        if tok.startswith('v'):
            good_res.append(int(tok[1:]))

    decoded_prefix = tokenizer.decode(seq[:train_start_pos+1], skip_special_tokens=False)
    
    return ValidationResult(
        hit_with_repeat=hit_with_repeat,
        hit_no_repeat=hit_no_repeat,
        wanted_output=str(good_res),
        got_output_with_repeat=str(gen_vertices),
        got_output_no_repeat=str(gen_vertices_no_repeat),
        decoded_prefix=decoded_prefix
    ) 


def validate_example_uniform(
    seq: List[int],
    possible_tokens: List,
    model,
    tokenizer,
    dag_size: int,
    device: str,
    vertex_token_ids: List[int],
    pad_id: int
) -> Optional[ValidationResultUniform]:
    """
    Validate a single example with possible next tokens information.
    
    Args:
        seq: Token sequence to validate
        possible_tokens: List of possible next tokens for each position
        model: The transformer model
        tokenizer: DAG tokenizer
        dag_size: Number of vertices in the DAG
        device: Device to run computation on
        vertex_token_ids: List of vertex token IDs
        pad_id: Padding token ID
        
    Returns:
        ValidationResultUniform if successful, None if malformed example
    """
    # First get standard validation result
    standard_result = validate_example(seq, model, tokenizer, dag_size, device, vertex_token_ids, pad_id)
    if standard_result is None:
        return None
    
    # Convert possible tokens to string representation
    possible_next_str = []
    for i, pos_tokens in enumerate(possible_tokens):
        if pos_tokens is None:
            possible_next_str.append("N/A")
        else:
            token_strs = [tokenizer.id_to_token[tid] for tid in pos_tokens]
            possible_next_str.append(f"{token_strs}")
    
    return ValidationResultUniform(
        hit_with_repeat=standard_result.hit_with_repeat,
        hit_no_repeat=standard_result.hit_no_repeat,
        wanted_output=standard_result.wanted_output,
        got_output_with_repeat=standard_result.got_output_with_repeat,
        got_output_no_repeat=standard_result.got_output_no_repeat,
        decoded_prefix=standard_result.decoded_prefix,
        possible_next_tokens=possible_next_str
    ) 