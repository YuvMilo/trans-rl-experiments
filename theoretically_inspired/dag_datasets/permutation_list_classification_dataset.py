import random
from typing import List, Tuple, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PermutationListClassificationDataset(Dataset):
    """
    Dataset that generates permutations of a specific graph structure.
    
    Each sample is constructed by:
    - Take a specific connectivity pattern (list of booleans)
    - Generate a random permutation of vertices
    - Connect vertex i to vertex i+1 if connectivity_pattern[i] is True
    - Choose source vertex based on starting_vertex parameter:
      * If starting_vertex == -1: Choose a random source vertex uniformly (original behavior)
      * If starting_vertex >= 0: Use permutation[starting_vertex] as source (with fallback if invalid)
      * If starting_vertex is a list [a, b, c]: Randomly choose one position from the list
      * If starting_vertex is a tuple (start, end): Randomly choose from permutation[start:end+1] as source
      * If starting_vertex is a dict {idx: prob}: Sample index according to probability distribution
    - Target is the last vertex of the chain that contains the source vertex
    
    The input sequence is a list of edge tokens followed by a single source vertex token.
    """

    def __init__(
        self,
        *,
        dag_tokenizer,
        dag_size: int,
        num_samples: int,
        connectivity_pattern: List[bool],
        starting_vertex: Union[int, Tuple[int, int], List[int], Dict[int, float]] = -1,  # -1 for random, specific index, or dict of {index: probability}
        seed: int = 42,
        add_bos: bool = False,  # Whether to add BOS token at the start of sequences
    ) -> None:
        super().__init__()
        assert len(connectivity_pattern) == dag_size - 1, f"connectivity_pattern must have length {dag_size - 1}, got {len(connectivity_pattern)}"
        
        self.tokenizer = dag_tokenizer
        self.dag_size = dag_size
        self.num_samples = num_samples
        self.connectivity_pattern = connectivity_pattern
        self.starting_vertex = starting_vertex
        self.add_bos = add_bos
        self.rng = random.Random(seed)

        # Validate and prepare dict distribution if provided
        self.starting_vertex_indices = None
        self.starting_vertex_weights = None
        if isinstance(starting_vertex, dict):
            # Validate that all keys are valid indices
            for idx in starting_vertex.keys():
                if not isinstance(idx, int) or idx < 0 or idx >= dag_size:
                    raise ValueError(f"All dict keys must be valid indices in range [0, {dag_size-1}], got {idx}")
            
            # Validate that probabilities sum to 1 (with tolerance for floating point errors)
            prob_sum = sum(starting_vertex.values())
            if abs(prob_sum - 1.0) > 1e-6:
                raise ValueError(f"Probabilities in starting_vertex dict must sum to 1, got {prob_sum}")
            
            # Prepare for efficient sampling using random.choices
            self.starting_vertex_indices = list(starting_vertex.keys())
            self.starting_vertex_weights = list(starting_vertex.values())

        self.samples: List[Tuple[torch.Tensor, int, Dict[str, Any]]] = []
        self._generate()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        seq_tensor, target_id, meta = self.samples[idx]
        return seq_tensor, target_id, meta

    # ------------------------------- helpers -------------------------------
    def _generate(self) -> None:
        for _ in tqdm(range(self.num_samples), desc="Generating permutation dataset samples"):
            # Generate random permutation of vertices
            perm = list(range(self.dag_size))
            self.rng.shuffle(perm)

            # Build chains based on connectivity pattern
            chains: List[List[int]] = []
            current_chain: List[int] = [perm[0]]
            edges: List[Tuple[int, int]] = []

            for i in range(self.dag_size - 1):
                u, v = perm[i], perm[i + 1]
                if self.connectivity_pattern[i]:  # Connect if pattern says True
                    edges.append((u, v))
                    current_chain.append(v)
                else:
                    # Finalize current chain and start new one
                    chains.append(current_chain)
                    current_chain = [v]
            
            # Finalize last chain
            if current_chain:
                chains.append(current_chain)

            # Precompute sinks (last vertex of each chain)
            sinks_vertices = [ch[-1] for ch in chains if len(ch) > 0]

            # Find chains with at least 2 elements (required for non-last selection)
            valid_chains = [ch for ch in chains if len(ch) >= 2]
            if not valid_chains:
                # Fallback: create a chain from first two vertices
                # This should rarely happen with reasonable connectivity patterns
                valid_chains = [[perm[0], perm[1]]]
                chains = valid_chains
                edges = [(perm[0], perm[1])]
                sinks_vertices = [perm[1]]
                
            # Select source vertex based on starting_vertex parameter
            if self.starting_vertex == -1:
                # Random selection (original behavior)
                chain = self.rng.choice(valid_chains)
                source_pos = self.rng.randrange(len(chain) - 1)  # never last element
                source_vertex = chain[source_pos]
                target_vertex = chain[-1]
            else:
                # Handle dict, list of positions, tuple range, or single index
                if isinstance(self.starting_vertex, dict):
                    # Dict distribution: sample according to probabilities
                    # Use random.choices for efficient weighted sampling
                    chosen_idx = self.rng.choices(self.starting_vertex_indices, weights=self.starting_vertex_weights, k=1)[0]
                    desired_source_vertex = perm[chosen_idx]
                elif isinstance(self.starting_vertex, list) and len(self.starting_vertex) > 0 and isinstance(self.starting_vertex[0], int):
                    # List of positions: randomly choose one from the list
                    chosen_idx = self.rng.choice(self.starting_vertex)
                    if chosen_idx < 0 or chosen_idx >= self.dag_size:
                        raise ValueError(f"starting_vertex list element must be in range [0, {self.dag_size-1}], got {chosen_idx}")
                    desired_source_vertex = perm[chosen_idx]
                elif isinstance(self.starting_vertex, tuple):
                    # Range selection: choose random index from the specified range
                    start_idx, end_idx = self.starting_vertex
                    if start_idx < 0 or end_idx >= self.dag_size or start_idx > end_idx:
                        raise ValueError(f"starting_vertex tuple must have valid range [0, {self.dag_size-1}], got {self.starting_vertex}")
                    
                    # Choose random index from the range [start_idx, end_idx] (inclusive)
                    chosen_idx = self.rng.randint(start_idx, end_idx)
                    desired_source_vertex = perm[chosen_idx]
                else:
                    # Use specified starting vertex (single index)
                    if self.starting_vertex < 0 or self.starting_vertex >= self.dag_size:
                        raise ValueError(f"starting_vertex must be -1 (random), list of positions, tuple [start, end], dict {{idx: prob}}, or in range [0, {self.dag_size-1}], got {self.starting_vertex}")
                    
                    # starting_vertex is the index in the permutation
                    # So starting_vertex=2 means use perm[2] as the source vertex
                    desired_source_vertex = perm[self.starting_vertex]
                
                # Try to use the desired vertex as source (common logic for both cases)
                target_chain = None
                for ch in valid_chains:
                    if desired_source_vertex in ch and ch.index(desired_source_vertex) < len(ch) - 1:
                        target_chain = ch
                        break
                
                if target_chain is not None:
                    # Great! We can use the desired vertex as source
                    source_vertex = desired_source_vertex
                    target_vertex = target_chain[-1]
                else:
                    # The desired vertex can't be used as source
                    # This happens when it's the last element of its chain or not in any chain
                    # In this case, let's try to find a valid source from the same chain
                    found_chain = None
                    for ch in valid_chains:
                        if desired_source_vertex in ch:
                            found_chain = ch
                            break
                    
                    if found_chain is not None and len(found_chain) > 1:
                        # Use a different vertex from the same chain as source
                        # Try to pick the vertex just before the desired one if possible
                        desired_pos = found_chain.index(desired_source_vertex)
                        if desired_pos > 0:
                            source_vertex = found_chain[desired_pos - 1]
                        else:
                            # Desired vertex is first in chain, use it anyway if it's not last
                            if desired_pos < len(found_chain) - 1:
                                source_vertex = desired_source_vertex
                            else:
                                # Single element chain - this shouldn't happen with valid_chains
                                source_vertex = found_chain[0]
                        target_vertex = found_chain[-1]
                    else:
                        # Desired vertex not found in any valid chain - complete fallback
                        chain = self.rng.choice(valid_chains)
                        source_pos = self.rng.randrange(len(chain) - 1)
                        source_vertex = chain[source_pos]
                        target_vertex = chain[-1]

            # Build token strings: edges first, then source vertex
            token_strs: List[str] = []
            for (i, j) in edges:
                token_strs.append(f'({i},{j})')
            token_strs.append(f'v{source_vertex}')
            # no EOS in the input; generation will produce EOS

            # Encode
            input_ids = self.tokenizer.encode(token_strs, add_bos=self.add_bos, add_eos=False)
            seq_tensor = torch.tensor(input_ids, dtype=torch.long)

            # Target id is the vertex token id of the sink of the chain
            target_id = self.tokenizer.token_to_id[f'v{target_vertex}']

            # Pre-compute sinks_token_ids for performance optimization
            sinks_token_ids = [self.tokenizer.token_to_id.get(f'v{v}', self.tokenizer.unk_token_id) for v in sinks_vertices]

            meta = {
                'edges': edges,
                'chains': chains,
                'source_vertex': source_vertex,
                'target_vertex': target_vertex,
                'sinks': sinks_vertices,
                'sinks_token_ids': sinks_token_ids,
                'decoded_tokens': token_strs,
                'permutation': perm,
                'connectivity_pattern': self.connectivity_pattern,
                'starting_vertex_config': self.starting_vertex,
            }
            self.samples.append((seq_tensor, target_id, meta))


