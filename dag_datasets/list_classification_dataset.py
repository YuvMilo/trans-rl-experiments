import random
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ListClassificationDataset(Dataset):
    """
    Dataset of disjoint directed lists (chains) over vertices 0..dag_size-1.

    Each sample is constructed by:
    - Draw a random permutation of vertices
    - Iterate consecutive pairs; with probability p_link, connect them with a directed edge
      (forming chains); otherwise start a new chain
    - Choose a random source vertex uniformly
    - Target is the last vertex of the chain that contains the source vertex

    The input sequence is a list of edge tokens followed by a single source vertex token.
    Tokens are expected to be encoded by the provided tokenizer.
    """

    def __init__(
        self,
        *,
        dag_tokenizer,
        dag_size: int,
        num_samples: int,
        p_link: float = 0.7,
        seed: int = 42,
        add_bos: bool = False,  # Whether to add BOS token at the start of sequences
    ) -> None:
        super().__init__()
        assert 0 <= p_link <= 1, "p_link must be in [0, 1]"
        self.tokenizer = dag_tokenizer
        self.dag_size = dag_size
        self.num_samples = num_samples
        self.p_link = p_link
        self.add_bos = add_bos
        self.rng = random.Random(seed)

        self.samples: List[Tuple[torch.Tensor, int, Dict[str, Any]]] = []
        self._generate()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        seq_tensor, target_id, meta = self.samples[idx]
        return seq_tensor, target_id, meta

    # ------------------------------- helpers -------------------------------
    def _generate(self) -> None:
        for _ in tqdm(range(self.num_samples), desc="Generating dataset samples"):
            # Build disjoint chains
            perm = list(range(self.dag_size))
            self.rng.shuffle(perm)

            chains: List[List[int]] = []
            current_chain: List[int] = [perm[0]]
            edges: List[Tuple[int, int]] = []

            for i in range(self.dag_size - 1):
                u, v = perm[i], perm[i + 1]
                if self.rng.random() < self.p_link:
                    # connect and continue same chain
                    edges.append((u, v))
                    current_chain.append(v)
                else:
                    # finalize chain and start a new one
                    chains.append(current_chain)
                    current_chain = [v]
            # finalize last chain
            if current_chain:
                chains.append(current_chain)

            # Precompute sinks (last vertex of each chain)
            sinks_vertices = [ch[-1] for ch in chains if len(ch) > 0]

            # Find chains with at least 2 elements (required for non-last selection)
            valid_chains = [ch for ch in chains if len(ch) >= 2]
            if not valid_chains:
                # Fallback: force at least one chain of length 2 by connecting first two vertices
                valid_chains = [[0, 1]]
                chains = valid_chains  # replace with fallback
                
            # Pick a random valid chain, then pick a random non-last element as source
            chain = self.rng.choice(valid_chains)
            source_pos = self.rng.randrange(len(chain) - 1)  # never last element
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
            }
            self.samples.append((seq_tensor, target_id, meta))


