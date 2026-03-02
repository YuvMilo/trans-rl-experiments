from typing import List

class DAGTokenizer:
    """Specialized tokenizer for DAG representations."""

    def __init__(self, dag_size: int):
        self.dag_size = dag_size
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'
        self.sep_token = '<SEP>'

        self.vocab: List[str] = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.sep_token]
        self.vocab.extend([f'v{i}' for i in range(dag_size)])
        for i in range(dag_size):
            for j in range(dag_size):
                if i != j:
                    self.vocab.append(f'({i},{j})')
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # Encoding / decoding helpers
    # ------------------------------------------------------------------
    def encode(self, tokens: List[str], *, add_bos=False, add_eos=False) -> List[int]:
        ids = [self.bos_token_id] if add_bos else []
        ids += [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], *, skip_special_tokens=True) -> List[str]:
        toks = []
        for idx in ids:
            tok = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and tok in {self.pad_token, self.bos_token, self.eos_token, self.unk_token}:
                continue
            toks.append(tok)
        return toks

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def is_vertex_token(self, tok: str) -> bool:
        """Return True if *tok* string represents a vertex (e.g. 'v3')."""
        return tok.startswith('v') and tok[1:].isdigit()

    def is_edge_token(self, tok: str) -> bool:
        """Return True if *tok* string represents an edge token like '(2,5)'."""
        return tok.startswith('(') and tok.endswith(')')

    # Deprecated alias for backwards-compatibility
    is_vertex = is_vertex_token

