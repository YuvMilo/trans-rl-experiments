import random
import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np


# ──────────────────────────────────────────────────────────────
# Polynomial equation data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class Monomial:
    """A monomial term: coefficient * prod(x_vi ^ ei)."""
    coefficient: int
    variables: Dict[int, int]  # variable_id -> exponent

    def evaluate(self, values: Dict[int, int]) -> int:
        result = self.coefficient
        for var, exp in self.variables.items():
            result *= values[var] ** exp
        return result

    def to_string(self) -> str:
        """Render the absolute-value body (sign handled by PolynomialEquation)."""
        if not self.variables:
            return str(abs(self.coefficient))
        parts: List[str] = []
        abs_c = abs(self.coefficient)
        if abs_c != 1:
            parts.append(str(abs_c))
        for var in sorted(self.variables):
            exp = self.variables[var]
            parts.append(f"x_{var}" if exp == 1 else f"x_{var}^{exp}")
        return " * ".join(parts)


@dataclass
class PolynomialEquation:
    """Equation: x_target = sum(monomials) + constant."""
    target_var: int
    monomials: List[Monomial]
    constant: int = 0

    def evaluate(self, values: Dict[int, int]) -> int:
        result = self.constant
        for m in self.monomials:
            result += m.evaluate(values)
        return result

    def get_parent_variables(self) -> Set[int]:
        parents: Set[int] = set()
        for m in self.monomials:
            parents.update(m.variables.keys())
        return parents

    def to_string(self, modulus: Optional[int] = None) -> str:
        pieces: List[Tuple[bool, str]] = []
        for m in self.monomials:
            pieces.append((m.coefficient > 0, m.to_string()))
        if self.constant != 0:
            pieces.append((self.constant > 0, str(abs(self.constant))))
        if not pieces:
            rhs = "0"
        else:
            parts: List[str] = []
            for i, (positive, text) in enumerate(pieces):
                if i == 0:
                    parts.append(text if positive else f"-{text}")
                else:
                    parts.append(f"+ {text}" if positive else f"- {text}")
            rhs = " ".join(parts)
        if modulus is not None:
            return f"x_{self.target_var} = ({rhs}) mod {modulus}"
        return f"x_{self.target_var} = {rhs}"


# ──────────────────────────────────────────────────────────────
# Graph generation
# ──────────────────────────────────────────────────────────────

def generate_chain_graph(
    n_nodes: int, min_chain_length: int = 2, max_chain_length: int = 6
) -> nx.DiGraph:
    """Independent chains (original behaviour)."""
    g = nx.DiGraph()
    cur = 0
    while cur < n_nodes:
        remaining = n_nodes - cur
        length = min(random.randint(min_chain_length, max_chain_length), remaining)
        nodes = list(range(cur, cur + length))
        g.add_nodes_from(nodes)
        for i in range(length - 1):
            g.add_edge(nodes[i], nodes[i + 1])
        cur += length
    return g


def generate_tree_graph(n_nodes: int, max_out_degree: int = 3) -> nx.DiGraph:
    """Single-rooted tree; edges go root -> leaves."""
    g = nx.DiGraph()
    if n_nodes == 0:
        return g
    g.add_node(0)
    available = [0]
    for i in range(1, n_nodes):
        if not available:
            break
        parent = random.choice(available)
        g.add_node(i)
        g.add_edge(parent, i)
        available.append(i)
        if g.out_degree(parent) >= max_out_degree:
            available.remove(parent)
    return g


def generate_reverse_tree_graph(n_nodes: int, max_in_degree: int = 3) -> nx.DiGraph:
    """Reverse tree: multiple roots converge to a single sink."""
    tree = generate_tree_graph(n_nodes, max_out_degree=max_in_degree)
    return tree.reverse()


def generate_dag_graph(
    n_nodes: int,
    max_in_degree: int = 3,
    edge_probability: float = 0.3,
    num_roots: int = None,
) -> nx.DiGraph:
    """General DAG. Nodes 0..num_roots-1 are roots; others have >=1 parent."""
    g = nx.DiGraph()
    if n_nodes == 0:
        return g
    if num_roots is None:
        num_roots = max(1, n_nodes // 5)
    num_roots = max(1, min(num_roots, n_nodes))
    for i in range(n_nodes):
        g.add_node(i)
    for j in range(num_roots, n_nodes):
        possible = list(range(j))
        n_parents = random.randint(1, min(max_in_degree, len(possible)))
        parents = set(random.sample(possible, n_parents))
        for p in possible:
            if p not in parents and random.random() < edge_probability and len(parents) < max_in_degree:
                parents.add(p)
        for p in parents:
            g.add_edge(p, j)
    return g


def generate_graph(n_nodes: int, graph_type: str = "dag", **kwargs) -> nx.DiGraph:
    if graph_type == "chain":
        return generate_chain_graph(
            n_nodes,
            kwargs.get("min_chain_length", 2),
            kwargs.get("max_chain_length", 6),
        )
    if graph_type == "tree":
        return generate_tree_graph(n_nodes, kwargs.get("max_out_degree", 3))
    if graph_type == "reverse_tree":
        return generate_reverse_tree_graph(n_nodes, kwargs.get("max_in_degree", 3))
    if graph_type == "dag":
        return generate_dag_graph(
            n_nodes,
            kwargs.get("max_in_degree", 3),
            kwargs.get("edge_probability", 0.3),
            kwargs.get("num_roots"),
        )
    raise ValueError(f"Unknown graph_type: {graph_type}")


# ──────────────────────────────────────────────────────────────
# Label randomisation
# ──────────────────────────────────────────────────────────────

def randomize_node_labels(
    g: nx.DiGraph, max_label_value: int = 100
) -> Tuple[nx.DiGraph, Dict[int, int]]:
    original = list(g.nodes)
    new_labels = random.sample(range(1, max_label_value + 1), len(original))
    mapping = dict(zip(original, new_labels))
    return nx.relabel_nodes(g, mapping, copy=True), mapping


def randomize_node_labels_fixed_set(
    g: nx.DiGraph, max_nodes: int
) -> Tuple[nx.DiGraph, Dict[int, int]]:
    original = list(g.nodes)
    if len(original) > max_nodes:
        raise ValueError(f"Graph has {len(original)} nodes but max_nodes is {max_nodes}")
    available = list(range(max_nodes))
    random.shuffle(available)
    mapping = dict(zip(original, available[: len(original)]))
    return nx.relabel_nodes(g, mapping, copy=True), mapping


# ──────────────────────────────────────────────────────────────
# Chain-specific helpers (kept for backward compat)
# ──────────────────────────────────────────────────────────────

def find_chain_end(g: nx.DiGraph, start_node: int) -> int:
    current = start_node
    while True:
        succs = list(g.successors(current))
        if not succs:
            return current
        current = succs[0]


def calculate_chain_distance(g: nx.DiGraph, start_node: int, end_node: int) -> int:
    current, dist = start_node, 0
    while current != end_node:
        succs = list(g.successors(current))
        if not succs:
            return -1
        current = succs[0]
        dist += 1
    return dist


def find_valid_start_nodes(
    g: nx.DiGraph, min_distance: int, max_distance: int = None
) -> List[int]:
    valid = []
    for node in g.nodes:
        end = find_chain_end(g, node)
        d = calculate_chain_distance(g, node, end)
        if d >= min_distance and (max_distance is None or d <= max_distance):
            valid.append(node)
    return valid


# ──────────────────────────────────────────────────────────────
# Polynomial equation generation
# ──────────────────────────────────────────────────────────────

def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _sample_coefficient(max_coefficient: int) -> int:
    coeff = random.randint(1, max_coefficient)
    if max_coefficient > 1 and random.random() < 0.3:
        coeff = -coeff
    return coeff


def _sample_nonzero_constant(max_constant: int) -> int:
    if max_constant <= 0:
        return 0
    value = random.randint(1, max_constant)
    return value if random.random() < 0.5 else -value


def _generate_probabilistic_pairwise_equation(
    target_var: int,
    parent_vars: List[int],
    max_coefficient: int,
    max_constant: int,
    single_variable_term_probability: float,
    pairwise_product_term_probability: float,
    constant_term_probability: float,
) -> PolynomialEquation:
    """
    Include each x_i, each pair x_i * x_j, and the constant independently.
    Missing parents are added back as single-variable terms so every incoming
    edge source appears in the equation.
    """
    _validate_probability("single_variable_term_probability", single_variable_term_probability)
    _validate_probability("pairwise_product_term_probability", pairwise_product_term_probability)
    _validate_probability("constant_term_probability", constant_term_probability)

    monomials: List[Monomial] = []
    covered_parents: Set[int] = set()

    for left, right in combinations(parent_vars, 2):
        if random.random() < pairwise_product_term_probability:
            monomials.append(
                Monomial(
                    coefficient=_sample_coefficient(max_coefficient),
                    variables={left: 1, right: 1},
                )
            )
            covered_parents.update((left, right))

    for parent in parent_vars:
        if random.random() < single_variable_term_probability:
            monomials.append(
                Monomial(
                    coefficient=_sample_coefficient(max_coefficient),
                    variables={parent: 1},
                )
            )
            covered_parents.add(parent)

    for parent in parent_vars:
        if parent not in covered_parents:
            monomials.append(
                Monomial(
                    coefficient=_sample_coefficient(max_coefficient),
                    variables={parent: 1},
                )
            )

    constant = (
        _sample_nonzero_constant(max_constant)
        if random.random() < constant_term_probability
        else 0
    )
    return PolynomialEquation(target_var=target_var, monomials=monomials, constant=constant)


def generate_polynomial_equation(
    target_var: int,
    parent_vars: List[int],
    max_degree: int = 2,
    max_coefficient: int = 2,
    max_terms: int = 2,
    terms_equal_in_degree: bool = False,
    max_constant: int = 5,
    probabilistic_pairwise_terms: bool = False,
    single_variable_term_probability: float = 0.5,
    pairwise_product_term_probability: float = 0.5,
    constant_term_probability: float = 0.5,
) -> PolynomialEquation:
    if not parent_vars:
        raise ValueError("Non-root node must have parent variables")

    if probabilistic_pairwise_terms:
        return _generate_probabilistic_pairwise_equation(
            target_var,
            parent_vars,
            max_coefficient,
            max_constant,
            single_variable_term_probability,
            pairwise_product_term_probability,
            constant_term_probability,
        )

    if terms_equal_in_degree:
        # Force one term per incoming edge (parent variable).
        num_terms = len(parent_vars)
    else:
        num_terms = random.randint(1, max_terms)

    shuffled = list(parent_vars)
    random.shuffle(shuffled)
    if terms_equal_in_degree:
        # In strict mode we enforce exactly one parent-variable per term.
        term_parents: List[List[int]] = [[p] for p in shuffled]
    else:
        # Round-robin parents into terms so every parent appears at least once.
        term_parents = [[] for _ in range(num_terms)]
        for i, p in enumerate(shuffled):
            term_parents[i % num_terms].append(p)
        # Fill any empty terms (single-parent, multi-term → repeated variable)
        for tp in term_parents:
            if not tp:
                tp.append(random.choice(parent_vars))

    monomials: List[Monomial] = []
    for tp in term_parents:
        variables = {v: random.randint(1, max_degree) for v in tp}
        coeff = _sample_coefficient(max_coefficient)
        monomials.append(Monomial(coefficient=coeff, variables=variables))

    constant = random.randint(-max_constant, max_constant)
    return PolynomialEquation(target_var=target_var, monomials=monomials, constant=constant)


# ──────────────────────────────────────────────────────────────
# Ancestor-subgraph–first generation
# ──────────────────────────────────────────────────────────────

def generate_ancestor_subgraph(
    ancestor_size: int,
    graph_type: str,
    max_in_degree: int = 3,
    max_out_degree: int = 3,
    edge_probability: float = 0.3,
    num_roots: int = None,
    min_depth: int = None,
    max_depth: int = None,
    force_unique_topo_order: bool = False,
) -> Tuple[nx.DiGraph, Optional[int]]:
    """
    Generate a DAG of *ancestor_size* nodes where **every** non-root node
    is an ancestor of the target (or the target itself).

    The target is always the highest-numbered node (topological sink).

    When ``force_unique_topo_order`` is True (dag graph_type only), edges
    ``i → i+1`` are added *first* so the topological order is forced to be
    ``0, 1, …, ancestor_size-1``. Extra random edges are then added within
    the remaining ``max_in_degree - 1`` budget per node. Only one root is
    produced in this mode.

    Returns ``(graph, target_id)`` or ``(graph, None)`` when constraints
    cannot be satisfied.
    """
    g = nx.DiGraph()
    if ancestor_size <= 0:
        return g, None
    if ancestor_size == 1:
        g.add_node(0)
        return g, 0

    target = ancestor_size - 1
    for i in range(ancestor_size):
        g.add_node(i)

    if graph_type == "tree":
        # Ancestor subgraph of a tree leaf = simple chain root → target
        for i in range(ancestor_size - 1):
            g.add_edge(i, i + 1)

    elif graph_type == "reverse_tree":
        # Each non-target node gets exactly 1 outgoing edge toward a later node
        for i in range(ancestor_size - 1):
            g.add_edge(i, random.randint(i + 1, target))

    elif force_unique_topo_order:
        # Chain edges first → forces a unique topological order (0,1,…,n-1).
        # max_in_degree must leave budget for these mandatory edges.
        if max_in_degree < 1:
            return g, None
        for i in range(ancestor_size - 1):
            g.add_edge(i, i + 1)
        # Now each node j >= 1 has in_degree 1 already; can accept up to
        # max_in_degree - 1 additional parents from {0, …, j-2}.
        for j in range(2, ancestor_size):
            extra_possible = list(range(0, j - 1))
            remaining_budget = max_in_degree - g.in_degree(j)
            if remaining_budget <= 0 or not extra_possible:
                continue
            # Sample extra parents with probability edge_probability,
            # capped at remaining_budget.
            chosen = [p for p in extra_possible if random.random() < edge_probability]
            if len(chosen) > remaining_budget:
                chosen = random.sample(chosen, remaining_budget)
            for p in chosen:
                g.add_edge(p, j)

    else:  # dag (default)
        n_roots = num_roots if num_roots is not None else max(1, ancestor_size // 5)
        n_roots = max(1, min(n_roots, ancestor_size - 1))
        for j in range(n_roots, ancestor_size):
            possible = list(range(j))
            n_par = random.randint(1, min(max_in_degree, len(possible)))
            parents = set(random.sample(possible, n_par))
            for p in possible:
                if p not in parents and random.random() < edge_probability and len(parents) < max_in_degree:
                    parents.add(p)
            for p in parents:
                g.add_edge(p, j)
        # Ensure every non-target node can reach the target (process in
        # reverse so later nodes are already guaranteed reachable).
        # Only pick destinations that still have room under max_in_degree.
        for i in range(ancestor_size - 2, -1, -1):
            if g.out_degree(i) == 0:
                candidates = [
                    n for n in range(i + 1, target + 1)
                    if g.in_degree(n) < max_in_degree
                ]
                if not candidates:
                    return g, None  # can't wire i to target without violating in-degree
                g.add_edge(i, random.choice(candidates))

    # Depth filter
    if min_depth is not None or max_depth is not None:
        depths: Dict[int, int] = {}
        for node in nx.topological_sort(g):
            preds = list(g.predecessors(node))
            depths[node] = 0 if not preds else max(depths[p] for p in preds) + 1
        d = depths.get(target, 0)
        if (min_depth is not None and d < min_depth) or (max_depth is not None and d > max_depth):
            return g, None

    return g, target


def generate_bucketed_dag(
    n_nodes: int,
    ancestor_depth: int,
    max_in_degree: int = 3,
    edge_probability: float = 0.3,
) -> Tuple[nx.DiGraph, Optional[int]]:
    """
    Build a DAG of ``n_nodes`` whose target's ancestor-graph depth is
    **exactly** ``ancestor_depth - 1`` (i.e. the longest root→target path
    has ``ancestor_depth`` nodes / ``ancestor_depth - 1`` edges).

    Construction:
      1. Partition the ``n_nodes`` nodes into ``ancestor_depth`` buckets,
         each holding at least 1 node. Bucket 0 is the "root level" and the
         last bucket is the "target level".
      2. Pick one *spine* node per bucket and chain them together
         ``spine[0] → spine[1] → … → spine[D-1]``. The last spine node is
         the **target**.
      3. Every non-spine node in bucket ``i ≥ 1`` receives at least one
         parent from a strictly earlier bucket so the graph is connected.
      4. Add random extra edges that strictly **descend bucket levels**
         (from bucket ``i`` to bucket ``j`` with ``j > i``), each with
         probability ``edge_probability`` and capped by ``max_in_degree``.

    Because every edge strictly increases bucket index, the longest path
    is bounded above by ``ancestor_depth - 1``; the spine forces it to be
    at least ``ancestor_depth - 1``, so equality holds.

    Non-target nodes in the last bucket are sinks and therefore *not*
    ancestors of the target — they act as distractors at the deepest
    level, which is permitted by the bucket-only constraint.

    Returns ``(graph, target_id)`` or ``(graph, None)`` when the inputs
    cannot satisfy the constraints (e.g. ``ancestor_depth > n_nodes``).
    """
    g = nx.DiGraph()
    if n_nodes <= 0 or ancestor_depth <= 0:
        return g, None
    if ancestor_depth > n_nodes:
        return g, None  # need ≥1 node per bucket
    if max_in_degree < 1:
        return g, None

    # ── 1. Partition n_nodes into ancestor_depth buckets, each ≥1
    bucket_sizes = [1] * ancestor_depth
    for _ in range(n_nodes - ancestor_depth):
        bucket_sizes[random.randrange(ancestor_depth)] += 1

    buckets: List[List[int]] = []
    cur = 0
    for sz in bucket_sizes:
        buckets.append(list(range(cur, cur + sz)))
        cur += sz

    for node_id in range(n_nodes):
        g.add_node(node_id)

    # ── 2. Spine chain through every bucket (forces depth ≥ D-1)
    spine = [random.choice(b) for b in buckets]
    for u, v in zip(spine, spine[1:]):
        g.add_edge(u, v)
    target = spine[-1]

    # ── 3. Mandatory parent for every non-root node not yet wired.
    #     Spine nodes in buckets ≥ 1 already have in_degree ≥ 1.
    for i in range(1, ancestor_depth):
        earlier = [n for j in range(i) for n in buckets[j]]
        for node in buckets[i]:
            if g.in_degree(node) >= 1:
                continue
            parent = random.choice(earlier)
            g.add_edge(parent, node)

    # ── 4. Extra random edges, strictly across bucket boundaries.
    #     Each edge respects in-degree budget; every edge increases bucket
    #     index by ≥1 so the depth bound stays at D-1.
    for i in range(ancestor_depth - 1):
        later = [n for j in range(i + 1, ancestor_depth) for n in buckets[j]]
        for src in buckets[i]:
            for dst in later:
                if g.has_edge(src, dst):
                    continue
                if g.in_degree(dst) >= max_in_degree:
                    continue
                if random.random() < edge_probability:
                    g.add_edge(src, dst)

    return g, target


def add_distractor_nodes(
    g: nx.DiGraph,
    ancestor_nodes: Set[int],
    n_distractors: int,
    max_in_degree: int = 3,
    edge_probability: float = 0.3,
    protected_nodes: Set[int] = None,
) -> None:
    """
    Append *n_distractors* nodes that **cannot** enlarge the ancestor set.

    * Edges are only added **from** ancestor / earlier-distractor nodes **to**
      the new distractor — never in the reverse direction.
    * *protected_nodes* (e.g. the target when ``target_sink_only=True``) are
      excluded from the pool of possible parents so they stay sinks.
    * ~20 % of distractors become roots (constant assignments).
    """
    if n_distractors <= 0:
        return
    start_id = max(g.nodes) + 1
    available_ancestors = sorted(ancestor_nodes - (protected_nodes or set()))

    for idx in range(n_distractors):
        d = start_id + idx
        g.add_node(d)
        if random.random() < 0.2:
            continue  # distractor root
        possible = available_ancestors + list(range(start_id, d))
        if not possible:
            continue
        n_par = random.randint(1, min(max_in_degree, len(possible)))
        parents = set(random.sample(possible, n_par))
        for p in possible:
            if p not in parents and random.random() < edge_probability and len(parents) < max_in_degree:
                parents.add(p)
        for p in parents:
            g.add_edge(p, d)


# ──────────────────────────────────────────────────────────────
# Depth helpers
# ──────────────────────────────────────────────────────────────

def compute_node_depths(g: nx.DiGraph) -> Dict[int, int]:
    """Longest path from any root to each node (topological DP)."""
    depths: Dict[int, int] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        depths[node] = 0 if not preds else max(depths[p] for p in preds) + 1
    return depths


def select_target_node(
    g: nx.DiGraph,
    root_nodes: Set[int],
    target_sink_only: bool = True,
    min_depth: int = None,
    max_depth: int = None,
) -> Optional[int]:
    depths = compute_node_depths(g)
    candidates = []
    for node in g.nodes:
        if node in root_nodes:
            continue
        if target_sink_only and g.out_degree(node) > 0:
            continue
        d = depths.get(node, 0)
        if min_depth is not None and d < min_depth:
            continue
        if max_depth is not None and d > max_depth:
            continue
        candidates.append(node)
    return random.choice(candidates) if candidates else None


# ──────────────────────────────────────────────────────────────
# String formatting
# ──────────────────────────────────────────────────────────────

def format_equation_system(
    root_values: Dict[int, int],
    equations: Dict[int, PolynomialEquation],
    target_node: int,
    shuffle: bool = True,
    modulus: Optional[int] = None,
) -> str:
    stmts: List[str] = []
    for var, val in root_values.items():
        stmts.append(f"x_{var} = {val}")
    for eq in equations.values():
        stmts.append(eq.to_string(modulus=modulus))
    if shuffle:
        random.shuffle(stmts)
    find_str = f"Find x_{target_node}." if modulus is None else f"Find x_{target_node} (mod {modulus})."
    return ". ".join(stmts) + ". " + find_str


# ──────────────────────────────────────────────────────────────
# Sampling helpers (unchanged)
# ──────────────────────────────────────────────────────────────

def _calculate_size_weights(min_nodes: int, max_nodes: int, strategy: str) -> dict:
    sizes = list(range(min_nodes, max_nodes + 1))
    if strategy in ("uniform_size", "stratified"):
        return {s: 1.0 for s in sizes}
    if strategy == "exponential":
        return {s: 2.0 ** (s - min_nodes) for s in sizes}
    if strategy == "quadratic":
        return {s: s ** 2 for s in sizes}
    if strategy == "proportional":
        return {s: 2.0 ** (s * (s - 1) // 2 / 5) for s in sizes}
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _sample_graph_size(min_nodes, max_nodes, strategy, weights=None):
    if strategy in ("uniform_size", "stratified"):
        return random.randint(min_nodes, max_nodes)
    if weights is None:
        weights = _calculate_size_weights(min_nodes, max_nodes, strategy)
    sizes = list(weights.keys())
    total = sum(weights.values())
    probs = [w / total for w in weights.values()]
    return int(np.random.choice(sizes, p=probs))


# ──────────────────────────────────────────────────────────────
# Single-sample generators
# ──────────────────────────────────────────────────────────────

def _generate_chain_sample(
    n_nodes: int,
    *,
    min_chain_length: int,
    max_chain_length: int,
    min_distance: int,
    max_distance: Optional[int],
    operations: List[str],
    min_constant: int,
    max_constant: int,
    min_start_value: int,
    max_start_value: int,
    randomize_labels_flag: bool,
    max_label_value: int,
    fixed_label_set: bool,
    max_nodes: int,
    max_value: Optional[int],
    modulus: Optional[int],
    shuffle_equations: bool,
) -> Optional[Tuple[str, str]]:
    """Chain generation — backward-compatible with the original code."""
    g = generate_chain_graph(n_nodes, min_chain_length, max_chain_length)
    valid = find_valid_start_nodes(g, min_distance, max_distance)
    if not valid:
        return None
    start_node = random.choice(valid)

    if randomize_labels_flag:
        if fixed_label_set:
            g, mapping = randomize_node_labels_fixed_set(g, max_nodes)
        else:
            g, mapping = randomize_node_labels(g, max_label_value)
        start_node = mapping[start_node]

    equations: Dict[int, PolynomialEquation] = {}
    for u, v in g.edges():
        op = random.choice(operations) if operations else "+"
        c = random.randint(min_constant, max_constant)
        if op == "-":
            c = -c
        equations[v] = PolynomialEquation(
            target_var=v,
            monomials=[Monomial(coefficient=1, variables={u: 1})],
            constant=c,
        )

    start_value = random.randint(min_start_value, max_start_value)
    if modulus is not None:
        start_value = start_value % modulus
    root_values = {start_node: start_value}
    target = find_chain_end(g, start_node)

    values: Dict[int, int] = {start_node: start_value}
    for node in nx.topological_sort(g):
        if node in values or node not in equations:
            continue
        eq = equations[node]
        if all(p in values for p in eq.get_parent_variables()):
            val = eq.evaluate(values)
            values[node] = val % modulus if modulus is not None else val

    if target not in values:
        return None
    if modulus is None and max_value is not None and any(abs(v) > max_value for v in values.values()):
        return None

    inp = format_equation_system(root_values, equations, target, shuffle_equations, modulus=modulus)
    return inp, str(values[target])


def _generate_general_sample(
    n_nodes: int,
    graph_type: str,
    *,
    max_in_degree: int,
    max_out_degree: int,
    edge_probability: float,
    num_roots: Optional[int],
    max_degree: int,
    max_coefficient: int,
    max_terms: int,
    terms_equal_in_degree: bool,
    max_constant: int,
    probabilistic_pairwise_terms: bool,
    single_variable_term_probability: float,
    pairwise_product_term_probability: float,
    constant_term_probability: float,
    min_start_value: int,
    max_start_value: int,
    max_value: Optional[int],
    modulus: Optional[int],
    target_sink_only: bool,
    min_depth: Optional[int],
    max_depth: Optional[int],
    ancestor_size: Optional[int],
    force_unique_topo_order: bool,
    ancestor_depth: Optional[int],
    randomize_labels_flag: bool,
    max_label_value: int,
    fixed_label_set: bool,
    max_nodes: int,
    shuffle_equations: bool,
) -> Optional[Tuple[str, str]]:
    """Tree / reverse_tree / dag generation with polynomial equations."""

    # ── Phase 1: build graph & (optionally) pre-select target ─────
    pre_target: Optional[int] = None

    if ancestor_depth is not None:
        # Bucketed depth-controlled DAG. Spans the whole graph (no
        # separate distractor pass): non-target nodes that happen to
        # live in the last bucket are themselves "deep distractors".
        if graph_type != "dag":
            raise ValueError(
                "ancestor_depth is only supported for graph_type='dag'"
            )
        if force_unique_topo_order:
            raise ValueError(
                "ancestor_depth and force_unique_topo_order are mutually "
                "exclusive — pick one strategy for ancestor-depth control"
            )
        g, pre_target = generate_bucketed_dag(
            n_nodes,
            ancestor_depth,
            max_in_degree=max_in_degree,
            edge_probability=edge_probability,
        )
        if pre_target is None:
            return None
    elif ancestor_size is not None:
        anc_size = min(ancestor_size, n_nodes)
        g, pre_target = generate_ancestor_subgraph(
            anc_size, graph_type,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            edge_probability=edge_probability,
            num_roots=num_roots,
            min_depth=min_depth,
            max_depth=max_depth,
            force_unique_topo_order=force_unique_topo_order,
        )
        if pre_target is None:
            return None
        n_distractors = n_nodes - anc_size
        if n_distractors > 0:
            protected = {pre_target} if target_sink_only else set()
            add_distractor_nodes(
                g, set(g.nodes), n_distractors,
                max_in_degree=max_in_degree,
                edge_probability=edge_probability,
                protected_nodes=protected,
            )
    else:
        g = generate_graph(
            n_nodes, graph_type,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            edge_probability=edge_probability,
            num_roots=num_roots,
        )

    # ── Phase 2: identify roots & relabel ─────────────────────────
    root_nodes_orig = set(n for n in g.nodes if g.in_degree(n) == 0)
    if not root_nodes_orig:
        return None

    if randomize_labels_flag:
        if fixed_label_set:
            g, mapping = randomize_node_labels_fixed_set(g, max_nodes)
        else:
            g, mapping = randomize_node_labels(g, max_label_value)
        root_nodes = set(mapping[n] for n in root_nodes_orig)
        if pre_target is not None:
            pre_target = mapping[pre_target]
    else:
        root_nodes = root_nodes_orig

    # ── Phase 3: equations & values ───────────────────────────────
    root_values = {r: random.randint(min_start_value, max_start_value) for r in root_nodes}
    if modulus is not None:
        root_values = {r: v % modulus for r, v in root_values.items()}

    equations: Dict[int, PolynomialEquation] = {}
    for node in nx.topological_sort(g):
        if node in root_nodes:
            continue
        parents = sorted(g.predecessors(node))
        if not parents:
            continue
        equations[node] = generate_polynomial_equation(
            node,
            parents,
            max_degree,
            max_coefficient,
            max_terms,
            terms_equal_in_degree,
            max_constant,
            probabilistic_pairwise_terms,
            single_variable_term_probability,
            pairwise_product_term_probability,
            constant_term_probability,
        )

    values: Dict[int, int] = dict(root_values)
    for node in nx.topological_sort(g):
        if node in values:
            continue
        if node in equations:
            val = equations[node].evaluate(values)
            values[node] = val % modulus if modulus is not None else val

    if modulus is None and max_value is not None and any(abs(v) > max_value for v in values.values()):
        return None

    # ── Phase 4: select target ────────────────────────────────────
    target = pre_target if pre_target is not None else select_target_node(
        g, root_nodes, target_sink_only, min_depth, max_depth
    )
    if target is None or target not in values:
        return None

    inp = format_equation_system(root_values, equations, target, shuffle_equations, modulus=modulus)
    return inp, str(values[target])


# ──────────────────────────────────────────────────────────────
# Main public API
# ──────────────────────────────────────────────────────────────

def generate_unique_equation_samples(
    n_samples: int,
    min_nodes: int = 5,
    max_nodes: int = 10,
    graph_type: str = "chain",
    # Chain-specific (backward compat)
    min_chain_length: int = 2,
    max_chain_length: int = 6,
    min_distance: int = 1,
    max_distance: int = None,
    operations: List[str] = None,
    # DAG / tree params
    max_in_degree: int = 3,
    max_out_degree: int = 3,
    edge_probability: float = 0.3,
    num_roots: int = None,
    # Polynomial equation params
    max_degree: int = 1,
    max_coefficient: int = 1,
    max_terms: int = 1,
    terms_equal_in_degree: bool = False,
    probabilistic_pairwise_terms: bool = False,
    single_variable_term_probability: float = 0.5,
    pairwise_product_term_probability: float = 0.5,
    constant_term_probability: float = 0.5,
    min_constant: int = 1,
    max_constant: int = 3,
    # Value params
    min_start_value: int = 1,
    max_start_value: int = 10,
    max_value: int = 10000,
    modulus: int = None,
    # Target selection
    target_sink_only: bool = True,
    min_depth: int = None,
    max_depth: int = None,
    # Ancestor-subgraph size control (tree / reverse_tree / dag only)
    min_ancestor_nodes: int = None,
    max_ancestor_nodes: int = None,
    # Force a unique topological order (dag only): guarantees edges i→i+1
    # so the ancestor subgraph is a chain plus optional extra parents.
    force_unique_topo_order: bool = False,
    # Bucketed depth control (dag only): partition ALL nodes into
    # `ancestor_depth` buckets (each ≥1 node), wire a spine through them,
    # and only allow edges that strictly descend bucket levels. This makes
    # the target's ancestor-graph depth exactly ``ancestor_depth - 1``.
    # Enable via min+max endpoints (contiguous range) OR via an explicit
    # `ancestor_depths` list (sparse). The two are mutually exclusive.
    min_ancestor_depth: int = None,
    max_ancestor_depth: int = None,
    ancestor_depths: Optional[List[int]] = None,
    # Labels
    randomize_labels: bool = True,
    max_label_value: int = 100,
    fixed_label_set: bool = False,
    # Sampling
    sampling_strategy: str = "uniform_size",
    shuffle_equations: bool = True,
) -> List[Tuple[str, str]]:
    """
    Generate unique equation-reasoning samples.

    Supports four graph topologies via *graph_type*:
      chain          – independent chains (original behaviour)
      tree           – single-rooted tree (ancestor path is a line to the root)
      reverse_tree   – edges converge to single sink (ancestor search is the subtree)
      dag            – arbitrary DAG with multiple roots and polynomial equations
    """
    if operations is None:
        operations = ["+", "-"]

    if ancestor_depths is not None:
        if min_ancestor_depth is not None or max_ancestor_depth is not None:
            raise ValueError(
                "ancestor_depths is mutually exclusive with "
                "min_ancestor_depth/max_ancestor_depth"
            )
        depths = sorted({int(d) for d in ancestor_depths})
        if not depths or any(d < 1 for d in depths):
            raise ValueError(
                f"Invalid ancestor_depths: must be a non-empty list of "
                f"positive ints, got {ancestor_depths}"
            )
        use_bucketed_depth = True
    elif min_ancestor_depth is not None and max_ancestor_depth is not None:
        if min_ancestor_depth < 1 or max_ancestor_depth < min_ancestor_depth:
            raise ValueError(
                f"Invalid ancestor depth range: "
                f"min={min_ancestor_depth}, max={max_ancestor_depth}"
            )
        depths = list(range(min_ancestor_depth, max_ancestor_depth + 1))
        use_bucketed_depth = True
    else:
        depths = []
        use_bucketed_depth = False

    if use_bucketed_depth:
        if graph_type != "dag":
            raise ValueError(
                "ancestor depth control requires graph_type='dag'"
            )
        if force_unique_topo_order:
            raise ValueError(
                "Bucketed depth control and force_unique_topo_order are mutually exclusive"
            )

    seen: Set[str] = set()
    samples: List[Tuple[str, str]] = []
    max_attempts = n_samples * 200

    def _gen(
        n: int,
        anc_size: Optional[int] = None,
        anc_depth: Optional[int] = None,
    ) -> Optional[Tuple[str, str]]:
        if graph_type == "chain":
            return _generate_chain_sample(
                n,
                min_chain_length=min_chain_length,
                max_chain_length=max_chain_length,
                min_distance=min_distance,
                max_distance=max_distance,
                operations=operations,
                min_constant=min_constant,
                max_constant=max_constant,
                min_start_value=min_start_value,
                max_start_value=max_start_value,
                randomize_labels_flag=randomize_labels,
                max_label_value=max_label_value,
                fixed_label_set=fixed_label_set,
                max_nodes=max_nodes,
                max_value=max_value,
                modulus=modulus,
                shuffle_equations=shuffle_equations,
            )
        return _generate_general_sample(
            n,
            graph_type,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            edge_probability=edge_probability,
            num_roots=num_roots,
            max_degree=max_degree,
            max_coefficient=max_coefficient,
            max_terms=max_terms,
            terms_equal_in_degree=terms_equal_in_degree,
            probabilistic_pairwise_terms=probabilistic_pairwise_terms,
            single_variable_term_probability=single_variable_term_probability,
            pairwise_product_term_probability=pairwise_product_term_probability,
            constant_term_probability=constant_term_probability,
            max_constant=max_constant,
            min_start_value=min_start_value,
            max_start_value=max_start_value,
            max_value=max_value,
            modulus=modulus,
            target_sink_only=target_sink_only,
            min_depth=min_depth,
            max_depth=max_depth,
            ancestor_size=anc_size,
            force_unique_topo_order=force_unique_topo_order,
            ancestor_depth=anc_depth,
            randomize_labels_flag=randomize_labels,
            max_label_value=max_label_value,
            fixed_label_set=fixed_label_set,
            max_nodes=max_nodes,
            shuffle_equations=shuffle_equations,
        )

    def _collect(result):
        if result is None:
            return False
        inp, ans = result
        if inp in seen:
            return False
        seen.add(inp)
        samples.append((inp, ans))
        return True

    use_ancestor_strat = (
        graph_type != "chain"
        and min_ancestor_nodes is not None
        and max_ancestor_nodes is not None
        and not use_bucketed_depth
    )
    ancestor_sizes = list(range(min_ancestor_nodes, max_ancestor_nodes + 1)) if use_ancestor_strat else [None]

    def _sample_anc_depth() -> Optional[int]:
        if not use_bucketed_depth:
            return None
        return random.choice(depths)

    attempts = 0
    if sampling_strategy == "stratified":
        sizes = list(range(min_nodes, max_nodes + 1))
        if use_bucketed_depth:
            stratify_buckets = [(sz, d) for sz in sizes for d in depths]
            per_bucket = n_samples // len(stratify_buckets)
            remainder = n_samples % len(stratify_buckets)
            print(
                f"Stratified sampling (bucketed depth): {per_bucket} per bucket, "
                f"{remainder} extra, {len(sizes)} sizes x {len(depths)} depths "
                f"{depths}"
            )
            for i, (size, d) in enumerate(stratify_buckets):
                target_count = per_bucket + (1 if i < remainder else 0)
                count = 0
                while count < target_count and attempts < max_attempts:
                    attempts += 1
                    if _collect(_gen(size, anc_depth=d)):
                        count += 1
                print(f"  Size {size}, depth {d}: {count} samples")
        else:
            stratify_buckets = [(sz, anc) for sz in sizes for anc in ancestor_sizes]
            per_bucket = n_samples // len(stratify_buckets)
            remainder = n_samples % len(stratify_buckets)
            print(f"Stratified sampling: {per_bucket} per bucket, {remainder} extra, "
                  f"{len(sizes)} sizes x {len(ancestor_sizes)} ancestor levels")
            for i, (size, anc) in enumerate(stratify_buckets):
                target_count = per_bucket + (1 if i < remainder else 0)
                count = 0
                while count < target_count and attempts < max_attempts:
                    attempts += 1
                    if _collect(_gen(size, anc_size=anc)):
                        count += 1
                print(f"  Size {size}, ancestors {anc}: {count} samples")
    elif use_ancestor_strat:
        per_anc = n_samples // len(ancestor_sizes)
        remainder = n_samples % len(ancestor_sizes)
        print(f"Using {sampling_strategy} sampling with ancestor stratification "
              f"(graph_type={graph_type}, {per_anc} per ancestor level)")
        for i, anc in enumerate(ancestor_sizes):
            target_count = per_anc + (1 if i < remainder else 0)
            count = 0
            while count < target_count and attempts < max_attempts:
                attempts += 1
                n = _sample_graph_size(min_nodes, max_nodes, sampling_strategy, _calculate_size_weights(min_nodes, max_nodes, sampling_strategy))
                if _collect(_gen(n, anc_size=anc)):
                    count += 1
            print(f"  Ancestors {anc}: {count} samples")
    else:
        weights = _calculate_size_weights(min_nodes, max_nodes, sampling_strategy)
        if use_bucketed_depth:
            print(
                f"Using {sampling_strategy} sampling with bucketed depth "
                f"(graph_type={graph_type}, depths={depths})"
            )
        else:
            print(f"Using {sampling_strategy} sampling (graph_type={graph_type})")
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            n = _sample_graph_size(min_nodes, max_nodes, sampling_strategy, weights)
            _collect(_gen(n, anc_depth=_sample_anc_depth()))

    if len(samples) < n_samples:
        print(f"Warning: only generated {len(samples)}/{n_samples} samples after {attempts} attempts")

    # Print stats
    import re as _re
    size_counts: Dict[int, int] = {}
    for inp, _ in samples:
        refs = _re.findall(r"x_(\d+)", inp)
        if refs:
            size_counts.setdefault(len(set(refs)), 0)
            size_counts[len(set(refs))] += 1

    print(f"\nGenerated {len(samples)} samples (graph_type={graph_type})")
    for sz in sorted(size_counts):
        pct = size_counts[sz] / len(samples) * 100
        print(f"  {sz} variables: {size_counts[sz]} ({pct:.1f}%)")
    if probabilistic_pairwise_terms:
        terms_desc = (
            "probabilistic_pairwise "
            f"(single={single_variable_term_probability}, "
            f"pair={pairwise_product_term_probability}, "
            f"const={constant_term_probability})"
        )
    else:
        terms_desc = "in_degree" if terms_equal_in_degree else f"<={max_terms}"
    print(f"  Polynomial params: degree<={max_degree}, coeff<={max_coefficient}, terms={terms_desc}, const<={max_constant}")
    print(f"  Start values: {min_start_value}-{max_start_value}, max_value={max_value}")

    return samples


# ──────────────────────────────────────────────────────────────
# Convenience / backward-compat wrappers
# ──────────────────────────────────────────────────────────────

def create_equation_dataset_dict(n_samples: int, **kwargs) -> dict:
    samples = generate_unique_equation_samples(n_samples, **kwargs)
    return {"input": [s[0] for s in samples], "target": [s[1] for s in samples]}


def create_dag_dataset_dict(*args, **kwargs):
    """Deprecated alias."""
    return create_equation_dataset_dict(*args, **kwargs)


def generate_unique_chain_samples_simplified(*args, **kwargs):
    """Deprecated alias."""
    return generate_unique_equation_samples(*args, **kwargs)


# ──────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for gtype in ("chain", "tree", "reverse_tree", "dag"):
        print(f"\n{'='*60}")
        print(f"  Graph type: {gtype}")
        print(f"{'='*60}")
        kw = dict(
            n_samples=3,
            min_nodes=10,
            max_nodes=10,
            graph_type=gtype,
            randomize_labels=True,
            max_label_value=20,
            min_start_value=1,
            max_start_value=5,
            shuffle_equations=True,
        )
        if gtype == "chain":
            kw.update(min_chain_length=10, max_chain_length=10, min_distance=3, max_distance=9)
        else:
            kw.update(
                max_degree=2,
                max_coefficient=2,
                max_terms=2,
                max_constant=3,
                target_sink_only=True,
            )
            if gtype == "dag":
                kw.update(max_in_degree=3, edge_probability=0.3)
            if gtype == "tree":
                kw.update(max_out_degree=3)
            if gtype == "reverse_tree":
                kw.update(max_in_degree=3)

        samples = generate_unique_equation_samples(**kw)
        for i, (inp, tgt) in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Input:  {inp}")
            print(f"  Target: {tgt}")

    # Ancestor-subgraph size control
    for gtype in ("tree", "reverse_tree", "dag"):
        print(f"\n{'='*60}")
        print(f"  Graph type: {gtype}  (ancestor control: 4–5 out of 12 nodes)")
        print(f"{'='*60}")
        kw = dict(
            n_samples=3,
            min_nodes=12,
            max_nodes=12,
            graph_type=gtype,
            min_ancestor_nodes=4,
            max_ancestor_nodes=5,
            max_degree=2,
            max_coefficient=2,
            max_terms=2,
            max_constant=3,
            target_sink_only=True,
            randomize_labels=True,
            max_label_value=20,
            min_start_value=1,
            max_start_value=5,
            shuffle_equations=True,
        )
        if gtype == "dag":
            kw.update(max_in_degree=3, edge_probability=0.3)
        if gtype == "tree":
            kw.update(max_out_degree=3)
        if gtype == "reverse_tree":
            kw.update(max_in_degree=3)
        samples = generate_unique_equation_samples(**kw)
        for i, (inp, tgt) in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"  Input:  {inp}")
            print(f"  Target: {tgt}")

    # Bucketed depth control (dag only): force ancestor-graph depth
    print(f"\n{'='*60}")
    print(f"  DAG bucketed depth (12 nodes, depth ∈ {{3, 4, 5}})")
    print(f"{'='*60}")
    samples = generate_unique_equation_samples(
        n_samples=4,
        min_nodes=12,
        max_nodes=12,
        graph_type="dag",
        max_in_degree=3,
        edge_probability=0.3,
        max_degree=2,
        max_coefficient=2,
        max_terms=2,
        max_constant=3,
        min_ancestor_depth=3,
        max_ancestor_depth=5,
        target_sink_only=True,
        randomize_labels=True,
        max_label_value=20,
        min_start_value=1,
        max_start_value=5,
        shuffle_equations=True,
    )
    for i, (inp, tgt) in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Input:  {inp}")
        print(f"  Target: {tgt}")

    # Modular arithmetic
    print(f"\n{'='*60}")
    print(f"  DAG with modulus=10")
    print(f"{'='*60}")
    samples = generate_unique_equation_samples(
        n_samples=3, min_nodes=10, max_nodes=10,
        graph_type="dag",
        max_degree=2, max_coefficient=3, max_terms=3, max_constant=5,
        max_in_degree=3, edge_probability=0.3,
        modulus=10,
        target_sink_only=True,
        randomize_labels=True, max_label_value=20,
        min_start_value=1, max_start_value=9,
        shuffle_equations=True,
    )
    for i, (inp, tgt) in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Input:  {inp}")
        print(f"  Target: {tgt}")
