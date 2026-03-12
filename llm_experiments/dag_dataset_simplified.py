import random
import networkx as nx
from typing import List, Tuple, Set, Dict, Optional
import numpy as np

def generate_independent_chains(n_nodes: int, min_chain_length: int = 2, max_chain_length: int = 6) -> Tuple[nx.DiGraph, Dict[int, int]]:
    """
    Generate a graph consisting of independent chains.
    Returns the graph and a mapping from each node to the last node in its chain.
    """
    g = nx.DiGraph()
    node_to_chain_end = {}
    current_node_id = 0
    
    # Keep generating chains until we use up all nodes
    while current_node_id < n_nodes:
        # Determine chain length (ensure we don't exceed remaining nodes)
        remaining_nodes = n_nodes - current_node_id
        chain_length = min(random.randint(min_chain_length, max_chain_length), remaining_nodes)
        
        # Create chain nodes
        chain_nodes = list(range(current_node_id, current_node_id + chain_length))
        g.add_nodes_from(chain_nodes)
        
        # Add chain edges: node[i] -> node[i+1]
        for i in range(chain_length - 1):
            g.add_edge(chain_nodes[i], chain_nodes[i + 1])
        
        # Record the chain end for all nodes in this chain
        chain_end = chain_nodes[-1]
        for node in chain_nodes:
            node_to_chain_end[node] = chain_end
        
        current_node_id += chain_length
    
    return g, node_to_chain_end

def randomize_node_labels(g: nx.DiGraph, max_label_value: int = 100) -> nx.DiGraph:
    """Randomize node labels with default max_label_value=100 for real LLMs."""
    original_nodes = list(g.nodes)
    new_labels = random.sample(range(1, max_label_value + 1), len(original_nodes))
    mapping = dict(zip(original_nodes, new_labels))
    return nx.relabel_nodes(g, mapping, copy=True)

def randomize_node_labels_fixed_set(g: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
    """Randomize node labels within a fixed set {0, 1, 2, ..., max_nodes-1}."""
    original_nodes = list(g.nodes)
    n_nodes = len(original_nodes)
    
    if n_nodes > max_nodes:
        raise ValueError(f"Graph has {n_nodes} nodes but max_nodes is {max_nodes}")
    
    # Create a random permutation of the available labels
    available_labels = list(range(max_nodes))
    random.shuffle(available_labels)
    new_labels = available_labels[:n_nodes]
    
    # Create mapping from original nodes to new labels
    mapping = dict(zip(original_nodes, new_labels))
    return nx.relabel_nodes(g, mapping, copy=True)

def chains_to_equations_string(
    g: nx.DiGraph, 
    start_node: int,
    start_value: int,
    edge_constants: Dict[Tuple[int, int], Tuple[str, int]],
    shuffle_equations: bool = True
) -> str:
    """
    Convert chains graph to equation format.
    
    Args:
        g: The graph
        start_node: The starting node
        start_value: The initial value of the start node
        edge_constants: Dictionary mapping edges to (operator, constant) tuples
        shuffle_equations: Whether to shuffle the equation order
    
    Returns:
        String like "x_3 = 10. x_1 = x_3 + 4. x_2 = x_1 - 7. Find x_2."
    """
    equations = []
    
    # Add start node equation
    equations.append(f"x_{start_node} = {start_value}")
    
    # Add equations for each edge
    for (u, v), (op, const) in edge_constants.items():
        equations.append(f"x_{v} = x_{u} {op} {const}")
    
    if shuffle_equations:
        random.shuffle(equations)
    
    # Find the end node to ask about
    end_node = find_chain_end(g, start_node)
    
    equations_str = ". ".join(equations)
    return f"{equations_str}. Find x_{end_node}."

def generate_edge_constants(
    g: nx.DiGraph,
    min_constant: int = 1,
    max_constant: int = 10,
    operations: List[str] = None
) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """
    Generate random constants and operations for each edge.
    
    Returns:
        Dictionary mapping (u, v) edge to (operator, constant) tuple
    """
    if operations is None:
        operations = ["+", "-"]
    
    edge_constants = {}
    for u, v in g.edges():
        op = random.choice(operations)
        const = random.randint(min_constant, max_constant)
        edge_constants[(u, v)] = (op, const)
    
    return edge_constants

def compute_chain_value(
    g: nx.DiGraph,
    start_node: int,
    start_value: int,
    edge_constants: Dict[Tuple[int, int], Tuple[str, int]]
) -> int:
    """
    Compute the final value at the end of the chain.
    """
    current = start_node
    value = start_value
    
    while True:
        successors = list(g.successors(current))
        if not successors:
            return value
        
        next_node = successors[0]
        op, const = edge_constants[(current, next_node)]
        
        if op == "+":
            value = value + const
        elif op == "-":
            value = value - const
        elif op == "*":
            value = value * const
        elif op == "/":
            value = value // const  # Integer division
        
        current = next_node

def find_chain_end(g: nx.DiGraph, start_node: int) -> int:
    """Find the last node in the chain starting from start_node"""
    current = start_node
    while True:
        # Get successors (outgoing edges)
        successors = list(g.successors(current))
        if not successors:
            # No outgoing edges, this is the end of the chain
            return current
        # In a chain, there should be exactly one successor
        current = successors[0]

def calculate_chain_distance(g: nx.DiGraph, start_node: int, end_node: int) -> int:
    """Calculate the distance (number of edges) from start_node to end_node along the chain"""
    current = start_node
    distance = 0
    
    while current != end_node:
        # Get successors (outgoing edges)
        successors = list(g.successors(current))
        if not successors:
            # Reached end of chain without finding end_node
            # This means end_node is not reachable from start_node
            return -1
        # In a chain, there should be exactly one successor
        current = successors[0]
        distance += 1
    
    return distance

def find_valid_start_nodes(g: nx.DiGraph, min_distance: int, max_distance: int = None) -> List[int]:
    """Find all nodes that are between min_distance and max_distance (inclusive) away from their chain end"""
    valid_nodes = []
    
    for node in g.nodes:
        chain_end = find_chain_end(g, node)
        distance = calculate_chain_distance(g, node, chain_end)
        if distance >= min_distance:
            if max_distance is None or distance <= max_distance:
                valid_nodes.append(node)
    
    return valid_nodes

def _calculate_size_weights(min_nodes: int, max_nodes: int, strategy: str) -> dict:
    """Calculate sampling weights for different graph sizes based on strategy."""
    sizes = list(range(min_nodes, max_nodes + 1))
    
    if strategy == "uniform_size":
        # Current behavior: equal probability for each size
        weights = {size: 1.0 for size in sizes}
    
    elif strategy == "stratified":
        # Equal number of samples per size (handled differently in main function)
        weights = {size: 1.0 for size in sizes}
    
    elif strategy == "exponential":
        # Weight exponentially by size to favor larger graphs
        weights = {size: 2.0 ** (size - min_nodes) for size in sizes}
    
    elif strategy == "quadratic":
        # Weight quadratically by size
        weights = {size: size ** 2 for size in sizes}
    
    elif strategy == "proportional":
        # Weight by estimated number of possible DAGs (very rough approximation)
        # This uses a heuristic based on the number of possible edges
        weights = {}
        for size in sizes:
            # Rough estimate: exponential in n^2 (number of possible edges)
            max_edges = size * (size - 1) // 2
            weights[size] = 2.0 ** (max_edges / 5)  # Scaled down to avoid overflow
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return weights

def _sample_graph_size(min_nodes: int, max_nodes: int, strategy: str, weights: dict = None) -> int:
    """Sample a graph size according to the specified strategy."""
    if strategy == "uniform_size":
        return random.randint(min_nodes, max_nodes)
    
    elif strategy == "stratified":
        # This is handled differently in the main function
        return random.randint(min_nodes, max_nodes)
    
    else:
        # Use weighted sampling
        if weights is None:
            weights = _calculate_size_weights(min_nodes, max_nodes, strategy)
        
        sizes = list(weights.keys())
        weight_values = list(weights.values())
        
        # Normalize weights to probabilities
        total_weight = sum(weight_values)
        probabilities = [w / total_weight for w in weight_values]
        
        # Sample according to probabilities
        return np.random.choice(sizes, p=probabilities)

def generate_unique_equation_samples(
    n_samples: int,
    min_nodes: int = 3,
    max_nodes: int = 8,
    randomize_labels: bool = True,
    max_label_value: int = 100,
    min_chain_length: int = 2,
    max_chain_length: int = 6,
    sampling_strategy: str = "uniform_size",
    fixed_label_set: bool = False,
    min_distance: int = 1,
    max_distance: int = None,
    min_start_value: int = 1,
    max_start_value: int = 10,
    min_constant: int = 1,
    max_constant: int = 3,
    operations: List[str] = ['+', '-'],
    shuffle_equations: bool = True,
) -> List[Tuple[str, str]]:
    """
    Generate unique equation reasoning samples.
    
    Args:
        n_samples: Number of samples to generate
        min_nodes: Minimum number of nodes
        max_nodes: Maximum number of nodes
        randomize_labels: Whether to randomize node labels
        max_label_value: Maximum value for node labels
        min_chain_length: Minimum length of individual chains
        max_chain_length: Maximum length of individual chains
        sampling_strategy: How to sample graph sizes. Options:
            - "uniform_size": Each graph size has equal probability (original behavior)
            - "stratified": Equal number of samples per graph size
            - "exponential": Weight exponentially by size (favors larger graphs)
            - "quadratic": Weight quadratically by size
            - "proportional": Weight by estimated number of possible DAGs
        fixed_label_set: If True, randomize labels within {0,1,2,...,max_nodes-1} instead of larger range
        min_distance: Minimum distance (number of edges) between start node and end of its chain
        max_distance: Maximum distance (number of edges) between start node and end of its chain (None = no limit)
        min_start_value: Minimum value for the starting node
        max_start_value: Maximum value for the starting node
        min_constant: Minimum value for edge constants
        max_constant: Maximum value for edge constants
        operations: List of operations to use (default: ["+", "-"])
        shuffle_equations: Whether to shuffle equation order in output
    
    Returns:
        List of (equation_input_string, computed_answer) tuples
    """
    if operations is None:
        operations = ["+", "-"]
    
    seen: Set[str] = set()
    samples = []
    
    def generate_single_sample(n: int) -> Optional[Tuple[str, str]]:
        """Generate a single sample with n nodes. Returns None if sample should be skipped."""
        # Generate independent chains
        g, node_to_chain_end = generate_independent_chains(
            n, min_chain_length, max_chain_length
        )
        
        # Find valid starting nodes that meet distance requirements
        valid_start_nodes = find_valid_start_nodes(g, min_distance, max_distance)
        if not valid_start_nodes:
            return None
        
        # Pick a random starting node from valid ones
        start_node = random.choice(valid_start_nodes)
        
        if randomize_labels:
            # Store original graph info before relabeling
            original_nodes = list(g.nodes)
            original_edges = list(g.edges)
            original_start_node = start_node
            
            if fixed_label_set:
                g_relabeled = randomize_node_labels_fixed_set(g, max_nodes)
            else:
                g_relabeled = randomize_node_labels(g, max_label_value)
            
            # Create mapping from original to new labels
            new_nodes = list(g_relabeled.nodes)
            node_mapping = dict(zip(original_nodes, new_nodes))
            
            # Update start_node with new label
            start_node = node_mapping[original_start_node]
            g = g_relabeled
        
        # Generate random constants for edges
        edge_constants = generate_edge_constants(g, min_constant, max_constant, operations)
        
        # Generate random start value
        start_value = random.randint(min_start_value, max_start_value)
        
        # Create input string with equations
        equation_input = chains_to_equations_string(
            g, start_node, start_value, edge_constants, shuffle_equations
        )
        
        if equation_input in seen:
            return None
        
        # Compute the answer
        answer = compute_chain_value(g, start_node, start_value, edge_constants)
        
        return equation_input, str(answer)
    
    if sampling_strategy == "stratified":
        # Stratified sampling: equal number of samples per graph size
        sizes = list(range(min_nodes, max_nodes + 1))
        samples_per_size = n_samples // len(sizes)
        remainder = n_samples % len(sizes)
        
        print(f"Stratified sampling: {samples_per_size} samples per size, {remainder} extra")
        
        for i, size in enumerate(sizes):
            # Give extra samples to first 'remainder' sizes
            target_samples = samples_per_size + (1 if i < remainder else 0)
            size_samples = 0
            
            while size_samples < target_samples:
                result = generate_single_sample(size)
                if result is None:
                    continue
                
                equation_input, answer = result
                seen.add(equation_input)
                samples.append((equation_input, answer))
                size_samples += 1
            
            print(f"  Generated {size_samples} samples for size {size}")
    
    else:
        # Weighted sampling strategies
        weights = _calculate_size_weights(min_nodes, max_nodes, sampling_strategy)
        
        print(f"Using {sampling_strategy} sampling strategy")
        if sampling_strategy != "uniform_size":
            print("Size weights:", {size: f"{weight:.2f}" for size, weight in weights.items()})
        
        while len(samples) < n_samples:
            n = _sample_graph_size(min_nodes, max_nodes, sampling_strategy, weights)
            
            result = generate_single_sample(n)
            if result is None:
                continue
            
            equation_input, answer = result
            seen.add(equation_input)
            samples.append((equation_input, answer))
    
    # Print final size distribution
    size_counts = {}
    for equation_input, _ in samples:
        # Count equations (each x_N = represents a node)
        import re
        node_refs = re.findall(r'x_(\d+)', equation_input)
        if node_refs:
            unique_nodes = len(set(node_refs))
            size_counts[unique_nodes] = size_counts.get(unique_nodes, 0) + 1
    
    print(f"\nFinal size distribution ({len(samples)} samples):")
    for size in sorted(size_counts.keys()):
        percentage = size_counts[size] / len(samples) * 100
        print(f"  {size} nodes: {size_counts[size]} samples ({percentage:.1f}%)")
    
    print(f"\n📐 Equation reasoning task: Solve the system of equations to find the target variable")
    print(f"   Chain lengths: {min_chain_length}-{max_chain_length} nodes per chain")
    print(f"   Distance from start to end: {min_distance}-{max_distance if max_distance else '∞'} edges")
    print(f"   Operations: {operations}")
    print(f"   Constants range: {min_constant}-{max_constant}")
    print(f"   Start values range: {min_start_value}-{max_start_value}")
    
    return samples

# Convenience function to create datasets for Hugging Face format
def create_equation_dataset_dict(n_samples: int, **kwargs) -> dict:
    """Create a dataset dictionary compatible with Hugging Face datasets."""
    samples = generate_unique_equation_samples(n_samples, **kwargs)
    
    return {
        "input": [sample[0] for sample in samples],
        "target": [sample[1] for sample in samples]
    }

# Keep old function names for backwards compatibility
def generate_unique_chain_samples_simplified(*args, **kwargs):
    """Deprecated: Use generate_unique_equation_samples instead."""
    return generate_unique_equation_samples(*args, **kwargs)

def create_dag_dataset_dict(*args, **kwargs):
    """Deprecated: Use create_equation_dataset_dict instead."""
    return create_equation_dataset_dict(*args, **kwargs)

if __name__ == "__main__":
    # Test the equation dataset generation
    print("Testing equation dataset generation...")
    samples = generate_unique_equation_samples(
        n_samples=5,
        min_nodes=20,
        max_nodes=20,
        randomize_labels=True,
        max_label_value=20,
        min_distance=1,
        max_distance=9,
        min_start_value=1,
        max_start_value=10,
        min_constant=1,
        max_constant=1,
        operations=["+", "-"],
        shuffle_equations=True
    )
    
    print("\nSample outputs:")
    for i, (input_str, target) in enumerate(samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"Input: {input_str}")
        print(f"Target: {target}")
