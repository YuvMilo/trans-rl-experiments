"""
Graph utility functions for DAG operations.
"""
from typing import List, Tuple


def is_valid_topo(vertex_order: List[int], edge_list: List[Tuple[int, int]], n: int) -> bool:
    """
    Check if vertex_order is a valid topological sort of the graph with given edges.
    
    Args:
        vertex_order: List of vertex indices in the order to check
        edge_list: List of directed edges as (source, target) tuples
        n: Total number of vertices expected
    
    Returns:
        True if vertex_order is a valid topological sort, False otherwise
    """
    if len(vertex_order) != n or len(set(vertex_order)) != n:
        return False
    
    pos = {v: i for i, v in enumerate(vertex_order)}
    for i, j in edge_list:              # edge i ➜ j
        if pos[i] >= pos[j]:
            return False
    return True 