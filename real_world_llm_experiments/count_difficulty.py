#!/usr/bin/env python3
"""
Count samples per difficulty metric in completion sample files.

Metrics:
  ancestors       – size of the target's ancestor subgraph (incl. target)
  equations       – number of equations in the prompt
  ancestor_depth  – longest-path depth (edges) of the target's ancestor subgraph

Streams the file to avoid loading multi-GB files entirely into memory.

Usage:
    python count_difficulty.py <completion_file> [<completion_file2> ...]
    python count_difficulty.py completion_samples/completion_samples_dag_*.txt
    python count_difficulty.py --metric ancestor_depth completion_samples/*.txt
"""

import argparse
import re
import sys
from collections import Counter

try:
    import networkx as nx
except ImportError:
    sys.exit("networkx is required: pip install networkx")


def parse_equation_graph(equation_input: str):
    target_match = re.search(r'Find x_(\d+)', equation_input)
    if not target_match:
        return None, set(), -1
    target_node = int(target_match.group(1))

    g = nx.DiGraph()
    root_nodes = set()

    parts = equation_input.split('.')
    for part in parts:
        part = part.strip()
        if not part or part.startswith('Find'):
            continue
        eq_match = re.match(r'x_(\d+)\s*=\s*(.*)', part)
        if not eq_match:
            continue
        lhs = int(eq_match.group(1))
        rhs = eq_match.group(2).strip()
        if not re.search(r'x_\d+', rhs):
            root_nodes.add(lhs)
            g.add_node(lhs)

    for part in parts:
        part = part.strip()
        if not part or part.startswith('Find'):
            continue
        eq_match = re.match(r'x_(\d+)\s*=\s*(.*)', part)
        if not eq_match:
            continue
        lhs = int(eq_match.group(1))
        rhs = eq_match.group(2).strip()
        rhs_vars = set(int(v) for v in re.findall(r'x_(\d+)', rhs))
        if rhs_vars and lhs not in root_nodes:
            g.add_node(lhs)
            for rv in rhs_vars:
                g.add_node(rv)
                g.add_edge(rv, lhs)

    if not root_nodes:
        return None, set(), -1
    return g, root_nodes, target_node


def num_ancestors(input_str: str) -> int:
    g, root_nodes, target_node = parse_equation_graph(input_str)
    if g is None:
        return -1
    try:
        return len(nx.ancestors(g, target_node) | {target_node})
    except nx.NetworkXError:
        return -1


def ancestor_depth(input_str: str) -> int:
    """
    Longest-path depth (edges) of the target's ancestor subgraph.

    A root constant target has depth 0. Matches the bucketed-generation
    convention where generation param ``ancestor_depth`` (node count) equals
    this edge depth + 1.
    """
    g, _root_nodes, target_node = parse_equation_graph(input_str)
    if g is None:
        return -1
    try:
        ancestor_nodes = nx.ancestors(g, target_node) | {target_node}
        sub = g.subgraph(ancestor_nodes)
        return nx.dag_longest_path_length(sub)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return -1


def num_equations(input_str: str) -> int:
    parts = input_str.split('.')
    count = 0
    for part in parts:
        part = part.strip()
        if not part or part.startswith('Find'):
            continue
        if re.match(r'x_\d+\s*=', part):
            count += 1
    return count


def stream_samples(filepath: str):
    """Yield (input_str, target_str) tuples by streaming the file."""
    current_input = None
    current_target = None
    in_completion = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped == '==============' or stripped == '':
                if in_completion and current_input is not None:
                    yield current_input, current_target
                    current_input = None
                    current_target = None
                    in_completion = False
                continue
            if stripped.startswith('Input: '):
                if in_completion and current_input is not None:
                    yield current_input, current_target
                current_input = stripped[len('Input: '):]
                current_target = None
                in_completion = False
            elif stripped.startswith('Target: '):
                current_target = stripped[len('Target: '):]
            elif stripped.startswith('Completion: '):
                in_completion = True

    if current_input is not None:
        yield current_input, current_target


def _want(metric: str, name: str) -> bool:
    if metric == "all":
        return True
    if metric == "both":
        return name in ("ancestors", "equations")
    return metric == name


def main():
    parser = argparse.ArgumentParser(description="Count samples per difficulty in completion files")
    parser.add_argument("files", nargs="+", help="Completion sample file(s)")
    parser.add_argument(
        "--metric",
        choices=["ancestors", "equations", "ancestor_depth", "both", "all"],
        default="both",
        help="Difficulty metric to count by (default: both = ancestors+equations; "
             "all = every metric)",
    )
    args = parser.parse_args()

    for filepath in args.files:
        print(f"\n{'='*60}")
        print(f"File: {filepath}")
        print(f"{'='*60}")

        ancestor_counts = Counter()
        equation_counts = Counter()
        depth_counts = Counter()
        total = 0
        parse_failures = 0

        want_anc = _want(args.metric, "ancestors")
        want_eq = _want(args.metric, "equations")
        want_depth = _want(args.metric, "ancestor_depth")

        for input_str, _ in stream_samples(filepath):
            total += 1
            if want_anc:
                na = num_ancestors(input_str)
                if na == -1:
                    parse_failures += 1
                else:
                    ancestor_counts[na] += 1
            if want_depth:
                nd = ancestor_depth(input_str)
                if nd == -1:
                    # Only count a parse failure once if ancestors already failed
                    if not want_anc:
                        parse_failures += 1
                else:
                    depth_counts[nd] += 1
            if want_eq:
                ne = num_equations(input_str)
                equation_counts[ne] += 1

            if total % 10000 == 0:
                print(f"  ... processed {total} samples", file=sys.stderr)

        print(f"\nTotal samples: {total}")
        if parse_failures:
            print(f"Parse failures: {parse_failures}")

        if want_anc:
            print(f"\n--- Samples by number of ancestors ---")
            print(f"{'Ancestors':>10}  {'Count':>8}  {'%':>6}")
            for k in sorted(ancestor_counts):
                pct = 100.0 * ancestor_counts[k] / total
                print(f"{k:>10}  {ancestor_counts[k]:>8}  {pct:>5.1f}%")

        if want_depth:
            print(f"\n--- Samples by ancestor depth (edges to target) ---")
            print(f"{'Depth':>10}  {'Count':>8}  {'%':>6}")
            for k in sorted(depth_counts):
                pct = 100.0 * depth_counts[k] / total
                print(f"{k:>10}  {depth_counts[k]:>8}  {pct:>5.1f}%")

        if want_eq:
            print(f"\n--- Samples by number of equations ---")
            print(f"{'Equations':>10}  {'Count':>8}  {'%':>6}")
            for k in sorted(equation_counts):
                pct = 100.0 * equation_counts[k] / total
                print(f"{k:>10}  {equation_counts[k]:>8}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
