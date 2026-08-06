#!/usr/bin/env python3
"""
Script to compare accuracy from two completion sample files on the same plot.

This script:
1. Parses two completion sample files (with Input, Target, Completion format)
2. Extracts answers from completions using regex
3. Calculates accuracy across a sliding window for both files
4. Plots both accuracy curves on the same graph for comparison
5. Provides summary statistics for both

Usage:
    python analyze_sliding_accuracy_compare.py <file1> <file2> [--window-size 100] [--output plot.png]
    
Example:
    python analyze_sliding_accuracy_compare.py file1.txt file2.txt --label1 "Run A" --label2 "Run B"
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

# Try to import plotting libraries, handle gracefully if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError as e:
    PLOTTING_AVAILABLE = False
    print(f"Warning: Plotting libraries not available ({e})")
    print("Install with: pip install matplotlib numpy")
    print("Running in statistics-only mode...")
    
    # Create dummy numpy for basic operations
    class DummyNumpy:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod  
        def std(x): 
            if not x: return 0
            mean_val = sum(x) / len(x)
            return (sum((xi - mean_val) ** 2 for xi in x) / len(x)) ** 0.5
        @staticmethod
        def min(x): return min(x) if x else 0
        @staticmethod
        def max(x): return max(x) if x else 0
        @staticmethod
        def median(x): 
            if not x: return 0
            sorted_x = sorted(x)
            n = len(sorted_x)
            return sorted_x[n//2] if n % 2 else (sorted_x[n//2-1] + sorted_x[n//2]) / 2
        @staticmethod
        def polyfit(x, y, deg): 
            if deg != 1 or len(x) != len(y) or len(x) < 2:
                return [0, 0]
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            return [slope, intercept]
        @staticmethod
        def histogram(data, bins):
            if not data: return [0] * (len(bins) - 1), bins
            hist = [0] * (len(bins) - 1)
            for value in data:
                for i in range(len(bins) - 1):
                    if bins[i] <= value < bins[i + 1]:
                        hist[i] += 1
                        break
                    elif i == len(bins) - 2 and value >= bins[i + 1]:
                        hist[i] += 1
                        break
            return hist, bins
        @staticmethod
        def linspace(start, stop, num):
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        @staticmethod
        def array(x):
            return list(x)
    
    np = DummyNumpy()

# Maximum finetuning steps for X-axis
MAX_FINETUNING_STEPS = 800

# Default smoothing window size
DEFAULT_SMOOTHING_WINDOW = 20


def smooth_data(data: List[float], window_size: int = DEFAULT_SMOOTHING_WINDOW) -> List[float]:
    """
    Smooth data using a moving average.
    
    Args:
        data: List of values to smooth
        window_size: Size of the moving average window
        
    Returns:
        Smoothed data list
    """
    if not PLOTTING_AVAILABLE:
        if len(data) < window_size:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            result.append(sum(data[start:end]) / (end - start))
        return result
    
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    padded = np.pad(data, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return list(smoothed)


def setup_plot_style():
    """Configure matplotlib for clean, professional plots."""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.figsize': (12, 7),
        'figure.dpi': 100,
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.labelsize': 18,
        'axes.titlesize': 22,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#CCCCCC',
        'legend.fontsize': 16,
        'legend.loc': 'lower right',
        'lines.linewidth': 2.5,
        'lines.antialiased': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
        'font.size': 16,
    })


# Color palette for comparison
COLORS = {
    'file1': '#2E86AB',      # Steel blue
    'file2': '#C73E1D',      # Red/orange
}


try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


def parse_equation_graph(equation_input: str):
    """
    Parse an equation string and build the dependency graph.

    Handles both old chain format (``x_v = x_u + c``) and new polynomial
    format (``x_v = 2 * x_u^2 * x_w + 3 * x_w - 7``).

    When a variable is defined both as a root constant *and* via a dependency
    equation, the root assignment takes precedence.

    Returns:
        (graph, root_nodes, target_node) or (None, set(), -1) on failure.
    """
    if not NX_AVAILABLE:
        return None, set(), -1

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


def calculate_num_ancestors(input_str: str) -> int:
    """
    Count the number of ancestor variables of the target (including the target).

    This is the primary hardness metric: how many variables must be resolved
    to compute the final answer.

    Returns:
        Number of ancestor nodes (including target), or -1 on failure.
    """
    g, root_nodes, target_node = parse_equation_graph(input_str)
    if g is None:
        return -1
    try:
        return len(nx.ancestors(g, target_node) | {target_node})
    except nx.NetworkXError:
        return -1


def filter_samples_by_num_ancestors(
    samples: List[Dict[str, str]], allowed_counts: List[int],
) -> List[Dict[str, str]]:
    """
    Keep only samples whose ancestor-subgraph size is in *allowed_counts*.
    """
    filtered = []
    for sample in samples:
        n = calculate_num_ancestors(sample['input'])
        if n in allowed_counts:
            filtered.append(sample)
    return filtered


def calculate_ancestor_depth(input_str: str) -> int:
    """
    Compute the depth of the ancestor subgraph of the target = the length of
    the longest path that ends at the target node.

    Depth is measured in number of edges, so a target that is itself a root
    constant has depth 0.

    Returns:
        Depth of the longest path ending at the target, or -1 on failure.
    """
    g, _root_nodes, target_node = parse_equation_graph(input_str)
    if g is None:
        return -1
    try:
        ancestor_nodes = nx.ancestors(g, target_node) | {target_node}
        sub = g.subgraph(ancestor_nodes)
        # In the induced ancestor subgraph the target is the unique sink, so
        # the longest path in the DAG necessarily ends at the target.
        return nx.dag_longest_path_length(sub)
    except (nx.NetworkXError, nx.NetworkXUnfeasible):
        return -1


def filter_samples_by_ancestor_depth(
    samples: List[Dict[str, str]], allowed_depths: List[int],
) -> List[Dict[str, str]]:
    """
    Keep only samples whose ancestor-subgraph depth is in *allowed_depths*.
    """
    filtered = []
    for sample in samples:
        d = calculate_ancestor_depth(sample['input'])
        if d in allowed_depths:
            filtered.append(sample)
    return filtered


def parse_completion_samples_file(filepath: str) -> List[Dict[str, str]]:
    """
    Parse completion samples file with Input, Target, Completion format.
    """
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = content.split('==============')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        input_match = re.search(r'Input: (.*?)(?=Target:|$)', section, re.DOTALL)
        target_match = re.search(r'Target: (.*?)(?=Completion:|$)', section, re.DOTALL)
        completion_match = re.search(r'Completion: (.*)', section, re.DOTALL)
        
        if input_match and target_match and completion_match:
            sample = {
                'input': input_match.group(1).strip(),
                'target': target_match.group(1).strip(),
                'completion': completion_match.group(1).strip()
            }
            samples.append(sample)
    
    return samples


def extract_answer_from_completion(completion: str) -> str:
    """
    Extract the answer from a completion using regex.
    """
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return re.sub(r'\s+', '', answer)
    return ""


def check_format(completion: str) -> bool:
    """
    Check if the completion has the correct format.
    
    The expected format is:
    <think>...</think>
    <answer>...</answer>
    """
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\n<answer>([\s\S]*?)</answer>$"
    match = re.search(regex, completion.strip(), re.DOTALL)
    return match is not None and len(match.groups()) == 2


def calculate_sliding_accuracy(samples: List[Dict[str, str]], window_size: int = 100, strict_format: bool = False) -> Tuple[List[float], List[int], List[float]]:
    """
    Calculate accuracy and answer rate across a sliding window.
    
    Args:
        samples: List of sample dictionaries
        window_size: Size of the sliding window
        strict_format: If True, only count answers as correct if format is also valid
    """
    if len(samples) < window_size:
        print(f"Warning: Only {len(samples)} samples available, less than window size {window_size}")
        window_size = len(samples)
    
    accuracies = []
    answer_rates = []
    window_centers = []
    
    for i in range(len(samples) - window_size + 1):
        window_samples = samples[i:i + window_size]
        correct = 0
        has_answer = 0
        
        for sample in window_samples:
            completion = sample['completion']
            predicted_answer = extract_answer_from_completion(completion)
            target_answer = re.sub(r'\s+', '', str(sample['target']))
            
            if predicted_answer:
                has_answer += 1
            
            # Check if answer is correct
            answer_correct = predicted_answer == target_answer
            
            # If strict_format is enabled, also require valid format
            if strict_format:
                if answer_correct and check_format(completion):
                    correct += 1
            else:
                if answer_correct:
                    correct += 1
        
        accuracy = correct / window_size
        answer_rate = has_answer / window_size
        accuracies.append(accuracy)
        answer_rates.append(answer_rate)
        window_centers.append(i + window_size // 2)
    
    return accuracies, window_centers, answer_rates


def scale_to_finetuning_steps(window_centers: List[int], total_samples: int, max_steps: int = MAX_FINETUNING_STEPS) -> List[float]:
    """
    Scale window centers to finetuning steps (0 to max_steps).
    """
    if total_samples <= 1:
        return window_centers
    
    return [pos * max_steps / total_samples for pos in window_centers]


def plot_comparison(
    accuracies1: List[float], window_centers1: List[int], total_samples1: int,
    accuracies2: List[float], window_centers2: List[int], total_samples2: int,
    raw_accuracies1: List[float], raw_window_centers1: List[int],
    raw_accuracies2: List[float], raw_window_centers2: List[int],
    label1: str, label2: str,
    window_size: int, output_path: str = None,
    x_label: str = None, y_label: str = None, title: str = None,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    raw_smoothing_window: int = 1,
    show_legend: bool = True):
    """
    Plot two accuracy curves on the same graph for comparison.
    Shows both raw (semi-transparent) and smoothed (solid) lines.
    
    Args:
        accuracies1/2: Accuracy data for solid trend line
        raw_accuracies1/2: Accuracy data for semi-transparent noisy line
        smoothing_window: Smoothing for the solid trend line
        raw_smoothing_window: Smoothing for the semi-transparent noisy line (1 = no smoothing)
    """
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting libraries not available. Install matplotlib and numpy to enable plotting.")
        return
    
    setup_plot_style()
    
    # Scale x-axis to finetuning steps (0-800)
    scaled_centers1 = scale_to_finetuning_steps(window_centers1, total_samples1)
    scaled_centers2 = scale_to_finetuning_steps(window_centers2, total_samples2)
    raw_scaled_centers1 = scale_to_finetuning_steps(raw_window_centers1, total_samples1)
    raw_scaled_centers2 = scale_to_finetuning_steps(raw_window_centers2, total_samples2)
    
    # Apply smoothing to both raw and trend data
    smoothed_raw1 = smooth_data(raw_accuracies1, raw_smoothing_window) if raw_smoothing_window > 1 else raw_accuracies1
    smoothed_raw2 = smooth_data(raw_accuracies2, raw_smoothing_window) if raw_smoothing_window > 1 else raw_accuracies2
    smoothed_accuracies1 = smooth_data(accuracies1, smoothing_window)
    smoothed_accuracies2 = smooth_data(accuracies2, smoothing_window)
    
    # Convert to percentages (multiply by 100)
    smoothed_raw1 = [v * 100 for v in smoothed_raw1]
    smoothed_raw2 = [v * 100 for v in smoothed_raw2]
    smoothed_accuracies1 = [v * 100 for v in smoothed_accuracies1]
    smoothed_accuracies2 = [v * 100 for v in smoothed_accuracies2]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot first file's raw accuracy (semi-transparent)
    ax.plot(raw_scaled_centers1, smoothed_raw1, 
            linewidth=1.5, 
            color=COLORS['file1'],
            alpha=0.25,
            zorder=2)
    
    # Plot first file's smoothed accuracy (solid)
    ax.plot(scaled_centers1, smoothed_accuracies1, 
            linewidth=2.5, 
            color=COLORS['file1'],
            alpha=1.0,
            label=label1,
            zorder=3)
    
    # Plot second file's raw accuracy (semi-transparent)
    ax.plot(raw_scaled_centers2, smoothed_raw2, 
            linewidth=1.5, 
            color=COLORS['file2'],
            alpha=0.25,
            zorder=2)
    
    # Plot second file's smoothed accuracy (solid)
    ax.plot(scaled_centers2, smoothed_accuracies2, 
            linewidth=2.5, 
            color=COLORS['file2'],
            alpha=1.0,
            label=label2,
            zorder=3)
    
    # Configure axes with custom or default labels
    ax.set_xlabel(x_label or 'Finetuning Steps', fontsize=20, fontweight='medium')
    ax.set_ylabel(y_label or 'Accuracy', fontsize=20, fontweight='medium')
    ax.set_title(title or f'Accuracy Comparison (window={window_size})', fontsize=24, fontweight='bold', pad=18)
    
    # Set X-axis limits to 0-800
    ax.set_xlim(0, MAX_FINETUNING_STEPS)
    ax.set_ylim(0, 105)
    
    # Configure X-axis ticks
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    
    # Configure Y-axis ticks (percentages)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    
    # Add minor grid
    ax.grid(True, which='major', linewidth=0.8, alpha=0.7)
    ax.grid(True, which='minor', linewidth=0.4, alpha=0.4)
    
    # Legend
    if show_legend:
        ax.legend(loc='lower right', fontsize=24, framealpha=0.95)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_comparison_statistics(
    samples1: List[Dict[str, str]], accuracies1: List[float], label1: str,
    samples2: List[Dict[str, str]], accuracies2: List[float], label2: str,
    window_size: int):
    """
    Print comparison statistics for both files.
    """
    print("="*80)
    print(f"ACCURACY COMPARISON")
    print("="*80)
    print(f"Window Size: {window_size}")
    print(f"X-axis scaled to: 0-{MAX_FINETUNING_STEPS} finetuning steps")
    print()
    
    print(f"{'Metric':<25} {label1:<20} {label2:<20}")
    print("-" * 65)
    print(f"{'Total Samples':<25} {len(samples1):<20} {len(samples2):<20}")
    print(f"{'Number of Windows':<25} {len(accuracies1):<20} {len(accuracies2):<20}")
    print(f"{'Mean Accuracy':<25} {np.mean(accuracies1):<20.4f} {np.mean(accuracies2):<20.4f}")
    print(f"{'Std Accuracy':<25} {np.std(accuracies1):<20.4f} {np.std(accuracies2):<20.4f}")
    print(f"{'Min Accuracy':<25} {np.min(accuracies1):<20.4f} {np.min(accuracies2):<20.4f}")
    print(f"{'Max Accuracy':<25} {np.max(accuracies1):<20.4f} {np.max(accuracies2):<20.4f}")
    print(f"{'Median Accuracy':<25} {np.median(accuracies1):<20.4f} {np.median(accuracies2):<20.4f}")
    print(f"{'Final Window Accuracy':<25} {accuracies1[-1]:<20.4f} {accuracies2[-1]:<20.4f}")
    
    # Calculate trends
    if len(accuracies1) > 1 and len(accuracies2) > 1:
        slope1 = np.polyfit(list(range(len(accuracies1))), accuracies1, 1)[0]
        slope2 = np.polyfit(list(range(len(accuracies2))), accuracies2, 1)[0]
        print(f"{'Trend (slope)':<25} {slope1:<20.6f} {slope2:<20.6f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare accuracy from two completion sample files")
    parser.add_argument("file1", help="Path to first completion samples file")
    parser.add_argument("file2", help="Path to second completion samples file")
    parser.add_argument("--window-size", "-w", type=int, default=300, 
                       help="Size of sliding window for the solid trend line (default: 300)")
    parser.add_argument("--raw-window-size", type=int, default=None,
                       help="Size of sliding window for the noisy line (default: same as --window-size)")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output file path for plot (optional)")
    parser.add_argument("--no-plot", action="store_true", 
                       help="Skip plotting and only show statistics")
    parser.add_argument("--max-steps", type=int, default=800,
                       help="Maximum finetuning steps for X-axis (default: 800)")
    parser.add_argument("--x-label", type=str, default=None,
                       help="Custom X-axis label (default: 'Finetuning Steps')")
    parser.add_argument("--y-label", type=str, default=None,
                       help="Custom Y-axis label (default: 'Accuracy')")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom graph title")
    parser.add_argument("--smoothing", "-s", type=int, default=300,
                       help="Smoothing window size for the solid trend line (default: 300, use 1 for no smoothing)")
    parser.add_argument("--raw-smoothing", type=int, default=1,
                       help="Smoothing window size for the semi-transparent noisy line (default: 1 = no smoothing)")
    parser.add_argument("--label1", type=str, default=None,
                       help="Label for first file in legend (default: filename)")
    parser.add_argument("--label2", type=str, default=None,
                       help="Label for second file in legend (default: filename)")
    parser.add_argument("--no-legend", action="store_true",
                       help="Hide the plot legend")
    parser.add_argument("--strict-format", action="store_true",
                       help="Only count answers as correct if format is also valid (<think>...</think>\\n<answer>...</answer>)")
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--num-ancestors", type=str, default=None,
                       help="Comma-separated list of ancestor-subgraph sizes to include (e.g., '3,5,8'). "
                            "Counts all variables needed to compute the target. Requires networkx. "
                            "Mutually exclusive with --ancestor-depth.")
    filter_group.add_argument("--path-lengths", type=str, default=None,
                       help="Deprecated alias for --num-ancestors (kept for backward compatibility).")
    filter_group.add_argument("--ancestor-depth", type=str, default=None,
                       help="Comma-separated list of ancestor-subgraph depths to include (e.g., '2,3,4'). "
                            "Depth = number of edges in the longest path that ends at the target. "
                            "Requires networkx. Mutually exclusive with --num-ancestors.")
    
    args = parser.parse_args()
    
    # Update global max steps if provided
    global MAX_FINETUNING_STEPS
    MAX_FINETUNING_STEPS = args.max_steps
    
    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        return 1
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        return 1
    
    # Set default labels from filenames
    label1 = args.label1 or Path(args.file1).stem
    label2 = args.label2 or Path(args.file2).stem
    
    try:
        # Parse ancestor-count filter (--num-ancestors or deprecated --path-lengths)
        ancestor_filter = None
        depth_filter = None
        raw_filter = args.num_ancestors or args.path_lengths
        if raw_filter:
            if not NX_AVAILABLE:
                print("Error: --num-ancestors requires networkx.  pip install networkx")
                return 1
            try:
                ancestor_filter = [int(x.strip()) for x in raw_filter.split(',')]
                print(f"Filtering samples to ancestor counts: {ancestor_filter}")
            except ValueError:
                print(f"Error: Invalid format: {raw_filter}")
                print("Expected comma-separated integers, e.g., '3,5,8'")
                return 1

        # Parse ancestor-depth filter (--ancestor-depth)
        if args.ancestor_depth:
            if not NX_AVAILABLE:
                print("Error: --ancestor-depth requires networkx.  pip install networkx")
                return 1
            try:
                depth_filter = [int(x.strip()) for x in args.ancestor_depth.split(',')]
                print(f"Filtering samples to ancestor depths: {depth_filter}")
            except ValueError:
                print(f"Error: Invalid format: {args.ancestor_depth}")
                print("Expected comma-separated integers, e.g., '2,3,4'")
                return 1
        
        # Parse first file
        print(f"Parsing file 1: {args.file1}")
        samples1 = parse_completion_samples_file(args.file1)
        print(f"  Parsed {len(samples1)} samples")
        
        # Parse second file
        print(f"Parsing file 2: {args.file2}")
        samples2 = parse_completion_samples_file(args.file2)
        print(f"  Parsed {len(samples2)} samples")
        
        # Apply ancestor-count filtering if specified
        if ancestor_filter:
            samples1 = filter_samples_by_num_ancestors(samples1, ancestor_filter)
            samples2 = filter_samples_by_num_ancestors(samples2, ancestor_filter)
            print(f"  After filtering - File 1: {len(samples1)} samples, File 2: {len(samples2)} samples")
        elif depth_filter:
            samples1 = filter_samples_by_ancestor_depth(samples1, depth_filter)
            samples2 = filter_samples_by_ancestor_depth(samples2, depth_filter)
            print(f"  After filtering - File 1: {len(samples1)} samples, File 2: {len(samples2)} samples")
        
        if len(samples1) == 0:
            print("Error: No samples found in file 1 (after filtering)")
            return 1
        if len(samples2) == 0:
            print("Error: No samples found in file 2 (after filtering)")
            return 1
        
        # Set raw window size (default to same as main window size)
        raw_window_size = args.raw_window_size if args.raw_window_size is not None else args.window_size
        
        # Calculate sliding window accuracy for both files (for trend line)
        print(f"Calculating sliding window accuracy (trend window: {args.window_size}, raw window: {raw_window_size})")
        if args.strict_format:
            print("  Using strict format checking (answer + format must be correct)")
        accuracies1, window_centers1, _ = calculate_sliding_accuracy(samples1, args.window_size, strict_format=args.strict_format)
        accuracies2, window_centers2, _ = calculate_sliding_accuracy(samples2, args.window_size, strict_format=args.strict_format)
        
        # Calculate sliding window accuracy for raw/noisy line (possibly different window size)
        if raw_window_size != args.window_size:
            raw_accuracies1, raw_window_centers1, _ = calculate_sliding_accuracy(samples1, raw_window_size, strict_format=args.strict_format)
            raw_accuracies2, raw_window_centers2, _ = calculate_sliding_accuracy(samples2, raw_window_size, strict_format=args.strict_format)
        else:
            raw_accuracies1, raw_window_centers1 = accuracies1, window_centers1
            raw_accuracies2, raw_window_centers2 = accuracies2, window_centers2
        
        if len(accuracies1) == 0 or len(accuracies2) == 0:
            print("Error: Could not calculate accuracy windows for one or both files")
            return 1
        
        # Print comparison statistics
        print_comparison_statistics(samples1, accuracies1, label1, samples2, accuracies2, label2, args.window_size)
        
        # Plot comparison
        if not args.no_plot and PLOTTING_AVAILABLE:
            plot_comparison(
                accuracies1, window_centers1, len(samples1),
                accuracies2, window_centers2, len(samples2),
                raw_accuracies1, raw_window_centers1,
                raw_accuracies2, raw_window_centers2,
                label1, label2,
                args.window_size, args.output,
                x_label=args.x_label, y_label=args.y_label, title=args.title,
                smoothing_window=args.smoothing,
                raw_smoothing_window=args.raw_smoothing,
                show_legend=not args.no_legend
            )
        elif not args.no_plot and not PLOTTING_AVAILABLE:
            print("Skipping plot generation (plotting libraries not available)")
            print("Use --no-plot flag to suppress this message")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

