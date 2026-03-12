#!/usr/bin/env python3
"""
Evaluation script for equation reasoning model.
Tests accuracy on different path lengths (number of reasoning steps from
constant variable to target variable), including out-of-distribution longer paths.
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dag_dataset_simplified import generate_unique_equation_samples, find_chain_end, calculate_chain_distance
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate equation reasoning model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="runs/qwen-2.5-1.5b-r1-dag-chain",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--samples_per_config",
        type=int,
        default=10,
        help="Number of samples to test per (chain_size, path_length) configuration",
    )
    parser.add_argument(
        "--min_path_length",
        type=int,
        default=1,
        help="Minimum path length to test (number of steps from constant to target)",
    )
    parser.add_argument(
        "--max_path_length",
        type=int,
        default=25,
        help="Maximum path length to test (includes OOD)",
    )
    parser.add_argument(
        "--training_min_path",
        type=int,
        default=8,
        help="Minimum path length seen during training (min_chain_length - 1)",
    )
    parser.add_argument(
        "--training_max_path",
        type=int,
        default=16,
        help="Maximum path length seen during training (max_chain_length - 1)",
    )
    parser.add_argument(
        "--chain_sizes",
        type=str,
        default=None,
        help="Comma-separated list of chain sizes to test (e.g., '10,15,20'). If None, uses path_length + 1",
    )
    parser.add_argument(
        "--training_min_chain_size",
        type=int,
        default=30,
        help="Minimum chain size seen during training",
    )
    parser.add_argument(
        "--training_max_chain_size",
        type=int,
        default=30,
        help="Maximum chain size seen during training",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_label_value",
        type=int,
        default=40,
        help="Maximum value for node labels (same as training)",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save sample outputs to file",
    )
    return parser.parse_args()


def calculate_chain_size(equation_input: str) -> int:
    """
    Calculate the chain size (number of nodes) from an equation string.
    """
    # Count unique x_N references
    node_refs = re.findall(r'x_(\d+)', equation_input)
    if node_refs:
        return len(set(node_refs))
    return -1


def parse_equation_graph(equation_input: str) -> Tuple[nx.DiGraph, int, int]:
    """
    Parse an equation string and build the dependency graph.
    
    Returns:
        Tuple of (graph, constant_node, target_node) or (None, -1, -1) on failure
    """
    # Find the target variable
    target_match = re.search(r'Find x_(\d+)', equation_input)
    if not target_match:
        return None, -1, -1
    target_node = int(target_match.group(1))
    
    # Find the constant variable (the one with just a number, no x_ reference)
    constant_node = None
    constant_pattern = re.compile(r'x_(\d+)\s*=\s*(\d+)(?:\s*\.|$)')
    
    # Also find all dependency equations: x_v = x_u +/- const
    dependency_pattern = re.compile(r'x_(\d+)\s*=\s*x_(\d+)\s*[+\-]\s*\d+')
    
    # Build dependency graph (edges go from dependency to dependent: u -> v means v depends on u)
    g = nx.DiGraph()
    
    for match in constant_pattern.finditer(equation_input):
        node = int(match.group(1))
        constant_node = node
        g.add_node(node)
    
    for match in dependency_pattern.finditer(equation_input):
        v = int(match.group(1))  # dependent variable
        u = int(match.group(2))  # dependency
        g.add_node(u)
        g.add_node(v)
        g.add_edge(u, v)  # u -> v means v = u +/- const
    
    if constant_node is None:
        return None, -1, -1
    
    return g, constant_node, target_node


def calculate_actual_path_length(equation_input: str) -> int:
    """
    Calculate the actual path length from an equation string.
    
    Parses the equations to build a graph and finds the distance
    from the constant variable to the target variable.
    """
    g, constant_node, target_node = parse_equation_graph(equation_input)
    if g is None:
        return -1
    
    # Calculate path length from constant_node to target_node
    try:
        path_length = nx.shortest_path_length(g, constant_node, target_node)
        return path_length
    except nx.NetworkXNoPath:
        return -1


def get_shortest_path_variables(equation_input: str) -> set:
    """
    Get the set of variable indices on the shortest path from constant to target.
    
    Returns:
        Set of variable indices (integers) on the shortest path, or empty set on failure
    """
    g, constant_node, target_node = parse_equation_graph(equation_input)
    if g is None:
        return set()
    
    try:
        path = nx.shortest_path(g, constant_node, target_node)
        return set(path)
    except nx.NetworkXNoPath:
        return set()


def get_all_variables_in_equation(equation_input: str) -> set:
    """
    Get all variable indices mentioned in the equation system.
    
    Returns:
        Set of all variable indices (integers) in the equation
    """
    node_refs = re.findall(r'x_(\d+)', equation_input)
    return set(int(n) for n in node_refs)


def extract_mentioned_variables(completion: str) -> set:
    """
    Extract all variable indices mentioned in the model's completion.
    
    Handles both x_N and x_{N} formats (LaTeX-style).
    
    Returns:
        Set of variable indices (integers) mentioned in the completion
    """
    # Find all x_N and x_{N} patterns in the completion
    # x_\{?(\d+)\}? matches: x_3, x_{3}, x_{12}, etc.
    mentions = re.findall(r'x_\{?(\d+)\}?', completion)
    return set(int(n) for n in mentions)


def create_prompt(tokenizer, equation_input: str) -> str:
    """Create the prompt in the same format as training."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve arithmetic equations systems."
        },
        {
            "role": "user",
            "content": f"Given this equation system: {equation_input}\n\nFind the value of the target variable  Show your reasoning in <think> </think> tags, and provide your final answer in <answer> </answer> tags, for example <answer>12</answer>."
        },
        {
            "role": "assistant",
            "content": "I'll think about the equation system and solve it.\n<think>"
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)


def extract_answer(completion: str) -> str:
    """Extract the answer from the completion."""
    # Add synthetic <think> as it's already part of the prompt
    completion = "<think>" + completion
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Remove whitespace
        answer = re.sub(r'\s+', '', answer)
        return answer
    return None


def check_format(completion: str) -> bool:
    """Check if the completion has the correct format."""
    completion = "<think>" + completion
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\n<answer>([\s\S]*?)</answer>$"
    match = re.search(regex, completion, re.DOTALL)
    return match is not None and len(match.groups()) == 2


def generate_test_samples_by_config(
    chain_sizes: List[int],
    path_lengths: List[int],
    samples_per_config: int,
    max_label_value: int = 40,
    seed: int = 42,
) -> Dict[Tuple[int, int], List[Tuple[str, str, int, int]]]:
    """
    Generate test samples grouped by (chain_size, path_length) configuration.
    
    Args:
        chain_sizes: List of chain sizes to test (number of nodes). If None/empty, uses path_length + 1
        path_lengths: List of path lengths to generate samples for
        samples_per_config: Number of samples per (chain_size, path_length) configuration
        max_label_value: Maximum value for node labels
        seed: Random seed
    
    Returns:
        Dict mapping (chain_size, path_length) -> list of (input, target, chain_size, path_length)
    """
    random.seed(seed)
    np.random.seed(seed)
    samples_by_config = defaultdict(list)
    
    # Build list of valid configurations
    # path_length must be <= chain_size - 1 (need at least path_length + 1 nodes)
    configs = []
    if chain_sizes:
        for cs in chain_sizes:
            for pl in path_lengths:
                if pl <= cs - 1:  # Valid: path_length < chain_size
                    configs.append((cs, pl))
    else:
        # No chain sizes specified, use exact size for each path length
        for pl in path_lengths:
            configs.append((pl + 1, pl))  # chain_size = path_length + 1
    
    if not configs:
        print("Warning: No valid (chain_size, path_length) configurations!")
        return samples_by_config
    
    # Track how many samples we need for each config
    samples_needed = {config: samples_per_config for config in configs}
    
    # Keep generating until we have enough samples for each config
    max_attempts = samples_per_config * len(configs) * 50  # Safety limit
    attempts = 0
    
    total_samples = sum(samples_needed.values())
    pbar = tqdm(total=total_samples, desc="Generating test samples")
    
    while any(count > 0 for count in samples_needed.values()) and attempts < max_attempts:
        attempts += 1
        
        # Pick a random config that still needs samples
        remaining_configs = [cfg for cfg, count in samples_needed.items() if count > 0]
        if not remaining_configs:
            break
        target_chain_size, target_path_length = random.choice(remaining_configs)
        
        # Generate a single sample with exact chain size
        samples = generate_unique_equation_samples(
            n_samples=1,
            min_nodes=target_chain_size,
            max_nodes=target_chain_size,
            min_chain_length=target_chain_size,
            max_chain_length=target_chain_size,
            min_distance=target_path_length,  # At least this many steps
            max_label_value=max_label_value,
            randomize_labels=True,
            fixed_label_set=False,
            min_start_value=1,
            max_start_value=10,
            min_constant=1,
            max_constant=3,
            operations=['+', '-'],
            shuffle_equations=True,
        )
        
        if not samples:
            continue
        
        input_str, target = samples[0]
        
        # Calculate the ACTUAL chain size and path length
        actual_chain_size = calculate_chain_size(input_str)
        actual_path_length = calculate_actual_path_length(input_str)
        
        if actual_chain_size < 0 or actual_path_length < 0:
            continue  # Failed to parse
        
        # Check if this matches a config that still needs samples
        actual_config = (actual_chain_size, actual_path_length)
        if actual_config in samples_needed and samples_needed[actual_config] > 0:
            samples_by_config[actual_config].append((input_str, target, actual_chain_size, actual_path_length))
            samples_needed[actual_config] -= 1
            pbar.update(1)
    
    pbar.close()
    
    # Report any configs that couldn't be fully generated
    for (cs, pl), remaining in samples_needed.items():
        if remaining > 0:
            generated = samples_per_config - remaining
            print(f"  Warning: Could only generate {generated}/{samples_per_config} samples for chain_size={cs}, path_length={pl}")
    
    return samples_by_config


def evaluate_model(
    model,
    tokenizer,
    samples: List[Tuple[str, str, int]],
    batch_size: int = 16,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate the model on a list of samples.
    
    Returns:
        Dict with accuracy, format_accuracy, and sample results
    """
    correct = 0
    format_correct = 0
    total = len(samples)
    results = []
    
    # Process in batches
    for i in tqdm(range(0, total, batch_size), desc="Evaluating", leave=False):
        batch = samples[i:i + batch_size]
        
        # Create prompts
        prompts = [create_prompt(tokenizer, s[0]) for s in batch]
        targets = [s[1] for s in batch]
        steps_list = [s[2] for s in batch]
        inputs_list = [s[0] for s in batch]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)
        
        # Generate (deterministic greedy decoding)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic results
                num_beams=1,  # No beam search
                temperature=None,  # Ignored when do_sample=False, but explicit
                top_p=None,  # Disabled
                top_k=None,  # Disabled
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        # Decode completions (only the generated part)
        for j, (output, prompt, target, steps, input_str) in enumerate(
            zip(outputs, prompts, targets, steps_list, inputs_list)
        ):
            # Decode the full output and extract completion via text matching
            # This is more robust than token-based extraction which can misalign
            full_output = tokenizer.decode(output, skip_special_tokens=True)
            
            # The prompt ends with "<think>" - find where the completion starts
            # We look for the assistant's prefilled content ending
            prompt_marker = "I'll think about the equation system and solve it.\n<think>"
            marker_pos = full_output.find(prompt_marker)
            if marker_pos != -1:
                completion = full_output[marker_pos + len(prompt_marker):]
            else:
                # Fallback: try to find just "<think>" near the expected position
                # This handles cases where the exact text might differ slightly
                think_pos = full_output.rfind("<think>")
                if think_pos != -1:
                    completion = full_output[think_pos + len("<think>"):]
                else:
                    # Last resort: use token-based extraction
                    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                    completion = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            
            # Extract answer
            predicted = extract_answer(completion)
            is_correct = predicted == target if predicted else False
            is_format_correct = check_format(completion)
            
            if is_correct:
                correct += 1
            if is_format_correct:
                format_correct += 1
            
            # Verify path length and chain size from the input
            actual_pl = calculate_actual_path_length(input_str)
            actual_cs = calculate_chain_size(input_str)
            
            # Check for off-path variable mentions
            shortest_path_vars = get_shortest_path_variables(input_str)
            all_equation_vars = get_all_variables_in_equation(input_str)
            mentioned_vars = extract_mentioned_variables(completion)
            
            # Off-path variables are those mentioned in the completion that:
            # 1. Are in the equation system (valid variables)
            # 2. Are NOT on the shortest path
            off_path_vars_in_equation = all_equation_vars - shortest_path_vars
            mentioned_off_path_vars = mentioned_vars & off_path_vars_in_equation
            mentions_off_path = len(mentioned_off_path_vars) > 0
            
            results.append({
                "input": input_str,
                "target": target,
                "predicted": predicted,
                "completion": completion,
                "is_correct": is_correct,
                "is_format_correct": is_format_correct,
                "path_length": actual_pl,  # Actual number of steps from constant to target
                "chain_size": actual_cs,  # Actual number of nodes in the chain
                "mentions_off_path": mentions_off_path,  # True if model mentions variables not on shortest path
                "off_path_vars_mentioned": sorted(list(mentioned_off_path_vars)),  # Which off-path vars were mentioned
                "shortest_path_vars": sorted(list(shortest_path_vars)),  # Variables on the shortest path
                "mentioned_vars": sorted(list(mentioned_vars)),  # All variables mentioned in completion
            })
    
    # Calculate efficient reasoning statistics (among correct answers only)
    # Efficient = correct AND doesn't mention off-path variables
    efficient_correct_count = sum(1 for r in results if r["is_correct"] and not r["mentions_off_path"])
    
    return {
        "accuracy": correct / total if total > 0 else 0,
        "format_accuracy": format_correct / total if total > 0 else 0,
        "efficient_rate": efficient_correct_count / correct if correct > 0 else 0,
        "correct": correct,
        "format_correct": format_correct,
        "efficient_correct_count": efficient_correct_count,
        "total": total,
        "results": results,
    }


def is_in_distribution(chain_size: int, path_length: int, args) -> bool:
    """Check if a (chain_size, path_length) configuration is in-distribution."""
    path_id = args.training_min_path <= path_length <= args.training_max_path
    chain_id = args.training_min_chain_size <= chain_size <= args.training_max_chain_size
    return path_id and chain_id


def main():
    args = parse_args()
    
    # Set random seeds for full reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse chain sizes
    chain_sizes = None
    if args.chain_sizes:
        chain_sizes = [int(x.strip()) for x in args.chain_sizes.split(',')]
    
    print(f"=" * 70)
    print(f"Equation Reasoning Model Evaluation")
    print(f"=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Training path lengths: {args.training_min_path}-{args.training_max_path} steps")
    print(f"Training chain sizes: {args.training_min_chain_size}-{args.training_max_chain_size} nodes")
    print(f"Test path lengths: {args.min_path_length}-{args.max_path_length} steps")
    if chain_sizes:
        print(f"Test chain sizes: {chain_sizes}")
    else:
        print(f"Test chain sizes: auto (path_length + 1 nodes)")
    print(f"Samples per configuration: {args.samples_per_config}")
    print(f"=" * 70)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models during generation
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Generate test samples for each (chain_size, path_length) configuration
    path_lengths = list(range(args.min_path_length, args.max_path_length + 1))
    print(f"\nGenerating test samples...")
    print(f"  Path lengths: {path_lengths}")
    if chain_sizes:
        print(f"  Chain sizes: {chain_sizes}")
    
    samples_by_config = generate_test_samples_by_config(
        chain_sizes=chain_sizes,
        path_lengths=path_lengths,
        samples_per_config=args.samples_per_config,
        max_label_value=args.max_label_value,
        seed=args.seed,
    )
    
    # Evaluate on each configuration
    print("\n" + "=" * 70)
    print("Evaluating by (chain_size, path_length) configuration...")
    print("=" * 70)
    
    results_by_config = {}
    all_results = []
    
    # Group results by chain_size and path_length
    results_by_chain_size = defaultdict(list)
    results_by_path_length = defaultdict(list)
    
    for (chain_size, path_length) in sorted(samples_by_config.keys()):
        samples = samples_by_config[(chain_size, path_length)]
        if not samples:
            continue
        
        # Convert to format expected by evaluate_model
        eval_samples = [(s[0], s[1], s[3]) for s in samples]  # (input, target, path_length)
        
        # Determine if this is in-distribution or out-of-distribution
        is_id = is_in_distribution(chain_size, path_length, args)
        dist_label = "ID" if is_id else "OOD"
        
        print(f"\nChain size {chain_size}, Path length {path_length} [{dist_label}]:")
        
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            samples=eval_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=str(device),
        )
        
        config_result = {
            "chain_size": chain_size,
            "path_length": path_length,
            "is_in_distribution": is_id,
            "accuracy": eval_results["accuracy"],
            "format_accuracy": eval_results["format_accuracy"],
            "efficient_rate": eval_results["efficient_rate"],
            "correct": eval_results["correct"],
            "format_correct": eval_results["format_correct"],
            "efficient_correct_count": eval_results["efficient_correct_count"],
            "total": eval_results["total"],
        }
        
        results_by_config[(chain_size, path_length)] = config_result
        results_by_chain_size[chain_size].append(config_result)
        results_by_path_length[path_length].append(config_result)
        
        # Add chain_size to individual results
        for r in eval_results["results"]:
            r["chain_size"] = chain_size
        all_results.extend(eval_results["results"])
        
        print(f"  Accuracy: {eval_results['accuracy']:.2%} ({eval_results['correct']}/{eval_results['total']})")
        print(f"  Format Accuracy: {eval_results['format_accuracy']:.2%}")
        print(f"  Efficient Reasoning: {eval_results['efficient_rate']:.2%} ({eval_results['efficient_correct_count']}/{eval_results['correct']} correct)")
    
    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("Aggregate Statistics")
    print("=" * 70)
    
    # Overall stats
    all_configs = list(results_by_config.values())
    overall_correct = sum(r["correct"] for r in all_configs)
    overall_total = sum(r["total"] for r in all_configs)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    overall_efficient = sum(r["efficient_correct_count"] for r in all_configs)
    overall_efficient_rate = overall_efficient / overall_correct if overall_correct > 0 else 0
    
    # In-distribution stats
    id_configs = [r for r in all_configs if r["is_in_distribution"]]
    id_correct = sum(r["correct"] for r in id_configs)
    id_total = sum(r["total"] for r in id_configs)
    id_accuracy = id_correct / id_total if id_total > 0 else 0
    id_efficient = sum(r["efficient_correct_count"] for r in id_configs)
    id_efficient_rate = id_efficient / id_correct if id_correct > 0 else 0
    
    # Out-of-distribution stats
    ood_configs = [r for r in all_configs if not r["is_in_distribution"]]
    ood_correct = sum(r["correct"] for r in ood_configs)
    ood_total = sum(r["total"] for r in ood_configs)
    ood_accuracy = ood_correct / ood_total if ood_total > 0 else 0
    ood_efficient = sum(r["efficient_correct_count"] for r in ood_configs)
    ood_efficient_rate = ood_efficient / ood_correct if ood_correct > 0 else 0
    
    print(f"\nOverall: {overall_accuracy:.2%} ({overall_correct}/{overall_total})")
    print(f"  Efficient Reasoning: {overall_efficient_rate:.2%} ({overall_efficient}/{overall_correct} correct)")
    print(f"In-Distribution: {id_accuracy:.2%} ({id_correct}/{id_total})")
    print(f"  Efficient Reasoning: {id_efficient_rate:.2%} ({id_efficient}/{id_correct} correct)")
    print(f"Out-of-Distribution: {ood_accuracy:.2%} ({ood_correct}/{ood_total})")
    print(f"  Efficient Reasoning: {ood_efficient_rate:.2%} ({ood_efficient}/{ood_correct} correct)")
    
    # Stats by chain size
    print("\n" + "-" * 40)
    print("By Chain Size:")
    print("-" * 40)
    for cs in sorted(results_by_chain_size.keys()):
        configs = results_by_chain_size[cs]
        cs_correct = sum(r["correct"] for r in configs)
        cs_total = sum(r["total"] for r in configs)
        cs_accuracy = cs_correct / cs_total if cs_total > 0 else 0
        cs_efficient = sum(r["efficient_correct_count"] for r in configs)
        cs_efficient_rate = cs_efficient / cs_correct if cs_correct > 0 else 0
        cs_is_id = args.training_min_chain_size <= cs <= args.training_max_chain_size
        dist = "ID" if cs_is_id else "OOD"
        print(f"  Chain size {cs} [{dist}]: {cs_accuracy:.2%} ({cs_correct}/{cs_total}), Efficient: {cs_efficient_rate:.2%}")
    
    # Stats by path length
    print("\n" + "-" * 40)
    print("By Path Length:")
    print("-" * 40)
    for pl in sorted(results_by_path_length.keys()):
        configs = results_by_path_length[pl]
        pl_correct = sum(r["correct"] for r in configs)
        pl_total = sum(r["total"] for r in configs)
        pl_accuracy = pl_correct / pl_total if pl_total > 0 else 0
        pl_efficient = sum(r["efficient_correct_count"] for r in configs)
        pl_efficient_rate = pl_efficient / pl_correct if pl_correct > 0 else 0
        pl_is_id = args.training_min_path <= pl <= args.training_max_path
        dist = "ID" if pl_is_id else "OOD"
        print(f"  Path length {pl} [{dist}]: {pl_accuracy:.2%} ({pl_correct}/{pl_total}), Efficient: {pl_efficient_rate:.2%}")
    
    # Save results
    results_summary = {
        "args": vars(args),
        "timestamp": timestamp,
        "results_by_config": {f"{cs},{pl}": v for (cs, pl), v in results_by_config.items()},
        "results_by_chain_size": {
            str(cs): {
                "accuracy": sum(r["correct"] for r in configs) / sum(r["total"] for r in configs) if sum(r["total"] for r in configs) > 0 else 0,
                "correct": sum(r["correct"] for r in configs),
                "total": sum(r["total"] for r in configs),
                "efficient_rate": sum(r["efficient_correct_count"] for r in configs) / sum(r["correct"] for r in configs) if sum(r["correct"] for r in configs) > 0 else 0,
                "efficient_correct_count": sum(r["efficient_correct_count"] for r in configs),
            }
            for cs, configs in results_by_chain_size.items()
        },
        "results_by_path_length": {
            str(pl): {
                "accuracy": sum(r["correct"] for r in configs) / sum(r["total"] for r in configs) if sum(r["total"] for r in configs) > 0 else 0,
                "correct": sum(r["correct"] for r in configs),
                "total": sum(r["total"] for r in configs),
                "efficient_rate": sum(r["efficient_correct_count"] for r in configs) / sum(r["correct"] for r in configs) if sum(r["correct"] for r in configs) > 0 else 0,
                "efficient_correct_count": sum(r["efficient_correct_count"] for r in configs),
            }
            for pl, configs in results_by_path_length.items()
        },
        "aggregate": {
            "overall": {
                "accuracy": overall_accuracy,
                "correct": overall_correct,
                "total": overall_total,
                "efficient_rate": overall_efficient_rate,
                "efficient_correct_count": overall_efficient,
            },
            "in_distribution": {
                "accuracy": id_accuracy,
                "correct": id_correct,
                "total": id_total,
                "efficient_rate": id_efficient_rate,
                "efficient_correct_count": id_efficient,
            },
            "out_of_distribution": {
                "accuracy": ood_accuracy,
                "correct": ood_correct,
                "total": ood_total,
                "efficient_rate": ood_efficient_rate,
                "efficient_correct_count": ood_efficient,
            },
        },
    }
    
    results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save sample outputs if requested
    if args.save_samples:
        samples_file = os.path.join(args.output_dir, f"sample_outputs_{timestamp}.json")
        with open(samples_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Sample outputs saved to: {samples_file}")
    
    # Print detailed table for easy copying
    print("\n" + "=" * 70)
    print("Detailed Results Table (for plotting)")
    print("=" * 70)
    print(f"{'Chain Size':<12}{'Path Len':<10}{'Dist':<6}{'Accuracy':<12}{'Efficient':<12}{'Correct':<10}{'Total':<8}")
    print("-" * 70)
    for (cs, pl) in sorted(results_by_config.keys()):
        r = results_by_config[(cs, pl)]
        dist = "ID" if r["is_in_distribution"] else "OOD"
        print(f"{cs:<12}{pl:<10}{dist:<6}{r['accuracy']:<12.4f}{r['efficient_rate']:<12.4f}{r['correct']:<10}{r['total']:<8}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

