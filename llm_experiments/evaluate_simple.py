#!/usr/bin/env python3
"""Single-config evaluation script.

Generate ``--n_samples`` examples with the supplied dataset parameters
(the same knobs used in ``run_r1_grpo_dag_v2.py``) and report the
model's accuracy / format-accuracy / efficient-rate on those samples.

No (num_equations, depth) bucket grid, no in-/out-of-distribution
labels, no backward-compat aliases. One distribution in, one accuracy
number out.
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import List, Optional, Set, Tuple

# Set HF_TOKEN for gated models (e.g. Llama). Provide via env var or paste here.
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dag_dataset_simplified import generate_unique_equation_samples


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool,
                  help_text: str = "") -> None:
    """Register a paired ``--{name}`` / ``--no_{name}`` boolean flag.

    ``argparse.BooleanOptionalAction`` is 3.9+; this works on 3.7+.
    """
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=name, action="store_true",
                     help=help_text + (" (default: enabled)" if default else ""))
    grp.add_argument(f"--no_{name}", dest=name, action="store_false",
                     help=f"disable {name}")
    parser.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a model on samples drawn from a single dataset configuration."
    )

    # Model / runtime
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=4096,
                   help="Tokenizer truncation cap for the prompt.")
    p.add_argument("--output_dir", type=str, default="evaluation_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_samples", action="store_true",
                   help="Also save per-sample completions to a JSON file.")

    # ── Dataset generation params ─────────────────────────────────────
    # Topology
    p.add_argument("--graph_type", type=str, default="chain",
                   choices=["chain", "tree", "reverse_tree", "dag"])
    p.add_argument("--min_nodes", type=int, default=15)
    p.add_argument("--max_nodes", type=int, default=15)

    # Chain-specific
    p.add_argument("--min_chain_length", type=int, default=2)
    p.add_argument("--max_chain_length", type=int, default=6)
    p.add_argument("--min_distance", type=int, default=1)
    p.add_argument("--max_distance", type=int, default=None)
    p.add_argument("--operations", type=str, default="+,-",
                   help="Comma-separated chain operators.")
    p.add_argument("--min_constant", type=int, default=1)

    # DAG / tree topology
    p.add_argument("--max_in_degree", type=int, default=3)
    p.add_argument("--max_out_degree", type=int, default=3)
    p.add_argument("--edge_probability", type=float, default=0.3)
    p.add_argument("--num_roots", type=int, default=None)

    # Polynomial equation params
    p.add_argument("--max_degree", type=int, default=1)
    p.add_argument("--max_coefficient", type=int, default=1)
    p.add_argument("--max_terms", type=int, default=1)
    _add_bool_arg(p, "terms_equal_in_degree", default=False,
                  help_text="Force exactly one term per parent variable.")
    _add_bool_arg(p, "probabilistic_pairwise_terms", default=False,
                  help_text="Independently sample single/pairwise/constant terms.")
    p.add_argument("--single_variable_term_probability", type=float, default=0.5)
    p.add_argument("--pairwise_product_term_probability", type=float, default=0.5)
    p.add_argument("--constant_term_probability", type=float, default=0.5)
    p.add_argument("--max_constant", type=int, default=3)

    # Values
    p.add_argument("--min_start_value", type=int, default=1)
    p.add_argument("--max_start_value", type=int, default=10)
    p.add_argument("--max_value", type=int, default=10000)
    p.add_argument("--modulus", type=int, default=None)

    # Target selection
    _add_bool_arg(p, "target_sink_only", default=True,
                  help_text="Restrict candidate target nodes to graph sinks.")
    p.add_argument("--min_depth", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)

    # Ancestor controls (tree / reverse_tree / dag)
    p.add_argument("--min_ancestor_nodes", type=int, default=None)
    p.add_argument("--max_ancestor_nodes", type=int, default=None)
    p.add_argument("--force_unique_topo_order", action="store_true")
    p.add_argument("--min_ancestor_depth", type=int, default=None)
    p.add_argument("--max_ancestor_depth", type=int, default=None)
    p.add_argument(
        "--ancestor_depths",
        type=int,
        nargs="+",
        default=None,
        help="Sparse list of ancestor depths (mutually exclusive with "
             "min/max_ancestor_depth).",
    )

    # Labels
    _add_bool_arg(p, "randomize_labels", default=True,
                  help_text="Randomly relabel graph node ids.")
    p.add_argument("--max_label_value", type=int, default=40)
    _add_bool_arg(p, "fixed_label_set", default=False,
                  help_text="Sample labels from [0, max_nodes) without replacement.")

    # Sampling
    p.add_argument("--sampling_strategy", type=str, default="uniform_size",
                   choices=["uniform_size", "stratified", "exponential",
                            "quadratic", "proportional"])
    _add_bool_arg(p, "shuffle_equations", default=True,
                  help_text="Shuffle equations in the formatted prompt.")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Prompt + parsing helpers (must match training format exactly)
# ──────────────────────────────────────────────────────────────────────

PROMPT_MARKER = "I'll think about the equation system and solve it.\n<think>"


def create_prompt(tokenizer, equation_input: str) -> str:
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant that can solve systems of equations."},
        {"role": "user",
         "content": (
             f"Given this equation system: {equation_input}\n\n"
             "Find the value of the target variable. Show your reasoning in "
             "<think> </think> tags, and provide your final answer in "
             "<answer> </answer> tags, for example <answer>12</answer>."
         )},
        {"role": "assistant",
         "content": PROMPT_MARKER},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, continue_final_message=True
    )


def extract_answer(completion: str) -> Optional[str]:
    completion = "<think>" + completion
    m = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not m:
        return None
    return re.sub(r"\s+", "", m.group(1).strip())


def check_format(completion: str) -> bool:
    completion = "<think>" + completion
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\n<answer>([\s\S]*?)</answer>$"
    m = re.search(regex, completion, re.DOTALL)
    return m is not None and len(m.groups()) == 2


def parse_equation_graph(equation_input: str) -> Tuple[Optional[nx.DiGraph], Set[int], int]:
    target_match = re.search(r"Find x_(\d+)", equation_input)
    if not target_match:
        return None, set(), -1
    target_node = int(target_match.group(1))

    g = nx.DiGraph()
    root_nodes: Set[int] = set()
    parts = equation_input.split(".")

    # Pass 1: identify root assignments (RHS has no x_ ref).
    for part in parts:
        part = part.strip()
        if not part or part.startswith("Find"):
            continue
        m = re.match(r"x_(\d+)\s*=\s*(.*)", part)
        if not m:
            continue
        lhs = int(m.group(1))
        rhs = m.group(2).strip()
        if not re.search(r"x_\d+", rhs):
            root_nodes.add(lhs)
            g.add_node(lhs)

    # Pass 2: dependency edges for non-root LHS variables.
    for part in parts:
        part = part.strip()
        if not part or part.startswith("Find"):
            continue
        m = re.match(r"x_(\d+)\s*=\s*(.*)", part)
        if not m:
            continue
        lhs = int(m.group(1))
        rhs = m.group(2).strip()
        rhs_vars = {int(v) for v in re.findall(r"x_(\d+)", rhs)}
        if rhs_vars and lhs not in root_nodes:
            g.add_node(lhs)
            for rv in rhs_vars:
                g.add_node(rv)
                g.add_edge(rv, lhs)

    if not root_nodes:
        return None, set(), -1
    return g, root_nodes, target_node


def calculate_actual_depth(equation_input: str) -> int:
    g, _, target = parse_equation_graph(equation_input)
    if g is None:
        return -1
    depths: dict = {}
    try:
        for node in nx.topological_sort(g):
            preds = list(g.predecessors(node))
            depths[node] = 0 if not preds else max(depths.get(p, 0) for p in preds) + 1
    except nx.NetworkXUnfeasible:
        return -1
    return depths.get(target, -1)


def get_ancestor_variables(equation_input: str) -> Set[int]:
    g, _, target = parse_equation_graph(equation_input)
    if g is None:
        return set()
    try:
        return nx.ancestors(g, target) | {target}
    except nx.NetworkXError:
        return set()


def get_all_variables(equation_input: str) -> Set[int]:
    return {int(n) for n in re.findall(r"x_(\d+)", equation_input)}


def extract_mentioned_variables(completion: str) -> Set[int]:
    return {int(n) for n in re.findall(r"x_\{?(\d+)\}?", completion)}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _generate_dataset(args: argparse.Namespace) -> List[Tuple[str, str]]:
    operations = [op.strip() for op in args.operations.split(",") if op.strip()]
    return generate_unique_equation_samples(
        n_samples=args.n_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        graph_type=args.graph_type,
        # chain
        min_chain_length=args.min_chain_length,
        max_chain_length=args.max_chain_length,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        operations=operations,
        # dag/tree topology
        max_in_degree=args.max_in_degree,
        max_out_degree=args.max_out_degree,
        edge_probability=args.edge_probability,
        num_roots=args.num_roots,
        # polynomial
        max_degree=args.max_degree,
        max_coefficient=args.max_coefficient,
        max_terms=args.max_terms,
        terms_equal_in_degree=args.terms_equal_in_degree,
        probabilistic_pairwise_terms=args.probabilistic_pairwise_terms,
        single_variable_term_probability=args.single_variable_term_probability,
        pairwise_product_term_probability=args.pairwise_product_term_probability,
        constant_term_probability=args.constant_term_probability,
        min_constant=args.min_constant,
        max_constant=args.max_constant,
        # values
        min_start_value=args.min_start_value,
        max_start_value=args.max_start_value,
        max_value=args.max_value,
        modulus=args.modulus,
        # target selection
        target_sink_only=args.target_sink_only,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        # ancestor controls
        min_ancestor_nodes=args.min_ancestor_nodes,
        max_ancestor_nodes=args.max_ancestor_nodes,
        force_unique_topo_order=args.force_unique_topo_order,
        min_ancestor_depth=args.min_ancestor_depth,
        max_ancestor_depth=args.max_ancestor_depth,
        ancestor_depths=args.ancestor_depths,
        # labels
        randomize_labels=args.randomize_labels,
        max_label_value=args.max_label_value,
        fixed_label_set=args.fixed_label_set,
        # sampling
        sampling_strategy=args.sampling_strategy,
        shuffle_equations=args.shuffle_equations,
    )


def _run_inference(model, tokenizer, samples: List[Tuple[str, str]],
                   batch_size: int, max_new_tokens: int,
                   max_prompt_length: int, device) -> List[dict]:
    results: List[dict] = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[i:i + batch_size]
        prompts = [create_prompt(tokenizer, s[0]) for s in batch]
        targets = [s[1] for s in batch]
        inputs_str = [s[0] for s in batch]

        toks = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        for output, target, inp_str in zip(outputs, targets, inputs_str):
            full = tokenizer.decode(output, skip_special_tokens=True)
            pos = full.find(PROMPT_MARKER)
            if pos != -1:
                completion = full[pos + len(PROMPT_MARKER):]
            else:
                tp = full.rfind("<think>")
                completion = full[tp + len("<think>"):] if tp != -1 else full

            predicted = extract_answer(completion)
            is_correct = predicted == target if predicted is not None else False
            is_format_correct = check_format(completion)

            necessary = get_ancestor_variables(inp_str)
            unnecessary = get_all_variables(inp_str) - necessary
            mentioned = extract_mentioned_variables(completion)
            mentioned_unnecessary = mentioned & unnecessary
            mentions_unnecessary = bool(mentioned_unnecessary)

            results.append({
                "input": inp_str,
                "target": target,
                "predicted": predicted,
                "completion": completion,
                "is_correct": is_correct,
                "is_format_correct": is_format_correct,
                "mentions_unnecessary": mentions_unnecessary,
                "depth": calculate_actual_depth(inp_str),
                "num_equations": len(get_all_variables(inp_str)),
                "necessary_vars": sorted(necessary),
                "unnecessary_mentioned": sorted(mentioned_unnecessary),
            })
    return results


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"Generating {args.n_samples} samples (graph_type={args.graph_type})")
    print("=" * 70)
    samples = _generate_dataset(args)
    if not samples:
        raise SystemExit("No samples were generated; check your dataset parameters.")
    if len(samples) < args.n_samples:
        print(f"Warning: only {len(samples)}/{args.n_samples} samples generated.")

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    print("\nRunning inference...")
    results = _run_inference(
        model, tokenizer, samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        device=device,
    )

    n = len(results)
    correct = sum(r["is_correct"] for r in results)
    format_correct = sum(r["is_format_correct"] for r in results)
    efficient = sum(1 for r in results if r["is_correct"] and not r["mentions_unnecessary"])

    summary = {
        "accuracy": correct / n if n else 0.0,
        "format_accuracy": format_correct / n if n else 0.0,
        "efficient_rate": efficient / correct if correct else 0.0,
        "correct": correct,
        "format_correct": format_correct,
        "efficient_correct": efficient,
        "total": n,
    }

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Total samples:  {n}")
    print(f"  Accuracy:       {summary['accuracy']:.2%}  ({correct}/{n})")
    print(f"  Format-correct: {summary['format_accuracy']:.2%}  ({format_correct}/{n})")
    print(f"  Efficient:      {summary['efficient_rate']:.2%}  ({efficient}/{correct} correct)")

    out = {
        "args": vars(args),
        "timestamp": timestamp,
        "summary": summary,
    }
    summary_file = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    if args.save_samples:
        samples_file = os.path.join(args.output_dir, f"eval_samples_{timestamp}.json")
        with open(samples_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Per-sample outputs saved to: {samples_file}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
