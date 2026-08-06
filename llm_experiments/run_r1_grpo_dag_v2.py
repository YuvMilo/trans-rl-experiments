"""GRPO training script for DAG equation reasoning.

Adapted for trl 1.2.0 / transformers 5.5.4.
Breaking changes handled vs the original run_r1_grpo_dag.py:
  - `ModelConfig.torch_dtype` renamed to `ModelConfig.dtype`
  - `GRPOConfig.max_prompt_length` removed; prompt-length capping is
    now done at the dataset level via `ScriptArguments.max_prompt_length`
  - The monkey-patch for `vllm_max_model_len` is gone; use the native
    `GRPOConfig.vllm_max_model_length` field instead
"""
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Required for gated models (e.g. Llama). Export HF_TOKEN before launching.
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN
import random
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

from dag_dataset_simplified import create_dag_dataset_dict

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
WRITE_SAMPLES_PROBABILITY = 1.0


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_samples: int = 10000
    tokenizer_name_or_path: str = None
    min_nodes: int = 20
    max_nodes: int = 20
    graph_type: str = "chain"
    # Chain-specific (backward compat)
    min_chain_length: int = 9
    max_chain_length: int = 17
    min_distance: int = 10
    max_distance: int = None
    # DAG / tree params
    max_in_degree: int = 3
    max_out_degree: int = 3
    edge_probability: float = 0.3
    num_roots: int = None
    # Polynomial equation params
    max_degree: int = 1
    max_coefficient: int = 1
    max_terms: int = 1
    terms_equal_in_degree: bool = False
    probabilistic_pairwise_terms: bool = False
    single_variable_term_probability: float = 0.5
    pairwise_product_term_probability: float = 0.5
    constant_term_probability: float = 0.5
    min_constant: int = 1
    max_constant: int = 3
    # Target selection
    target_sink_only: bool = True
    min_depth: int = None
    max_depth: int = None
    max_value: int = 10000
    modulus: int = None
    # Ancestor-subgraph size control (tree / reverse_tree / dag only)
    min_ancestor_nodes: int = None
    max_ancestor_nodes: int = None
    # Force a unique topological order on the ancestor subgraph (dag only)
    force_unique_topo_order: bool = False
    # Bucketed depth control (dag only). Partitions ALL n_nodes into
    # `ancestor_depth` buckets (each ≥1 node) and only allows edges that
    # strictly descend bucket levels, fixing the target's ancestor-graph
    # depth to `ancestor_depth - 1`. Enable via min+max endpoints (equal
    # for a fixed depth, or a contiguous range) OR via an explicit
    # `ancestor_depths` list (sparse). The two are mutually exclusive;
    # both take precedence over `min_ancestor_nodes`/`max_ancestor_nodes`
    # and conflict with `force_unique_topo_order`.
    min_ancestor_depth: int = None
    max_ancestor_depth: int = None
    ancestor_depths: List[int] = None
    # Labels & values
    sampling_strategy: str = "stratified"
    max_label_value: int = 40
    fixed_label_set: bool = False
    min_start_value: int = 1
    max_start_value: int = 10
    write_samples_probability: float = 1.0
    # Prompt length cap: replaces the removed GRPOConfig.max_prompt_length.
    # Examples whose tokenized prompt exceeds this are dropped from the dataset.
    max_prompt_length: int = 512


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Reward functions
########################

def format_reward_func(completions, **kwargs):
    """+1 if completion matches <think>...</think>\n<answer>...</answer>, else 0."""
    target = kwargs.get('target', [])
    input_data = kwargs.get('input', [])

    rewards = []

    for i, completion in enumerate(completions):
        try:
            completion = "<think>" + completion
            if random.random() < WRITE_SAMPLES_PROBABILITY:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join(
                    "completion_samples",
                    f"completion_samples_dag_{RUN_TIMESTAMP}.txt",
                )
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    dag_input = input_data[i] if i < len(input_data) else "N/A"
                    gt = target[i] if i < len(target) else "N/A"
                    f.write(f"Input: {dag_input}\n")
                    f.write(f"Target: {gt}\n")
                    f.write(f"Completion: {completion}\n")

            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def dag_answer_reward_func(completions, **kwargs):
    """+1 if the extracted <answer> exactly matches the ground-truth target, else 0."""
    target = kwargs.get('target', [])
    input_data = kwargs.get('input', [])

    rewards = []
    for i, completion in enumerate(completions):
        try:
            gt = target[i] if i < len(target) else ""
            dag_input = input_data[i] if i < len(input_data) else ""

            completion = "<think>" + completion
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue

            answer = match.group(1).strip()
            answer = re.sub(r'\s+', '', answer)
            gt = re.sub(r'\s+', '', str(gt))

            if answer == gt:
                rewards.append(1.0)
                if random.random() < 0.10:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples",
                        f"success_completion_samples_dag_{RUN_TIMESTAMP}.txt",
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(f"Input: {dag_input}\n")
                        f.write(f"Target: {gt}\n")
                        f.write(f"Completion: {completion}\n")
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Generate datasets
    ###############
    logger.info(f"Generating {script_args.graph_type} equation reasoning dataset...")
    dataset_dict = create_dag_dataset_dict(
        n_samples=script_args.dataset_samples,
        min_nodes=script_args.min_nodes,
        max_nodes=script_args.max_nodes,
        graph_type=script_args.graph_type,
        min_chain_length=script_args.min_chain_length,
        max_chain_length=script_args.max_chain_length,
        min_distance=script_args.min_distance,
        max_distance=script_args.max_distance,
        max_in_degree=script_args.max_in_degree,
        max_out_degree=script_args.max_out_degree,
        edge_probability=script_args.edge_probability,
        num_roots=script_args.num_roots,
        max_degree=script_args.max_degree,
        max_coefficient=script_args.max_coefficient,
        max_terms=script_args.max_terms,
        terms_equal_in_degree=script_args.terms_equal_in_degree,
        probabilistic_pairwise_terms=script_args.probabilistic_pairwise_terms,
        single_variable_term_probability=script_args.single_variable_term_probability,
        pairwise_product_term_probability=script_args.pairwise_product_term_probability,
        constant_term_probability=script_args.constant_term_probability,
        min_constant=script_args.min_constant,
        max_constant=script_args.max_constant,
        target_sink_only=script_args.target_sink_only,
        min_depth=script_args.min_depth,
        max_depth=script_args.max_depth,
        max_value=script_args.max_value,
        modulus=script_args.modulus,
        min_ancestor_nodes=script_args.min_ancestor_nodes,
        max_ancestor_nodes=script_args.max_ancestor_nodes,
        force_unique_topo_order=script_args.force_unique_topo_order,
        min_ancestor_depth=script_args.min_ancestor_depth,
        max_ancestor_depth=script_args.max_ancestor_depth,
        ancestor_depths=script_args.ancestor_depths,
        sampling_strategy=script_args.sampling_strategy,
        max_label_value=script_args.max_label_value,
        fixed_label_set=script_args.fixed_label_set,
        min_start_value=script_args.min_start_value,
        max_start_value=script_args.max_start_value,
        randomize_labels=True,
    )

    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Generated {len(dataset)} DAG chain reasoning samples")

    #####################
    # Prepare and format dataset
    #####################

    def generate_r1_dag_prompt(dag_input, target):
        r1_prefix = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can solve systems of equations.",
            },
            {
                "role": "user",
                "content": (
                    f"Given this equation system: {dag_input}\n\nFind the value of the "
                    "target variable. Show your reasoning in <think> </think> tags, and "
                    "provide your final answer in <answer> </answer> tags, for example "
                    "<answer>12</answer>."
                ),
            },
            {
                "role": "assistant",
                "content": "I'll think about the equation system and solve it.\n<think>",
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenize=False, continue_final_message=True
            ),
            "target": target,
            "input": dag_input,
        }

    dataset = dataset.map(lambda x: generate_r1_dag_prompt(x["input"], x["target"]))

    # Drop examples whose tokenized prompt exceeds max_prompt_length.
    # Replaces the removed GRPOConfig.max_prompt_length truncation.
    max_prompt_length = script_args.max_prompt_length
    if max_prompt_length is not None:
        pre = len(dataset)

        def _within_length(example):
            ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
            return len(ids) <= max_prompt_length

        dataset = dataset.filter(_within_length, num_proc=4)
        logger.info(
            f"Filtered prompts by max_prompt_length={max_prompt_length}: "
            f"{pre} -> {len(dataset)} samples"
        )

    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    global WRITE_SAMPLES_PROBABILITY
    WRITE_SAMPLES_PROBABILITY = script_args.write_samples_probability

    #########################
    # Instantiate GRPO trainer
    #########################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, dag_answer_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    logger.info(
        f'*** Starting DAG chain reasoning training '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
        f'for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "dag-reasoning", "chain-tracing"]})
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** DAG chain reasoning training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
