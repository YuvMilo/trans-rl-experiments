import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

# Import our simplified DAG dataset
from dag_dataset_simplified import create_dag_dataset_dict

# Global variable to store run timestamp for consistent file naming
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# Set in grpo_function from config; used by format_reward_func for write_samples_probability
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
    min_chain_length: int = 9
    max_chain_length: int = 17
    sampling_strategy: str = "stratified"
    max_label_value: int = 40
    fixed_label_set: bool = False
    min_distance: int = 10
    max_distance: int = None
    min_start_value: int = 1
    max_start_value: int = 10
    write_samples_probability: float = 1.0  # chance (0-1) to write each completion to completion_samples file


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
# Helper functions
########################

def format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        **kwargs: Additional parameters including 'target' and 'input' from the dataset
      
      Returns:
          list[float]: Reward scores
    """
    # Extract target and input_data from kwargs (passed by GRPO trainer)
    target = kwargs.get('target', [])
    input_data = kwargs.get('input', [])
    
    rewards = []

    for i, completion in enumerate(completions):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        if random.random() < WRITE_SAMPLES_PROBABILITY:
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", f"completion_samples_dag_{RUN_TIMESTAMP}.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            # Get corresponding input and target for this completion
            dag_input = input_data[i] if i < len(input_data) else "N/A"
            gt = target[i] if i < len(target) else "N/A"
            f.write(f"Input: {dag_input}\n")
            f.write(f"Target: {gt}\n")
            f.write(f"Completion: {completion}\n")
        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def dag_answer_reward_func(completions, **kwargs):
    """
    Evaluates completions based on correctness of the DAG chain reasoning answer.

    Args:
        completions (list[str]): Generated outputs
        **kwargs: Contains 'target' and 'input' from the dataset
    
    Returns:
        list[float]: Reward scores
    """
    # Extract target and input_data from kwargs (passed by GRPO trainer)
    target = kwargs.get('target', [])
    input_data = kwargs.get('input', [])
    
    rewards = []
    for i, completion in enumerate(completions):
      try:
        # Safely access target and input with bounds checking
        gt = target[i] if i < len(target) else ""
        dag_input = input_data[i] if i < len(input_data) else ""
        
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        
        # Extract the "answer" part from the completion
        answer = match.group(1).strip()
        
        # Remove any extra whitespace and normalize
        answer = re.sub(r'\s+', '', answer)
        gt = re.sub(r'\s+', '', str(gt))
        
        # Check if the answer matches the ground truth
        if answer == gt:
            rewards.append(1.0)
            if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", f"success_completion_samples_dag_{RUN_TIMESTAMP}.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"Input: {dag_input}\n")
                    f.write(f"Target: {gt}\n")
                    f.write(f"Completion: {completion}\n")
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
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
    #########################
    # Log parameters
    #########################
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
    logger.info("Generating DAG chain reasoning dataset...")
    dataset_dict = create_dag_dataset_dict(
        n_samples=script_args.dataset_samples,
        min_nodes=script_args.min_nodes,
        max_nodes=script_args.max_nodes,
        min_chain_length=script_args.min_chain_length,
        max_chain_length=script_args.max_chain_length,
        sampling_strategy=script_args.sampling_strategy,
        max_label_value=script_args.max_label_value,
        fixed_label_set=script_args.fixed_label_set,
        min_distance=script_args.min_distance,
        max_distance=script_args.max_distance,
        min_start_value=script_args.min_start_value,
        max_start_value=script_args.max_start_value,
        randomize_labels=True
    )
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Generated {len(dataset)} DAG chain reasoning samples")

    #####################
    # Prepare and format dataset
    #####################

    # Generate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_dag_prompt(dag_input, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant that can solve arithmetic equations systems."
          },
          { 
            "role": "user",
            "content": f"Given this equation system: {dag_input}\n\nFind the value of the target variable. Show your reasoning in <think> </think> tags, and provide your final answer in <answer> </answer> tags, for example <answer>12</answer>."
          },
          {
            "role": "assistant",
            "content": "I'll think about the equation system and solve it.\n<think>"

          }]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
            "target": target, 
            "input": dag_input
        }

    # Convert our dataset to the r1 prompt format
    dataset = dataset.map(lambda x: generate_r1_dag_prompt(x["input"], x["target"]))

    # Split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Make write_samples_probability available to format_reward_func
    global WRITE_SAMPLES_PROBABILITY
    WRITE_SAMPLES_PROBABILITY = script_args.write_samples_probability

    #########################
    # Instantiate GRPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, dag_answer_reward_func],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting DAG chain reasoning training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
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
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "dag-reasoning", "chain-tracing"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** DAG chain reasoning training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
