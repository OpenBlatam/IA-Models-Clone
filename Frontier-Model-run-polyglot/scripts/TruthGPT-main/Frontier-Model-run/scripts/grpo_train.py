# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import DatasetDict, load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

from scripts.config_parser import DEFAULT_EVAL_STRATEGY
from scripts.logging_utils import setup_logging as setup_logging_util

DEFAULT_DATASET_BATCH_SIZE = 100
DEFAULT_DATASET_DESC = "Formatting"
DEFAULT_DATASET_NUM_PROC = 4
DEFAULT_MESSAGES_COLUMN = "messages"
DEFAULT_MODEL_TAG = "open-r1"
LOG_EVAL_SECTION = "*** Evaluate ***"
LOG_PUSHING_TO_HUB = "Pushing to hub..."
LOG_SAVE_MODEL_SECTION = "*** Save model ***"
LOG_TRAIN_SECTION = "*** Train ***"
METRICS_EVAL = "eval"
METRICS_EVAL_SAMPLES = "eval_samples"
METRICS_TRAIN = "train"
METRICS_TRAIN_SAMPLES = "train_samples"

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script."""
    reward_funcs: List[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )

REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "cosine": lambda args: get_cosine_scaled_reward(
        min_value_wrong=args.cosine_min_value_wrong,
        max_value_wrong=args.cosine_max_value_wrong,
        min_value_correct=args.cosine_min_value_correct,
        max_value_correct=args.cosine_max_value_correct,
        max_len=args.cosine_max_len,
    ),
    "repetition_penalty": lambda args: get_repetition_penalty_reward(
        ngram_size=args.repetition_n_grams,
        max_penalty=args.repetition_max_penalty,
    ),
    "length": len_reward,
    "code": code_reward,
    "code_format": lambda args: get_code_format_reward(language=args.code_language),
    "tag_count": tag_count_reward,
}

def _is_callable_with_args(func: Any) -> bool:
    """Check if function is callable and requires arguments."""
    if not callable(func):
        return False
    code = getattr(func, '__code__', None)
    return code is not None and code.co_argcount > 0


def make_conversation(example: Dict, system_prompt: Optional[str] = None) -> Dict:
    """Create conversation format from example.
    
    Args:
        example: Dictionary containing the problem to convert
        system_prompt: Optional system prompt to include
        
    Returns:
        Dictionary with 'prompt' key containing conversation format
        
    Raises:
        ValueError: If example doesn't contain 'problem' key
    """
    if "problem" not in example:
        raise ValueError("Example must contain 'problem' key")
    
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["problem"]})
    return {"prompt": prompt}


def _get_torch_dtype(model_args: ModelConfig) -> Union[str, torch.dtype, None]:
    """Get torch dtype from model arguments."""
    if model_args.torch_dtype in ["auto", None]:
        return model_args.torch_dtype
    
    if not isinstance(model_args.torch_dtype, str):
        raise ValueError(f"torch_dtype must be a string, got {type(model_args.torch_dtype)}")
    
    dtype = getattr(torch, model_args.torch_dtype, None)
    if dtype is None:
        raise ValueError(f"Invalid torch dtype: {model_args.torch_dtype}")
    return dtype


def _build_model_kwargs(model_args: ModelConfig, training_args: ModelConfig) -> Dict[str, Any]:
    """Build model initialization kwargs."""
    torch_dtype = _get_torch_dtype(model_args)
    return {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'attn_implementation': model_args.attn_implementation,
        'torch_dtype': torch_dtype,
        'use_cache': not training_args.gradient_checkpointing,
    }


def _initialize_training(training_args: ModelConfig, logger: logging.Logger) -> None:
    """Initialize training environment."""
    set_seed(training_args.seed)
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)


def _get_checkpoint(training_args: ModelConfig, logger: logging.Logger) -> Optional[str]:
    """Get checkpoint for resuming training."""
    output_dir = Path(training_args.output_dir)
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if output_dir.is_dir() else None
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")
    return last_checkpoint


def _log_training_config(script_args: GRPOScriptArguments, training_args: ModelConfig, model_args: ModelConfig, logger: logging.Logger) -> None:
    """Log training configuration."""
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


def get_reward_funcs(args: GRPOScriptArguments) -> List[Callable]:
    """Get reward functions from registry based on script arguments."""
    if not args.reward_funcs:
        raise ValueError("At least one reward function must be specified")
    
    funcs = []
    for func_name in args.reward_funcs:
        if func_name not in REWARD_FUNCS_REGISTRY:
            raise ValueError(f"Unknown reward function: {func_name}. Available: {list(REWARD_FUNCS_REGISTRY.keys())}")
        
        func = REWARD_FUNCS_REGISTRY[func_name]
        if _is_callable_with_args(func):
            try:
                funcs.append(func(args))
            except Exception as e:
                logging.warning(f"Failed to initialize reward function '{func_name}': {e}. Using function directly.")
                funcs.append(func)
        else:
            funcs.append(func)
    return funcs


def load_and_prepare_dataset(script_args: GRPOScriptArguments, training_args: ModelConfig, logger: logging.Logger) -> DatasetDict:
    """Load and prepare dataset for training."""
    if not script_args.dataset_name:
        raise ValueError("dataset_name must be specified")
    
    try:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{script_args.dataset_name}': {e}")
    
    if not dataset:
        raise ValueError(f"Dataset '{script_args.dataset_name}' is empty")
    
    logger.info("Formatting dataset into conversations...")
    try:
        dataset = dataset.map(
            lambda ex: make_conversation(ex, training_args.system_prompt),
            num_proc=DEFAULT_DATASET_NUM_PROC,
            desc=DEFAULT_DATASET_DESC,
            batch_size=DEFAULT_DATASET_BATCH_SIZE,
            batched=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to format dataset: {e}")
    
    for split in dataset:
        if DEFAULT_MESSAGES_COLUMN in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns(DEFAULT_MESSAGES_COLUMN)
    return dataset


def build_trainer(model_args: ModelConfig, training_args: ModelConfig, script_args: GRPOScriptArguments, reward_funcs: List[Callable], dataset: DatasetDict, tokenizer: Any, logger: logging.Logger) -> GRPOTrainer:
    """Build and configure GRPOTrainer."""
    model_kwargs = _build_model_kwargs(model_args, training_args)
    training_args.model_init_kwargs = model_kwargs
    return GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != DEFAULT_EVAL_STRATEGY else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )


def _log_and_save_metrics(trainer: GRPOTrainer, metrics: Dict[str, Any], metric_prefix: str, logger: logging.Logger) -> None:
    """Log and save metrics for training or evaluation.
    
    Args:
        trainer: The GRPOTrainer instance
        metrics: Dictionary of metrics to log and save
        metric_prefix: Prefix for metric keys (e.g., 'train' or 'eval')
        logger: Logger instance
    """
    trainer.log_metrics(metric_prefix, metrics)
    trainer.save_metrics(metric_prefix, metrics)


def _train_model(trainer: GRPOTrainer, training_args: ModelConfig, last_checkpoint: Optional[str], dataset: DatasetDict, script_args: GRPOScriptArguments, logger: logging.Logger) -> None:
    """Train the model.
    
    Args:
        trainer: The GRPOTrainer instance
        training_args: Training configuration
        last_checkpoint: Optional checkpoint path to resume from
        dataset: Training dataset
        script_args: Script arguments
        logger: Logger instance
    """
    logger.info(LOG_TRAIN_SECTION)
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics[METRICS_TRAIN_SAMPLES] = len(dataset[script_args.dataset_train_split])
    _log_and_save_metrics(trainer, metrics, METRICS_TRAIN, logger)
    trainer.save_state()


def _evaluate_model(trainer: GRPOTrainer, dataset: DatasetDict, script_args: GRPOScriptArguments, logger: logging.Logger) -> None:
    """Evaluate the model.
    
    Args:
        trainer: The GRPOTrainer instance
        dataset: Evaluation dataset
        script_args: Script arguments
        logger: Logger instance
    """
    logger.info(LOG_EVAL_SECTION)
    metrics = trainer.evaluate()
    metrics[METRICS_EVAL_SAMPLES] = len(dataset[script_args.dataset_test_split])
    _log_and_save_metrics(trainer, metrics, METRICS_EVAL, logger)


def train_and_evaluate(trainer: GRPOTrainer, dataset: DatasetDict, script_args: GRPOScriptArguments, training_args: ModelConfig, last_checkpoint: Optional[str], logger: logging.Logger) -> None:
    """Train and optionally evaluate the model."""
    _train_model(trainer, training_args, last_checkpoint, dataset, script_args, logger)
    if training_args.do_eval:
        _evaluate_model(trainer, dataset, script_args, logger)


def _save_model_config(trainer: GRPOTrainer, training_args: ModelConfig, kwargs: Dict[str, Any]) -> None:
    """Save model configuration."""
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


def save_and_push(trainer: GRPOTrainer, training_args: ModelConfig, script_args: GRPOScriptArguments, logger: logging.Logger) -> None:
    """Save model and optionally push to hub."""
    logger.info(LOG_SAVE_MODEL_SECTION)
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": [DEFAULT_MODEL_TAG],
    }
    _save_model_config(trainer, training_args, kwargs)
    if training_args.push_to_hub:
        logger.info(LOG_PUSHING_TO_HUB)
        trainer.push_to_hub(**kwargs)


def main(script_args: GRPOScriptArguments, training_args: ModelConfig, model_args: ModelConfig) -> None:
    """Main training function."""
    logger = setup_logging_util(training_args.get_process_log_level())
    _log_training_config(script_args, training_args, model_args, logger)
    _initialize_training(training_args, logger)
    last_checkpoint = _get_checkpoint(training_args, logger)
    
    dataset = load_and_prepare_dataset(script_args, training_args, logger)
    tokenizer = get_tokenizer(model_args, training_args)
    reward_funcs = get_reward_funcs(script_args)
    trainer = build_trainer(model_args, training_args, script_args, reward_funcs, dataset, tokenizer, logger)
    train_and_evaluate(trainer, dataset, script_args, training_args, last_checkpoint, logger)
    save_and_push(trainer, training_args, script_args, logger)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args) 