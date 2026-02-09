#!/usr/bin/env python3
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

import mlflow
from loguru import logger
from transformers import set_seed
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

from scripts.data_loader import load_tokenizer, load_training_dataset
from scripts.experiment_tracking import finish_experiment_tracking, setup_experiment_tracking
from scripts.logging_utils import setup_logging
from scripts.model_factory import create_model_from_config, setup_deepseek_optimizations
from scripts.training_utils import setup_distributed_training, setup_environment, setup_training_config
from scripts.types import KFGRPOScriptArguments

setup_logging()


def _load_data_and_tokenizer(
    script_args: KFGRPOScriptArguments,
    training_args: ModelConfig
) -> tuple:
    """Load dataset and tokenizer.
    
    Args:
        script_args: Script arguments containing dataset configuration
        training_args: Training arguments containing cache directory
        
    Returns:
        Tuple containing (dataset, tokenizer)
    """
    dataset = load_training_dataset(
        dataset_name=script_args.dataset_name,
        dataset_config=script_args.dataset_config,
        cache_dir=training_args.cache_dir,
        streaming=True
    )
    tokenizer = load_tokenizer(
        model_name=script_args.deepseek_config.model_name,
        trust_remote_code=True
    )
    return dataset, tokenizer


def _create_and_optimize_model(script_args: KFGRPOScriptArguments):
    """Create model and apply optimizations if needed.
    
    Args:
        script_args: Script arguments containing model configuration
        
    Returns:
        Configured and optimized model
    """
    model = create_model_from_config(
        deepseek_config=script_args.deepseek_config,
        use_native=script_args.deepseek_config.use_native_implementation,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
        use_deepspeed=script_args.use_deepspeed
    )
    
    if script_args.use_deepseek_optimizations:
        setup_deepseek_optimizations(model, script_args.deepseek_config)
    
    return model


def _create_trainer(
    model,
    dataset,
    tokenizer,
    script_args: KFGRPOScriptArguments,
    training_args: ModelConfig,
    model_args: ModelConfig
) -> GRPOTrainer:
    """Create and configure GRPOTrainer."""
    eval_dataset = (
        dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None
    )
    
    return GRPOTrainer(
        model=model,
        reward_funcs=script_args.reward_funcs,
        args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )


def _train_model(trainer: GRPOTrainer) -> float:
    """Train the model and return final loss."""
    final_loss = trainer.train()
    mlflow.log_metric("final_loss", final_loss)
    mlflow.pytorch.log_model(trainer.model, "model")
    return final_loss


def _save_and_push_model(
    trainer: GRPOTrainer,
    training_args: ModelConfig,
    script_args: KFGRPOScriptArguments
) -> None:
    """Save model and optionally push to hub."""
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def main(
    script_args: KFGRPOScriptArguments,
    training_args: ModelConfig,
    model_args: ModelConfig
) -> None:
    """Main training function."""
    setup_environment()
    config = setup_training_config(script_args)
    setup_experiment_tracking(script_args, config)
    setup_distributed_training(script_args)
    set_seed(training_args.seed)
    
    try:
        dataset, tokenizer = _load_data_and_tokenizer(script_args, training_args)
        model = _create_and_optimize_model(script_args)
        trainer = _create_trainer(
            model, dataset, tokenizer, script_args, training_args, model_args
        )
        _train_model(trainer)
        _save_and_push_model(trainer, training_args, script_args)
            
    except (ValueError, RuntimeError, OSError) as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        raise
    finally:
        finish_experiment_tracking(script_args)

if __name__ == "__main__":
    parser = TrlParser((KFGRPOScriptArguments, ModelConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)    