"""
Basic Usage Example
===================

Simple example showing basic usage of CustomLLMTrainer.

Author: BUL System
Date: 2024
"""

from llm_trainer import CustomLLMTrainer

# Basic usage with defaults
trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    output_dir="./checkpoints"
)

trainer.train()

