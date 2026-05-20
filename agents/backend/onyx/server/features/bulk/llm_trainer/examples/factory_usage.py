"""
Factory Pattern Usage Example
==============================

Example showing how to use TrainerFactory for creating trainers.

Author: BUL System
Date: 2024
"""

from llm_trainer import TrainerFactory

# Create basic trainer
factory = TrainerFactory()
trainer = factory.create_basic_trainer(
    model_name="gpt2",
    dataset_path="data/training.json"
)

# Or create advanced trainer with evaluation
trainer = factory.create_advanced_trainer(
    model_name="gpt2",
    dataset_path="data/training.json",
    enable_early_stopping=True,
    enable_evaluation=True
)

# Or create memory-efficient trainer
trainer = factory.create_memory_efficient_trainer(
    model_name="gpt2",
    dataset_path="data/training.json"
)

trainer.train()

