"""
Builder Pattern Usage Example
==============================

Example showing how to use ConfigBuilder for building configurations.

Author: BUL System
Date: 2024
"""

from llm_trainer import ConfigBuilder, CustomLLMTrainer

# Build configuration using fluent API
config = (ConfigBuilder()
    .with_model("gpt2", model_type="causal")
    .with_dataset("data/training.json")
    .with_output_dir("./checkpoints")
    .with_learning_rate(3e-5)
    .with_epochs(3)
    .with_batch_size(8)
    .with_early_stopping(patience=3)
    .with_gradient_checkpointing(True)
    .with_mixed_precision(fp16=True)
    .build())

# Create trainer from configuration
trainer = CustomLLMTrainer(**config)
trainer.train()

