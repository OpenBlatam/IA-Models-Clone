"""
Monitoring Example
=================

Example showing how to use monitoring and logging features.

Author: BUL System
Date: 2024
"""

from llm_trainer import (
    CustomLLMTrainer,
    TrainingMonitor,
    TensorBoardLogger,
    WandBLogger,
)

# Example 1: Training Monitor
print("=" * 80)
print("Example 1: Training Monitor")
print("=" * 80)

monitor = TrainingMonitor()

# Track metrics
monitor.track_metric("loss", 0.5, step=100)
monitor.track_metric("accuracy", 0.95, step=100)
monitor.track_metrics({"loss": 0.4, "accuracy": 0.96}, step=200)

# Get history
history = monitor.get_history()
print(f"Metrics tracked: {list(history.keys())}")

# Get summary
summary = monitor.get_summary()
print(f"Summary: {summary}")

# Example 2: TensorBoard Integration
print("\n" + "=" * 80)
print("Example 2: TensorBoard Integration")
print("=" * 80)

tb_logger = TensorBoardLogger("./logs/tensorboard")

trainer = CustomLLMTrainer(
    model_name="gpt2",
    dataset_path="data.json",
    output_dir="./checkpoints"
)

# Add custom callback to log to TensorBoard
from transformers import TrainerCallback

class TensorBoardCallback(TrainerCallback):
    def __init__(self, tb_logger):
        self.tb_logger = tb_logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.tb_logger.log_metrics(logs, step=state.global_step)

trainer.add_callback(TensorBoardCallback(tb_logger))

# Train
trainer.train()

# Close logger
tb_logger.close()

# Example 3: Weights & Biases Integration
print("\n" + "=" * 80)
print("Example 3: Weights & Biases Integration")
print("=" * 80)

wandb_logger = WandBLogger(
    project="llm-training",
    name="gpt2-finetune",
    config={
        "model": "gpt2",
        "learning_rate": 3e-5,
        "batch_size": 8,
    }
)

# Log metrics during training
wandb_logger.log_metric("loss", 0.5, step=100)
wandb_logger.log_metrics({"loss": 0.4, "accuracy": 0.95}, step=200)

# Finish
wandb_logger.finish()

