"""
Plugin System Example
====================

Example showing how to create and use custom plugins.

Author: BUL System
Date: 2024
"""

from llm_trainer import CustomLLMTrainer, CallbackPlugin, PluginRegistry

# Example 1: Custom Callback Plugin
class CustomLoggingPlugin(CallbackPlugin):
    """Custom plugin that logs to a file."""
    
    def __init__(self, log_file: str = "training.log"):
        super().__init__("custom_logging", "1.0.0")
        self.log_file = log_file
    
    def handle_log(self, args, state, control, logs=None, **kwargs):
        """Handle log events."""
        if self.enabled and logs:
            with open(self.log_file, "a") as f:
                f.write(f"Step {state.global_step}: {logs}\n")


# Example 2: Using plugins with trainer
if __name__ == "__main__":
    # Create trainer
    trainer = CustomLLMTrainer(
        model_name="gpt2",
        dataset_path="data/training.json",
        output_dir="./checkpoints"
    )
    
    # Add custom plugin
    plugin = CustomLoggingPlugin("custom_training.log")
    trainer.add_callback(plugin)
    
    # Train
    trainer.train()

