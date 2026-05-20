from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from .tracker import (
from .checkpointing import (
from .metrics import (
from ml.experiment_tracking import create_tracker, create_checkpoint_manager
from ml.config import get_config
from ml.experiment_tracking import TensorBoardTracker, WandbTracker
from ml.experiment_tracking import CompositeTracker, TensorBoardTracker, WandbTracker
from ml.experiment_tracking import CheckpointManager, CheckpointStrategy
from ml.experiment_tracking import MetricsTracker, MetricLogger
from ml.experiment_tracking import create_tracker, create_checkpoint_manager
from ml.config import get_config
from ml.experiment_tracking import create_tracker, create_checkpoint_manager
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Experiment Tracking Module for Key Messages ML Pipeline
Provides unified interfaces for TensorBoard, Weights & Biases, and MLflow
"""

    ExperimentTracker,
    TensorBoardTracker,
    WandbTracker,
    MLflowTracker,
    CompositeTracker
)
    CheckpointManager,
    ModelCheckpoint,
    TrainingCheckpoint,
    CheckpointStrategy
)
    MetricsTracker,
    MetricLogger,
    MetricAggregator
)

# Version information
__version__ = "1.0.0"

# Module exports
__all__ = [
    # Experiment tracking
    "ExperimentTracker",
    "TensorBoardTracker", 
    "WandbTracker",
    "MLflowTracker",
    "CompositeTracker",
    
    # Checkpointing
    "CheckpointManager",
    "ModelCheckpoint",
    "TrainingCheckpoint", 
    "CheckpointStrategy",
    
    # Metrics
    "MetricsTracker",
    "MetricLogger",
    "MetricAggregator"
]

# Convenience functions
def create_tracker(config: dict) -> ExperimentTracker:
    """
    Create experiment tracker from configuration.
    
    Args:
        config: Configuration dictionary with experiment tracking settings
        
    Returns:
        Configured experiment tracker
    """
    trackers = []
    
    # TensorBoard tracker
    if config.get("tensorboard", {}).get("enabled", False):
        tb_config = config["tensorboard"]
        trackers.append(TensorBoardTracker(
            log_dir=tb_config.get("log_dir", "./logs"),
            update_freq=tb_config.get("update_freq", 100),
            flush_secs=tb_config.get("flush_secs", 120)
        ))
    
    # Weights & Biases tracker
    if config.get("wandb", {}).get("enabled", False):
        wandb_config = config["wandb"]
        trackers.append(WandbTracker(
            project=wandb_config.get("project", "key_messages"),
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config_exclude_keys=wandb_config.get("config_exclude_keys", [])
        ))
    
    # MLflow tracker
    if config.get("mlflow", {}).get("enabled", False):
        mlflow_config = config["mlflow"]
        trackers.append(MLflowTracker(
            tracking_uri=mlflow_config.get("tracking_uri", "sqlite:///mlflow.db"),
            experiment_name=mlflow_config.get("experiment_name", "key_messages"),
            log_models=mlflow_config.get("log_models", True)
        ))
    
    if not trackers:
        # Return a no-op tracker if none are enabled
        return ExperimentTracker()
    
    if len(trackers) == 1:
        return trackers[0]
    
    return CompositeTracker(trackers)

def create_checkpoint_manager(config: dict) -> CheckpointManager:
    """
    Create checkpoint manager from configuration.
    
    Args:
        config: Configuration dictionary with checkpointing settings
        
    Returns:
        Configured checkpoint manager
    """
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    save_steps = config.get("save_steps", 1000)
    save_total_limit = config.get("save_total_limit", 3)
    save_best_only = config.get("save_best_only", False)
    monitor = config.get("monitor", "loss")
    mode = config.get("mode", "min")
    
    strategy = CheckpointStrategy(
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_best_only=save_best_only,
        monitor=monitor,
        mode=mode
    )
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        strategy=strategy
    )

# Example usage
def example_usage():
    """
    Example usage of the experiment tracking system.
    """
    print("""
# Experiment Tracking Usage Examples

## 1. Basic Experiment Tracking Setup
```python

# Load configuration
config = get_config("production")
exp_config = config["experiment_tracking"]

# Create tracker and checkpoint manager
tracker = create_tracker(exp_config)
checkpoint_manager = create_checkpoint_manager(config["training"]["default"])

# Initialize experiment
tracker.init_experiment(
    experiment_name="key_messages_training",
    config=config
)

# Log metrics during training
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Training step
        loss = model(inputs, targets)
        
        # Log metrics
        tracker.log_metrics({
            "train/loss": loss.item(),
            "train/epoch": epoch,
            "train/step": global_step
        }, step=global_step)
        
        # Save checkpoint
        if checkpoint_manager.should_save_checkpoint(global_step, loss.item()):
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                metrics={"loss": loss.item()}
            )
```

## 2. Using Individual Trackers
```python

# TensorBoard only
tb_tracker = TensorBoardTracker(
    log_dir="./logs",
    update_freq=100
)

# Weights & Biases only
wandb_tracker = WandbTracker(
    project="key_messages",
    entity="your_entity",
    tags=["production", "gpt2"]
)

# Initialize experiments
tb_tracker.init_experiment("tensorboard_run", config)
wandb_tracker.init_experiment("wandb_run", config)

# Log metrics
tb_tracker.log_metrics({"loss": 0.5}, step=100)
wandb_tracker.log_metrics({"loss": 0.5}, step=100)
```

## 3. Composite Tracking
```python

# Create multiple trackers
trackers = [
    TensorBoardTracker(log_dir="./logs"),
    WandbTracker(project="key_messages")
]

# Combine into composite tracker
composite_tracker = CompositeTracker(trackers)

# Single call logs to all trackers
composite_tracker.log_metrics({"loss": 0.5}, step=100)
```

## 4. Advanced Checkpointing
```python

# Custom checkpoint strategy
strategy = CheckpointStrategy(
    save_steps=500,
    save_total_limit=5,
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

checkpoint_manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    strategy=strategy
)

# During training
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate_epoch()
    
    # Save checkpoint if conditions are met
    if checkpoint_manager.should_save_checkpoint(global_step, val_loss):
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss
            }
        )

# Load best checkpoint
best_checkpoint = checkpoint_manager.load_best_checkpoint()
model.load_state_dict(best_checkpoint.model_state)
```

## 5. Metrics Tracking
```python

# Create metrics tracker
metrics_tracker = MetricsTracker()

# Log different types of metrics
metrics_tracker.log_scalar("loss", 0.5, step=100)
metrics_tracker.log_scalars({
    "train_loss": 0.5,
    "val_loss": 0.4,
    "learning_rate": 1e-4
}, step=100)

# Log histograms
metrics_tracker.log_histogram("gradients", gradients, step=100)

# Log text
metrics_tracker.log_text("generated_text", "Sample generated message", step=100)

# Log images
metrics_tracker.log_image("attention_weights", attention_weights, step=100)

# Get aggregated metrics
avg_loss = metrics_tracker.get_average("loss", window=100)
```

## 6. Integration with Training Loop
```python

class TrainingManager:
    def __init__(self, config) -> Any:
        self.tracker = create_tracker(config["experiment_tracking"])
        self.checkpoint_manager = create_checkpoint_manager(config["training"]["default"])
        self.metrics_tracker = MetricsTracker()
        
    def train(self, model, train_loader, val_loader, num_epochs) -> Any:
        # Initialize experiment
        self.tracker.init_experiment("training_run", config)
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(model, train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(model, val_loader)
            
            # Log metrics
            self.tracker.log_metrics({
                **train_metrics,
                **val_metrics,
                "epoch": epoch
            }, step=global_step)
            
            # Save checkpoint
            if self.checkpoint_manager.should_save_checkpoint(
                global_step, val_metrics["val_loss"]
            ):
                self.checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    metrics={**train_metrics, **val_metrics}
                )
        
        # Finalize experiment
        self.tracker.finalize_experiment()
```

## 7. Configuration-Driven Setup
```yaml
# config.yaml
experiment_tracking:
  tensorboard:
    enabled: true
    log_dir: "./logs"
    update_freq: 100
    flush_secs: 120
    
  wandb:
    enabled: true
    project: "key_messages"
    entity: "your_entity"
    tags: ["production", "gpt2"]
    notes: "Production training run"
    
  mlflow:
    enabled: false
    tracking_uri: "sqlite:///mlflow.db"
    experiment_name: "key_messages"
    log_models: true

training:
  default:
    save_steps: 1000
    save_total_limit: 3
    save_best_only: true
    monitor: "val_loss"
    mode: "min"
    checkpoint_dir: "./checkpoints"
```

```python
# Usage with configuration

config = get_config("production")
tracker = create_tracker(config["experiment_tracking"])
checkpoint_manager = create_checkpoint_manager(config["training"]["default"])
```
""")

match __name__:
    case "__main__":
    example_usage() 