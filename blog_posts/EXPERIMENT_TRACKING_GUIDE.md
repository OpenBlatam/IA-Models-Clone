# Experiment Tracking and Model Checkpointing Guide

## Overview

This guide covers the comprehensive experiment tracking and model checkpointing system designed for deep learning projects. The system provides:

- **Multi-backend experiment tracking** (WandB, TensorBoard, MLflow)
- **Robust model checkpointing** with metadata management
- **Performance monitoring** and visualization
- **Reproducibility** and versioning
- **Integration** with modular architecture

## Key Components

### 1. Experiment Metadata Management

The `ExperimentMetadata` class captures comprehensive information about experiments:

```python
from experiment_tracking import ExperimentMetadata

metadata = ExperimentMetadata(
    experiment_name="transformer_classification",
    experiment_id="20231201_143022_abc123",
    description="Fine-tuning BERT for text classification",
    tags=["nlp", "transformer", "classification"],
    version="1.0.0"
)
```

**Key Features:**
- Automatic timestamp generation
- System information capture (Python, PyTorch, CUDA versions)
- Status tracking (running, completed, failed)
- Configuration hash for reproducibility

### 2. Model Checkpointing System

The `ModelCheckpointer` provides robust checkpoint management:

```python
from experiment_tracking import ModelCheckpointer

checkpointer = ModelCheckpointer(
    checkpoint_dir="checkpoints",
    max_checkpoints=10
)

# Save checkpoint
checkpoint_path = checkpointer.save_checkpoint(
    experiment_id="exp_123",
    model=model,
    epoch=5,
    step=1000,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loss=0.25,
    val_loss=0.30,
    train_accuracy=0.92,
    val_accuracy=0.89,
    is_best=True
)

# Load checkpoint
metadata, epoch, step = checkpointer.load_checkpoint(
    checkpoint_path, model, optimizer, scheduler
)
```

**Features:**
- Comprehensive metadata storage
- Automatic cleanup of old checkpoints
- Best model tracking
- Registry management
- File size tracking

### 3. Multi-Backend Tracking

Support for multiple experiment tracking backends:

#### Weights & Biases (WandB)

```python
config = {
    "tracking": {
        "use_wandb": True,
        "wandb_project": "deep-learning-project"
    }
}

with experiment_tracking("my_experiment", config) as tracker:
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
    tracker.log_hyperparameters({"lr": 1e-4, "batch_size": 32})
    tracker.log_model("model.pt", "best_model")
```

#### TensorBoard

```python
config = {
    "tracking": {
        "use_tensorboard": True,
        "tensorboard_log_dir": "logs/tensorboard"
    }
}

# TensorBoard automatically logs:
# - Metrics over time
# - Hyperparameters
# - Model graphs
# - Images and text
```

#### MLflow

```python
config = {
    "tracking": {
        "use_mlflow": True,
        "mlflow_tracking_uri": "file:./mlruns"
    }
}

# MLflow provides:
# - Experiment management
# - Model versioning
# - Artifact storage
# - Model registry
```

### 4. Experiment Tracker

The main `ExperimentTracker` class orchestrates all tracking and checkpointing:

```python
from experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker("transformer_experiment", config)

# Log metrics
tracker.log_metrics({
    "train_loss": 0.25,
    "val_loss": 0.30,
    "train_accuracy": 0.92,
    "val_accuracy": 0.89
}, step=100)

# Save checkpoint
checkpoint_path = tracker.save_checkpoint(
    model=model,
    epoch=5,
    step=1000,
    optimizer=optimizer,
    train_loss=0.25,
    val_loss=0.30,
    is_best=(val_loss < best_val_loss)
)

# Load best checkpoint
best_checkpoint = tracker.get_best_checkpoint()
if best_checkpoint:
    epoch, step = tracker.load_checkpoint(best_checkpoint, model, optimizer)

# Create performance plots
tracker.create_performance_plots("plots/")

# Finish experiment
tracker.finish("completed")
```

## Configuration

### Basic Configuration

```yaml
# configs/experiment_tracking.yaml
description: "Transformer fine-tuning experiment"
tags: ["nlp", "transformer", "classification"]

tracking:
  use_wandb: true
  use_tensorboard: true
  use_mlflow: false
  wandb_project: "deep-learning-project"
  tensorboard_log_dir: "logs/tensorboard"
  mlflow_tracking_uri: "file:./mlruns"

checkpoint_dir: "checkpoints"
max_checkpoints: 10

model:
  type: "transformer"
  name: "bert-base-uncased"

training:
  epochs: 10
  learning_rate: 2e-5
  batch_size: 16
```

### Advanced Configuration

```yaml
# Advanced tracking configuration
tracking:
  use_wandb: true
  wandb_project: "production-models"
  wandb_entity: "my-team"
  wandb_tags: ["production", "v2.0"]
  
  use_tensorboard: true
  tensorboard_log_dir: "logs/tensorboard"
  tensorboard_flush_secs: 30
  
  use_mlflow: true
  mlflow_tracking_uri: "sqlite:///mlflow.db"
  mlflow_registry_uri: "sqlite:///mlflow.db"

# Checkpointing configuration
checkpointing:
  checkpoint_dir: "checkpoints"
  max_checkpoints: 20
  save_frequency: 100  # Save every 100 steps
  save_best_only: false
  compression: "gzip"  # Compress checkpoints
  
# Performance monitoring
monitoring:
  log_gpu_usage: true
  log_memory_usage: true
  log_learning_rate: true
  create_plots: true
  plot_frequency: 1000
```

## Usage Patterns

### 1. Training Loop Integration

```python
from experiment_tracking import experiment_tracking

config = load_config("configs/experiment_tracking.yaml")

with experiment_tracking("transformer_training", config) as tracker:
    # Log hyperparameters
    tracker.log_hyperparameters({
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"]
    })
    
    best_val_loss = float('inf')
    
    for epoch in range(config["training"]["epochs"]):
        for step, (batch, labels) in enumerate(train_loader):
            # Training step
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if step % 10 == 0:
                tracker.log_metrics({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch * len(train_loader) + step)
            
            # Save checkpoint
            if step % 100 == 0:
                val_loss = validate_model(model, val_loader, criterion)
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                tracker.save_checkpoint(
                    model=model,
                    epoch=epoch,
                    step=epoch * len(train_loader) + step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    is_best=is_best
                )
```

### 2. Model Evaluation and Comparison

```python
def compare_experiments(experiment_ids):
    """Compare multiple experiments."""
    results = {}
    
    for exp_id in experiment_ids:
        # Load experiment metadata
        metadata_path = f"experiments/{exp_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load best checkpoint
        checkpointer = ModelCheckpointer("checkpoints")
        best_checkpoint = checkpointer.get_best_checkpoint(exp_id)
        
        if best_checkpoint:
            # Load model and evaluate
            model = create_model(metadata["model_config"])
            checkpoint_info = checkpointer.load_checkpoint(best_checkpoint, model)
            
            results[exp_id] = {
                "best_val_loss": checkpoint_info[0].val_loss,
                "best_val_accuracy": checkpoint_info[0].val_accuracy,
                "training_time": metadata["end_time"] - metadata["start_time"]
            }
    
    return results
```

### 3. Reproducibility and Versioning

```python
def create_reproducible_experiment(config, seed=42):
    """Create reproducible experiment with versioning."""
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create experiment with versioning
    experiment_name = f"{config['model']['name']}_v{config['version']}"
    
    with experiment_tracking(experiment_name, config) as tracker:
        # Log configuration hash for reproducibility
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        
        tracker.log_hyperparameters({
            "config_hash": config_hash,
            "random_seed": seed,
            "git_commit": get_git_commit_hash()
        })
        
        # Training loop...
```

## Performance Optimization

### 1. Efficient Checkpointing

```python
# Use compression for large models
checkpointer = ModelCheckpointer(
    checkpoint_dir="checkpoints",
    max_checkpoints=10,
    compression="gzip"
)

# Save only essential components
checkpoint_data = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "step": step,
    "metrics": {"val_loss": val_loss}
}
```

### 2. Asynchronous Logging

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncExperimentTracker(ExperimentTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def log_metrics_async(self, metrics, step=None):
        """Log metrics asynchronously."""
        self.executor.submit(self.log_metrics, metrics, step)
```

### 3. Batch Logging

```python
class BatchExperimentTracker(ExperimentTracker):
    def __init__(self, *args, batch_size=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.metrics_buffer = []
    
    def log_metrics(self, metrics, step=None):
        """Buffer metrics and log in batches."""
        self.metrics_buffer.append((metrics, step))
        
        if len(self.metrics_buffer) >= self.batch_size:
            self._flush_metrics_buffer()
    
    def _flush_metrics_buffer(self):
        """Flush buffered metrics to all trackers."""
        for metrics, step in self.metrics_buffer:
            super().log_metrics(metrics, step)
        self.metrics_buffer.clear()
```

## Monitoring and Visualization

### 1. Performance Plots

The system automatically creates performance visualizations:

```python
# Create custom plots
tracker.create_performance_plots("plots/")

# Available plots:
# - Training/validation loss curves
# - Accuracy curves
# - Learning curves (log scale)
# - Overfitting indicators
```

### 2. Real-time Monitoring

```python
# Monitor GPU usage
if torch.cuda.is_available():
    gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    tracker.log_metrics({"gpu_memory_usage": gpu_usage}, step=step)

# Monitor training progress
progress = (epoch * len(train_loader) + step) / (total_epochs * len(train_loader))
tracker.log_metrics({"training_progress": progress}, step=step)
```

### 3. Custom Visualizations

```python
def create_custom_plots(tracker, model, data_loader):
    """Create custom model-specific visualizations."""
    # Attention weights visualization
    attention_weights = get_attention_weights(model, data_loader)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title("Attention Weights")
    plt.savefig("attention_weights.png")
    plt.close()
    
    tracker.log_image("attention_weights.png", "attention_weights")
```

## Best Practices

### 1. Experiment Organization

```python
# Use descriptive experiment names
experiment_name = f"{model_name}_{task}_{date}_{version}"

# Use tags for organization
tags = ["production", "v2.0", "transformer", "classification"]

# Use consistent naming conventions
checkpoint_dir = f"checkpoints/{experiment_name}"
log_dir = f"logs/{experiment_name}"
```

### 2. Checkpoint Management

```python
# Save checkpoints at appropriate intervals
save_frequency = 100  # Save every 100 steps
eval_frequency = 500  # Evaluate every 500 steps

# Keep only essential checkpoints
max_checkpoints = 5  # Keep last 5 checkpoints + best

# Use meaningful checkpoint names
checkpoint_name = f"epoch_{epoch}_step_{step}_loss_{val_loss:.4f}.pt"
```

### 3. Error Handling

```python
try:
    with experiment_tracking("my_experiment", config) as tracker:
        # Training loop
        pass
except Exception as e:
    # Log error and mark experiment as failed
    logger.error(f"Experiment failed: {e}")
    # Experiment will be automatically marked as failed
```

### 4. Resource Management

```python
# Monitor resource usage
import psutil
import GPUtil

def log_system_metrics(tracker, step):
    """Log system resource usage."""
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    
    tracker.log_metrics({
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage
    }, step=step)
    
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        tracker.log_metrics({
            "gpu_utilization": gpu.load * 100,
            "gpu_memory_used": gpu.memoryUsed,
            "gpu_memory_total": gpu.memoryTotal
        }, step=step)
```

## Integration with Modular Architecture

The experiment tracking system integrates seamlessly with the modular architecture:

```python
from modular_architecture import ExperimentRunner
from experiment_tracking import experiment_tracking

class TrackedExperimentRunner(ExperimentRunner):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.config = load_config(config_path)
    
    def run_experiment(self):
        """Run experiment with tracking."""
        with experiment_tracking(
            self.config["experiment"]["name"],
            self.config
        ) as tracker:
            # Initialize components
            model = self.model_factory.create_model()
            optimizer = self.optimizer_factory.create_optimizer(model)
            scheduler = self.scheduler_factory.create_scheduler(optimizer)
            
            # Log hyperparameters
            tracker.log_hyperparameters(self.config["training"])
            
            # Training loop with tracking
            for epoch in range(self.config["training"]["epochs"]):
                for step, batch in enumerate(self.train_loader):
                    # Training step
                    loss = self.train_step(model, optimizer, batch)
                    
                    # Log metrics
                    if step % 10 == 0:
                        tracker.log_metrics({"train_loss": loss}, step=step)
                    
                    # Save checkpoint
                    if step % 100 == 0:
                        val_loss = self.evaluate(model, self.val_loader)
                        tracker.save_checkpoint(
                            model=model,
                            epoch=epoch,
                            step=step,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            train_loss=loss,
                            val_loss=val_loss
                        )
```

## Troubleshooting

### Common Issues

1. **WandB Authentication**
   ```bash
   wandb login
   ```

2. **TensorBoard Not Showing Data**
   ```bash
   tensorboard --logdir=logs/tensorboard --port=6006
   ```

3. **Checkpoint Loading Errors**
   ```python
   # Check checkpoint compatibility
   checkpoint = torch.load("checkpoint.pt", map_location="cpu")
   print(checkpoint.keys())
   ```

4. **Memory Issues**
   ```python
   # Use gradient checkpointing for large models
   model.gradient_checkpointing_enable()
   
   # Use mixed precision training
   scaler = torch.cuda.amp.GradScaler()
   ```

### Performance Tips

1. **Use async logging for high-frequency metrics**
2. **Batch checkpoint saves for large models**
3. **Compress checkpoints to save disk space**
4. **Use appropriate save frequencies**
5. **Monitor disk usage regularly**

## Conclusion

The experiment tracking and model checkpointing system provides a comprehensive solution for managing deep learning experiments. It ensures reproducibility, enables performance monitoring, and facilitates model comparison and selection.

Key benefits:
- **Reproducibility**: Complete experiment tracking with configuration versioning
- **Performance**: Efficient checkpointing and logging
- **Flexibility**: Support for multiple tracking backends
- **Integration**: Seamless integration with modular architecture
- **Monitoring**: Real-time performance tracking and visualization

This system is essential for production-grade deep learning projects and research workflows. 