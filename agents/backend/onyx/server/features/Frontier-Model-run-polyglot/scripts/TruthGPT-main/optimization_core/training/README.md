# 🎯 Training Module - Professional Training Components

[![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)]()

Comprehensive training components for TruthGPT optimization core with support for mixed precision, gradient accumulation, checkpointing, EMA, and experiment tracking.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Components](#components)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## 🚀 Quick Start

### Installation

```bash
# Training components are part of optimization_core
pip install -r requirements.txt
```

### Basic Usage

```python
from training import (
    create_training_loop,
    create_checkpoint_manager,
    create_ema_manager,
    create_evaluator,
    create_experiment_tracker
)

# Create training components
training_loop = create_training_loop(
    use_amp=True,
    max_grad_norm=1.0,
    grad_accum_steps=4
)

checkpoint_manager = create_checkpoint_manager("./checkpoints")
ema_manager = create_ema_manager(decay=0.999)
evaluator = create_evaluator(use_amp=True)
tracker = create_experiment_tracker(
    trackers=["wandb", "tensorboard"],
    project="truthgpt",
    run_name="experiment-1"
)
```

## 🧩 Components

### 1. TrainingLoop

Core training loop with:
- ✅ Automatic Mixed Precision (AMP)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ DataParallel support
- ✅ Loss validation

```python
from training import TrainingLoop

loop = TrainingLoop(
    use_amp=True,
    max_grad_norm=1.0,
    grad_accum_steps=4
)

# Training step
metrics = loop.train_step(
    model=model,
    batch=batch,
    optimizer=optimizer,
    scaler=scaler,
    step=global_step
)
```

### 2. CheckpointManager

Manages training checkpoints with:
- ✅ SafeTensors support
- ✅ Model, optimizer, scheduler state
- ✅ EMA state support
- ✅ Tokenizer saving
- ✅ Automatic directory creation

```python
from training import CheckpointManager

checkpoint = CheckpointManager("./checkpoints")

# Save checkpoint
checkpoint.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    step=step,
    path="./checkpoints/step-1000",
    tokenizer=tokenizer,
    scaler=scaler,
    ema_state=ema_manager.state_dict()
)

# Load checkpoint
state = checkpoint.load_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    path="./checkpoints/step-1000"
)
```

### 3. EMAManager

Exponential Moving Average for model weights:
- ✅ Configurable decay
- ✅ DataParallel support
- ✅ State save/load
- ✅ Apply/restore operations

```python
from training import EMAManager

ema = EMAManager(decay=0.999, model=model)

# Update EMA during training
for batch in dataloader:
    # Training step...
    ema.update(model)

# Apply EMA for evaluation
ema.apply_to_model(model)
metrics = evaluate(model, val_loader)

# Restore original weights
ema.restore_from_model(model)
```

### 4. Evaluator

Model evaluation with:
- ✅ AMP support
- ✅ Comprehensive metrics
- ✅ Loss aggregation
- ✅ Device handling

```python
from training import Evaluator

evaluator = Evaluator(use_amp=True, device=device)

# Evaluate model
metrics = evaluator.evaluate(
    model=model,
    data_loader=val_loader,
    device=device
)

# Returns: {"loss": 0.5, "perplexity": 2.71, ...}
```

### 5. ExperimentTracker

Experiment tracking with:
- ✅ WandB integration
- ✅ TensorBoard support
- ✅ Multiple trackers
- ✅ Metrics logging
- ✅ Model artifacts

```python
from training import ExperimentTracker

tracker = ExperimentTracker(
    trackers=["wandb", "tensorboard"],
    project="truthgpt",
    run_name="experiment-1"
)

# Log metrics
tracker.log({
    "train/loss": 0.5,
    "train/lr": 1e-4,
    "val/loss": 0.4,
    "val/perplexity": 2.71
}, step=step)

# Log model
tracker.log_model(model, "best_model")
```

## 📖 API Reference

### Factory Functions

#### `create_training_component(component_type, config)`

Unified factory for creating training components.

```python
# Create any component
loop = create_training_component(
    "training_loop",
    {"use_amp": True, "max_grad_norm": 1.0}
)
```

#### `list_available_training_components()`

List all available component types.

```python
components = list_available_training_components()
# Returns: ['training_loop', 'checkpoint_manager', 'ema_manager', 'evaluator', 'experiment_tracker']
```

#### `get_training_component_info(component_type)`

Get information about a component.

```python
info = get_training_component_info("training_loop")
# Returns: {'name': 'training_loop', 'class': 'TrainingLoop', 'description': '...', ...}
```

### Convenience Functions

- `create_training_loop(**kwargs)` - Create training loop
- `create_checkpoint_manager(output_dir)` - Create checkpoint manager
- `create_ema_manager(decay, model)` - Create EMA manager
- `create_evaluator(**kwargs)` - Create evaluator
- `create_experiment_tracker(**kwargs)` - Create experiment tracker

## 💡 Examples

### Complete Training Setup

```python
from training import (
    create_training_loop,
    create_checkpoint_manager,
    create_ema_manager,
    create_evaluator,
    create_experiment_tracker
)
from torch.cuda.amp import GradScaler

# Initialize components
training_loop = create_training_loop(
    use_amp=True,
    max_grad_norm=1.0,
    grad_accum_steps=4
)

checkpoint = create_checkpoint_manager("./checkpoints")
ema = create_ema_manager(decay=0.999, model=model)
evaluator = create_evaluator(use_amp=True)
tracker = create_experiment_tracker(
    trackers=["wandb"],
    project="truthgpt"
)

scaler = GradScaler()

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # Training step
        metrics = training_loop.train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            step=global_step
        )
        
        # Update EMA
        ema.update(model)
        
        # Log metrics
        tracker.log(metrics, step=global_step)
        
        # Save checkpoint
        if global_step % 1000 == 0:
            checkpoint.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                path=f"./checkpoints/step-{global_step}",
                ema_state=ema.state_dict()
            )
        
        global_step += 1
    
    # Evaluation
    ema.apply_to_model(model)
    val_metrics = evaluator.evaluate(
        model=model,
        data_loader=val_loader,
        device=device
    )
    tracker.log(val_metrics, step=global_step)
    ema.restore_from_model(model)
```

### Using Factory Pattern

```python
from training import create_training_component

# Create components using factory
loop = create_training_component(
    "training_loop",
    {"use_amp": True, "grad_accum_steps": 4}
)

checkpoint = create_training_component(
    "checkpoint_manager",
    {"output_dir": "./checkpoints"}
)

# List available components
from training import list_available_training_components
components = list_available_training_components()
print(components)  # ['training_loop', 'checkpoint_manager', ...]
```

## 🎯 Best Practices

### 1. Mixed Precision Training

Always use AMP for faster training:

```python
loop = create_training_loop(use_amp=True)
scaler = GradScaler()
```

### 2. Gradient Accumulation

Use gradient accumulation for effective larger batch sizes:

```python
loop = create_training_loop(grad_accum_steps=4)
# Effective batch size = batch_size * grad_accum_steps
```

### 3. EMA for Better Models

Use EMA for more stable model weights:

```python
ema = create_ema_manager(decay=0.999, model=model)

# Update after each step
ema.update(model)

# Use EMA weights for evaluation
ema.apply_to_model(model)
```

### 4. Regular Checkpointing

Save checkpoints regularly:

```python
if step % 1000 == 0:
    checkpoint.save_checkpoint(...)
```

### 5. Experiment Tracking

Track all experiments:

```python
tracker = create_experiment_tracker(
    trackers=["wandb", "tensorboard"],
    project="truthgpt"
)
```

## 📊 Component Registry

All components are registered in `TRAINING_COMPONENT_REGISTRY`:

```python
from training import TRAINING_COMPONENT_REGISTRY

for name, info in TRAINING_COMPONENT_REGISTRY.items():
    print(f"{name}: {info['description']}")
```

## 🔧 Configuration

### TrainingLoop Config

```python
{
    "use_amp": True,           # Enable AMP
    "max_grad_norm": 1.0,      # Gradient clipping
    "grad_accum_steps": 4,     # Gradient accumulation
}
```

### CheckpointManager Config

```python
{
    "output_dir": "./checkpoints",  # Output directory
}
```

### EMAManager Config

```python
{
    "decay": 0.999,  # EMA decay factor
}
```

### Evaluator Config

```python
{
    "use_amp": True,  # Enable AMP
    "device": device, # Evaluation device
}
```

### ExperimentTracker Config

```python
{
    "trackers": ["wandb", "tensorboard"],  # Trackers to use
    "project": "truthgpt",                  # Project name
    "run_name": "experiment-1",             # Run name
    "log_dir": "./logs",                    # Log directory
}
```

## 📚 Additional Resources

- [Training Best Practices](../../docs/TRAINING_BEST_PRACTICES.md)
- [Checkpoint Management Guide](../../docs/CHECKPOINT_GUIDE.md)
- [Experiment Tracking Guide](../../docs/EXPERIMENT_TRACKING.md)

---

**Version:** 2.0.0  
**Status:** ✅ Production Ready









