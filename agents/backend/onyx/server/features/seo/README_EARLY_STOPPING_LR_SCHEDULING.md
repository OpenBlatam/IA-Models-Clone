# Early Stopping and Learning Rate Scheduling Framework

A comprehensive framework for implementing advanced early stopping strategies and learning rate scheduling algorithms in deep learning training pipelines, specifically optimized for SEO tasks.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Early Stopping](#early-stopping)
6. [Learning Rate Scheduling](#learning-rate-scheduling)
7. [Training Monitor](#training-monitor)
8. [Training Optimizer](#training-optimizer)
9. [Best Practices](#best-practices)
10. [Integration](#integration)
11. [Examples](#examples)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)

## Overview

This framework provides robust early stopping and learning rate scheduling capabilities for deep learning models, with special optimizations for SEO-related tasks. It includes multiple strategies, comprehensive monitoring, and seamless integration with existing training pipelines.

### Key Benefits

- **Multiple Early Stopping Strategies**: Basic patience-based, adaptive patience, plateau detection, overfitting detection
- **Comprehensive LR Scheduling**: Step, cosine, plateau, exponential, multi-step, OneCycle, warmup cosine
- **Advanced Monitoring**: Real-time metrics tracking, visualization, logging
- **Easy Integration**: Works with any PyTorch model and optimizer
- **SEO Optimized**: Tailored for SEO-specific training scenarios

## Features

### Early Stopping Features

- ✅ **Basic Early Stopping**: Patience-based stopping with configurable thresholds
- ✅ **Adaptive Patience**: Dynamic patience adjustment based on training progress
- ✅ **Multiple Metric Monitoring**: Monitor multiple metrics simultaneously
- ✅ **Plateau Detection**: Automatic detection of training plateaus
- ✅ **Overfitting Detection**: Monitor train-validation gaps
- ✅ **Best Weight Restoration**: Automatically restore best model weights
- ✅ **Checkpoint Saving**: Save best model checkpoints
- ✅ **Comprehensive Logging**: Detailed training history and metrics

### Learning Rate Scheduling Features

- ✅ **Step LR**: Fixed step size learning rate decay
- ✅ **Cosine Annealing**: Smooth cosine-based learning rate decay
- ✅ **Cosine Warm Restarts**: Cosine annealing with warm restarts
- ✅ **Reduce on Plateau**: Reduce LR when validation metric plateaus
- ✅ **Exponential Decay**: Exponential learning rate decay
- ✅ **Multi-Step LR**: Step decay at specific milestones
- ✅ **OneCycle**: Fast training with one-cycle policy
- ✅ **Warmup Cosine**: Cosine decay with warmup phase
- ✅ **Custom Functions**: Support for custom LR functions
- ✅ **Real-time Monitoring**: Track LR changes during training

### Training Monitor Features

- ✅ **Unified Interface**: Single interface for early stopping and LR scheduling
- ✅ **Comprehensive Metrics**: Track all training metrics
- ✅ **Visualization**: Plot training curves and LR schedules
- ✅ **Logging**: Save training logs and summaries
- ✅ **Integration**: Seamless integration with training frameworks

## Installation

```bash
# Install required dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn

# For GPU support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

```python
import torch
import torch.nn as nn
import torch.optim as optim
from early_stopping_lr_scheduling import (
    EarlyStoppingConfig, LRSchedulerConfig, TrainingOptimizer
)

# Create model and optimizer
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Configure early stopping
early_stopping_config = EarlyStoppingConfig(
    patience=10,
    monitor="val_loss",
    mode="min",
    restore_best_weights=True
)

# Configure LR scheduling
lr_scheduler_config = LRSchedulerConfig(
    scheduler_type="cosine",
    initial_lr=1e-3,
    T_max=100
)

# Create training optimizer
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train with early stopping and LR scheduling
summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=100)
```

## Early Stopping

### Basic Configuration

```python
from early_stopping_lr_scheduling import EarlyStoppingConfig

# Basic early stopping
config = EarlyStoppingConfig(
    enabled=True,
    patience=10,              # Number of epochs to wait
    min_delta=1e-4,          # Minimum improvement required
    mode="min",              # "min" for loss, "max" for accuracy
    monitor="val_loss",      # Metric to monitor
    restore_best_weights=True,
    verbose=True
)
```

### Advanced Configuration

```python
# Advanced early stopping with multiple strategies
config = EarlyStoppingConfig(
    # Basic settings
    patience=15,
    min_delta=1e-4,
    mode="min",
    monitor="val_loss",
    
    # Multiple metric monitoring
    monitor_multiple=True,
    monitors=["val_loss", "val_accuracy"],
    monitor_weights=[1.0, 0.5],
    
    # Adaptive patience
    adaptive_patience=True,
    patience_factor=1.2,
    min_patience=5,
    max_patience=30,
    
    # Plateau detection
    plateau_detection=True,
    plateau_window=5,
    plateau_threshold=1e-3,
    
    # Overfitting detection
    overfitting_detection=True,
    train_val_gap_threshold=0.15,
    overfitting_patience=8,
    
    # Checkpointing
    save_checkpoint=True,
    checkpoint_path="./checkpoints/best_model.pth"
)
```

### Early Stopping Strategies

#### 1. Basic Patience-Based Stopping

```python
# Simple patience-based early stopping
config = EarlyStoppingConfig(
    patience=10,
    monitor="val_loss",
    mode="min"
)
```

#### 2. Adaptive Patience

```python
# Adaptive patience that increases if model is improving slowly
config = EarlyStoppingConfig(
    adaptive_patience=True,
    patience_factor=1.5,
    min_patience=5,
    max_patience=50
)
```

#### 3. Multiple Metric Monitoring

```python
# Monitor multiple metrics with different weights
config = EarlyStoppingConfig(
    monitor_multiple=True,
    monitors=["val_loss", "val_accuracy", "val_f1"],
    monitor_weights=[1.0, 0.5, 0.3]
)
```

#### 4. Plateau Detection

```python
# Detect when training has plateaued
config = EarlyStoppingConfig(
    plateau_detection=True,
    plateau_window=5,
    plateau_threshold=1e-3
)
```

#### 5. Overfitting Detection

```python
# Detect overfitting based on train-validation gap
config = EarlyStoppingConfig(
    overfitting_detection=True,
    train_val_gap_threshold=0.15,
    overfitting_patience=8
)
```

## Learning Rate Scheduling

### Basic Configuration

```python
from early_stopping_lr_scheduling import LRSchedulerConfig

# Basic cosine annealing
config = LRSchedulerConfig(
    scheduler_type="cosine",
    initial_lr=1e-3,
    T_max=100,
    eta_min=1e-6
)
```

### Available Schedulers

#### 1. Step LR

```python
config = LRSchedulerConfig(
    scheduler_type="step",
    initial_lr=1e-3,
    step_size=30,
    gamma=0.1
)
```

#### 2. Cosine Annealing

```python
config = LRSchedulerConfig(
    scheduler_type="cosine",
    initial_lr=1e-3,
    T_max=100,
    eta_min=1e-6
)
```

#### 3. Cosine Warm Restarts

```python
config = LRSchedulerConfig(
    scheduler_type="cosine_warm_restarts",
    initial_lr=1e-3,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)
```

#### 4. Reduce on Plateau

```python
config = LRSchedulerConfig(
    scheduler_type="plateau",
    initial_lr=1e-3,
    min_lr=1e-6,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4
)
```

#### 5. Exponential Decay

```python
config = LRSchedulerConfig(
    scheduler_type="exponential",
    initial_lr=1e-3,
    decay_rate=0.95
)
```

#### 6. Multi-Step LR

```python
config = LRSchedulerConfig(
    scheduler_type="multistep",
    initial_lr=1e-3,
    milestones=[30, 60, 90],
    gamma=0.5
)
```

#### 7. OneCycle

```python
config = LRSchedulerConfig(
    scheduler_type="onecycle",
    initial_lr=1e-3,
    max_lr=1e-2,
    epochs=100,
    steps_per_epoch=100,
    pct_start=0.3,
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,
    final_div_factor=1e4
)
```

#### 8. Warmup Cosine

```python
config = LRSchedulerConfig(
    scheduler_type="warmup_cosine",
    initial_lr=1e-3,
    warmup_steps=1000,
    warmup_start_lr=1e-6,
    T_max=10000,
    eta_min=1e-6
)
```

#### 9. Custom LR Function

```python
def custom_lr_fn(epoch):
    """Custom learning rate function"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return 1e-6 + (1e-3 - 1e-6) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (30 - warmup_epochs)
        return 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + np.cos(np.pi * progress))

config = LRSchedulerConfig(
    scheduler_type="custom",
    custom_lr_fn=custom_lr_fn
)
```

## Training Monitor

The `TrainingMonitor` class provides a unified interface for managing early stopping and learning rate scheduling during training.

### Basic Usage

```python
from early_stopping_lr_scheduling import TrainingMonitor

# Create monitor
monitor = TrainingMonitor(
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config,
    optimizer=optimizer,
    model=model
)

# During training
for epoch in range(max_epochs):
    # Train one epoch
    metrics = train_epoch(train_loader, val_loader, criterion, device)
    
    # Update monitor
    should_stop = monitor.update(metrics)
    
    if should_stop:
        print("Early stopping triggered")
        break

# Get training summary
summary = monitor.get_training_summary()
```

### Visualization

```python
# Plot training curves
monitor.plot_training_curves(save_path="training_curves.png")

# Save training log
monitor.save_training_log("training_log.json")
```

## Training Optimizer

The `TrainingOptimizer` class provides a high-level interface for training with early stopping and learning rate scheduling.

### Basic Usage

```python
from early_stopping_lr_scheduling import TrainingOptimizer

# Create training optimizer
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train
summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=100)
```

### Advanced Usage

```python
# Custom training loop with monitoring
trainer = TrainingOptimizer(model, optimizer, early_stopping_config, lr_scheduler_config)

for epoch in range(max_epochs):
    # Train one epoch
    metrics = trainer.train_epoch(train_loader, val_loader, criterion, device)
    
    # Update monitor
    should_stop = trainer.monitor.update(metrics)
    
    # Custom logging
    print(f"Epoch {epoch}: Loss={metrics.train_loss:.4f}, Val Loss={metrics.val_loss:.4f}")
    
    if should_stop:
        break

# Restore best weights
trainer.monitor.early_stopping.restore_best_weights(model)
```

## Best Practices

### Early Stopping Best Practices

1. **Choose Appropriate Patience**
   ```python
   # For fast convergence
   patience = 5-10
   
   # For stable training
   patience = 15-25
   
   # For complex models
   patience = 30-50
   ```

2. **Monitor the Right Metric**
   ```python
   # For classification
   monitor = "val_accuracy"  # or "val_f1"
   
   # For regression
   monitor = "val_loss"
   
   # For multi-task
   monitor_multiple = True
   monitors = ["val_loss", "val_accuracy"]
   ```

3. **Use Adaptive Patience for Complex Models**
   ```python
   config = EarlyStoppingConfig(
       adaptive_patience=True,
       patience_factor=1.2,
       min_patience=5,
       max_patience=50
   )
   ```

4. **Enable Overfitting Detection**
   ```python
   config = EarlyStoppingConfig(
       overfitting_detection=True,
       train_val_gap_threshold=0.15
   )
   ```

### Learning Rate Scheduling Best Practices

1. **Choose Scheduler Based on Task**
   ```python
   # For stable training
   scheduler_type = "cosine"
   
   # For fast training
   scheduler_type = "onecycle"
   
   # For plateau detection
   scheduler_type = "plateau"
   
   # For custom schedules
   scheduler_type = "custom"
   ```

2. **Set Appropriate Learning Rate Range**
   ```python
   config = LRSchedulerConfig(
       initial_lr=1e-3,
       min_lr=1e-6,
       max_lr=1e-2
   )
   ```

3. **Use Warmup for Large Models**
   ```python
   config = LRSchedulerConfig(
       scheduler_type="warmup_cosine",
       warmup_steps=1000,
       warmup_start_lr=1e-6
   )
   ```

4. **Monitor Learning Rate Changes**
   ```python
   config = LRSchedulerConfig(
       verbose=True,
       log_interval=10
   )
   ```

### SEO-Specific Best Practices

1. **Use Multiple Metrics for SEO Tasks**
   ```python
   config = EarlyStoppingConfig(
       monitor_multiple=True,
       monitors=["val_loss", "val_accuracy", "val_ranking_score"],
       monitor_weights=[1.0, 0.3, 0.7]
   )
   ```

2. **Adaptive Patience for SEO Models**
   ```python
   config = EarlyStoppingConfig(
       adaptive_patience=True,
       patience_factor=1.5,  # More patience for SEO tasks
       max_patience=100
   )
   ```

3. **Cosine Scheduling for SEO Training**
   ```python
   config = LRSchedulerConfig(
       scheduler_type="cosine",
       T_max=200,  # Longer training for SEO
       eta_min=1e-7  # Lower minimum LR
   )
   ```

## Integration

### Integration with Training Framework

```python
from model_training_evaluation import TrainingConfig, ModelTrainer

# Create training configuration with early stopping and LR scheduling
training_config = TrainingConfig(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    optimizer="adamw",
    scheduler="cosine",
    early_stopping_patience=15,
    early_stopping_min_delta=1e-4
)

# Train with integrated early stopping and LR scheduling
trainer = ModelTrainer(training_config)
metrics = trainer.train()
```

### Integration with Custom Training Loops

```python
# Custom training loop
trainer = TrainingOptimizer(model, optimizer, early_stopping_config, lr_scheduler_config)

for epoch in range(max_epochs):
    # Custom training logic
    train_loss = train_epoch()
    val_loss = validate_epoch()
    
    # Create metrics
    metrics = TrainingMetrics(
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        train_accuracy=train_acc,
        val_accuracy=val_acc,
        learning_rate=trainer.monitor.lr_scheduler.get_lr()
    )
    
    # Update monitor
    should_stop = trainer.monitor.update(metrics)
    
    if should_stop:
        break
```

### Integration with Data Splitting Framework

```python
from data_splitting_cross_validation import DataSplitter

# Create data splits
splitter = DataSplitter()
train_data, val_data, test_data = splitter.split_data(dataset)

# Create training optimizer
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train with proper splits
summary = trainer.train(train_loader, val_loader, criterion, device)
```

## Examples

### Example 1: Basic Training with Early Stopping

```python
import torch
import torch.nn as nn
import torch.optim as optim
from early_stopping_lr_scheduling import (
    EarlyStoppingConfig, LRSchedulerConfig, TrainingOptimizer
)

# Create model
model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 3)
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Configure early stopping
early_stopping_config = EarlyStoppingConfig(
    patience=10,
    monitor="val_loss",
    mode="min",
    restore_best_weights=True
)

# Configure LR scheduling
lr_scheduler_config = LRSchedulerConfig(
    scheduler_type="cosine",
    initial_lr=1e-3,
    T_max=100
)

# Create trainer
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train
summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=100)
```

### Example 2: Advanced Training with Multiple Strategies

```python
# Advanced early stopping configuration
early_stopping_config = EarlyStoppingConfig(
    patience=15,
    min_delta=1e-4,
    mode="min",
    monitor="val_loss",
    restore_best_weights=True,
    save_checkpoint=True,
    checkpoint_path="./checkpoints/best_model.pth",
    
    # Multiple metric monitoring
    monitor_multiple=True,
    monitors=["val_loss", "val_accuracy"],
    monitor_weights=[1.0, 0.5],
    
    # Adaptive patience
    adaptive_patience=True,
    patience_factor=1.2,
    min_patience=5,
    max_patience=30,
    
    # Plateau detection
    plateau_detection=True,
    plateau_window=5,
    plateau_threshold=1e-3,
    
    # Overfitting detection
    overfitting_detection=True,
    train_val_gap_threshold=0.15,
    overfitting_patience=8,
    
    verbose=True
)

# Advanced LR scheduling
lr_scheduler_config = LRSchedulerConfig(
    scheduler_type="warmup_cosine",
    initial_lr=1e-3,
    warmup_steps=1000,
    warmup_start_lr=1e-6,
    T_max=10000,
    eta_min=1e-6,
    verbose=True
)

# Create trainer
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

# Train
summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=100)

# Plot results
trainer.monitor.plot_training_curves("advanced_training.png")
trainer.monitor.save_training_log("advanced_training.json")
```

### Example 3: OneCycle Training for Fast Convergence

```python
# OneCycle configuration for fast training
lr_scheduler_config = LRSchedulerConfig(
    scheduler_type="onecycle",
    initial_lr=1e-3,
    max_lr=1e-2,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,
    final_div_factor=1e4
)

# Conservative early stopping for OneCycle
early_stopping_config = EarlyStoppingConfig(
    patience=20,  # More patience for OneCycle
    monitor="val_loss",
    mode="min",
    restore_best_weights=True
)

# Train with OneCycle
trainer = TrainingOptimizer(
    model=model,
    optimizer=optimizer,
    early_stopping_config=early_stopping_config,
    lr_scheduler_config=lr_scheduler_config
)

summary = trainer.train(train_loader, val_loader, criterion, device, max_epochs=50)
```

## API Reference

### EarlyStoppingConfig

```python
@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "min"
    monitor: str = "val_loss"
    restore_best_weights: bool = True
    save_checkpoint: bool = True
    checkpoint_path: str = "./checkpoints/best_model.pth"
    monitor_multiple: bool = False
    monitors: List[str] = field(default_factory=lambda: ["val_loss", "val_accuracy"])
    monitor_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    adaptive_patience: bool = False
    patience_factor: float = 1.5
    min_patience: int = 5
    max_patience: int = 50
    plateau_detection: bool = False
    plateau_window: int = 5
    plateau_threshold: float = 1e-3
    overfitting_detection: bool = False
    train_val_gap_threshold: float = 0.1
    overfitting_patience: int = 5
    verbose: bool = True
    log_interval: int = 1
```

### LRSchedulerConfig

```python
@dataclass
class LRSchedulerConfig:
    scheduler_type: str = "cosine"
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 0.0
    T_0: int = 10
    T_mult: int = 2
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    decay_rate: float = 0.95
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    warmup_steps: int = 1000
    warmup_start_lr: float = 1e-6
    custom_lr_fn: Optional[Callable] = None
    verbose: bool = True
    log_interval: int = 10
```

### TrainingMetrics

```python
@dataclass
class TrainingMetrics:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
```

### TrainingOptimizer

```python
class TrainingOptimizer:
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 early_stopping_config: Optional[EarlyStoppingConfig] = None,
                 lr_scheduler_config: Optional[LRSchedulerConfig] = None)
    
    def train_epoch(self, train_loader, val_loader, criterion, device) -> TrainingMetrics
    def train(self, train_loader, val_loader, criterion, device, max_epochs: int = 100) -> Dict[str, Any]
```

## Troubleshooting

### Common Issues

1. **Early Stopping Not Triggering**
   ```python
   # Check patience value
   config = EarlyStoppingConfig(patience=5)  # Too low
   
   # Check min_delta value
   config = EarlyStoppingConfig(min_delta=1e-2)  # Too high
   
   # Check monitor metric
   config = EarlyStoppingConfig(monitor="val_loss")  # Ensure metric exists
   ```

2. **Learning Rate Not Changing**
   ```python
   # Check scheduler type
   config = LRSchedulerConfig(scheduler_type="cosine")  # Ensure valid type
   
   # Check T_max for cosine
   config = LRSchedulerConfig(T_max=100)  # Should match total epochs
   
   # Check step() calls
   scheduler.step()  # Ensure called after each epoch
   ```

3. **Training Diverging**
   ```python
   # Reduce learning rate
   config = LRSchedulerConfig(initial_lr=1e-4)  # Lower initial LR
   
   # Use gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   
   # Use warmup
   config = LRSchedulerConfig(
       scheduler_type="warmup_cosine",
       warmup_steps=1000
   )
   ```

4. **Overfitting**
   ```python
   # Enable overfitting detection
   config = EarlyStoppingConfig(
       overfitting_detection=True,
       train_val_gap_threshold=0.1
   )
   
   # Use plateau detection
   config = EarlyStoppingConfig(
       plateau_detection=True,
       plateau_window=5
   )
   ```

### Performance Optimization

1. **GPU Memory Issues**
   ```python
   # Use gradient accumulation
   accumulation_steps = 4
   loss = loss / accumulation_steps
   loss.backward()
   
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

2. **Training Speed**
   ```python
   # Use OneCycle scheduler
   config = LRSchedulerConfig(scheduler_type="onecycle")
   
   # Use mixed precision
   scaler = GradScaler()
   with autocast():
       output = model(data)
       loss = criterion(output, target)
   ```

3. **Monitoring Overhead**
   ```python
   # Reduce logging frequency
   config = EarlyStoppingConfig(log_interval=5)
   config = LRSchedulerConfig(log_interval=10)
   ```

## Conclusion

This early stopping and learning rate scheduling framework provides comprehensive tools for optimizing deep learning training, with special considerations for SEO tasks. The framework is designed to be flexible, easy to use, and highly configurable while maintaining excellent performance and monitoring capabilities.

For more advanced usage and integration examples, refer to the example scripts and the main training framework documentation. 