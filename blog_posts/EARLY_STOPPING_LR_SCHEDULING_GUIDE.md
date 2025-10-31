# 🚀 Early Stopping & Learning Rate Scheduling Guide

## Overview

This guide covers the production-ready early stopping and learning rate scheduling system for Blatam Academy's AI training pipeline. The system provides enterprise-grade functionality for intelligent training termination and dynamic learning rate adjustment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Early Stopping Strategies](#early-stopping-strategies)
4. [Learning Rate Scheduling](#learning-rate-scheduling)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

## Quick Start

### Basic Early Stopping

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling import (
    create_training_monitor, create_early_stopping_config, EarlyStoppingStrategy, EarlyStoppingMode
)
from agents.backend.onyx.server.features.blog_posts.production_transformers import DeviceManager

async def basic_early_stopping_example():
    # Create training monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup early stopping
    es_config = create_early_stopping_config(
        enabled=True,
        strategy=EarlyStoppingStrategy.PATIENCE,
        mode=EarlyStoppingMode.MIN,
        patience=5,
        monitor="val_loss"
    )
    monitor.setup_early_stopping(es_config)
    
    # Simulate training
    import torch.nn as nn
    model = nn.Linear(10, 1)
    
    for epoch in range(20):
        # Simulate metrics
        metrics = {
            'train_loss': 1.0 / (epoch + 1),
            'val_loss': 1.2 / (epoch + 1),
            'learning_rate': 1e-3
        }
        
        # Update monitor
        should_stop = monitor.update(epoch, metrics, model)
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Get summary
    summary = monitor.get_training_summary()
    print(f"Training completed: {summary['total_epochs']} epochs")

asyncio.run(basic_early_stopping_example())
```

### Basic Learning Rate Scheduling

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling import (
    create_training_monitor, create_lr_scheduler_config, LRSchedulerType
)

async def basic_lr_scheduling_example():
    # Create training monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.COSINE_ANNEALING,
        initial_lr=1e-3,
        min_lr=1e-6,
        max_lr=1e-2
    )
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
    
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    # Simulate training
    for epoch in range(10):
        # Step scheduler
        monitor.lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: LR = {current_lr:.2e}")

asyncio.run(basic_lr_scheduling_example())
```

## Core Concepts

### Early Stopping Configuration

The `EarlyStoppingConfig` class controls early stopping behavior:

```python
from agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling import (
    EarlyStoppingConfig, EarlyStoppingStrategy, EarlyStoppingMode
)

config = EarlyStoppingConfig(
    enabled=True,                           # Enable early stopping
    strategy=EarlyStoppingStrategy.PATIENCE, # Stopping strategy
    mode=EarlyStoppingMode.MIN,             # Monitor mode (min/max)
    patience=10,                            # Patience epochs
    min_delta=0.0,                          # Minimum improvement
    min_percentage=0.01,                    # Minimum percentage improvement
    moving_average_window=5,                # Moving average window
    restore_best_weights=True,              # Restore best model
    verbose=True,                           # Verbose logging
    monitor="val_loss",                     # Metric to monitor
    min_epochs=0                            # Minimum epochs before stopping
)
```

### Learning Rate Scheduler Configuration

The `LRSchedulerConfig` class controls learning rate scheduling:

```python
from agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling import (
    LRSchedulerConfig, LRSchedulerType
)

config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.COSINE_ANNEALING,  # Scheduler type
    initial_lr=1e-3,                                  # Initial learning rate
    min_lr=1e-6,                                      # Minimum learning rate
    max_lr=1e-2,                                      # Maximum learning rate
    step_size=30,                                     # Step size for step schedulers
    gamma=0.1,                                        # Decay factor
    milestones=[30, 60, 90],                         # Milestones for multi-step
    T_max=100,                                        # Max iterations for cosine
    eta_min=0.0,                                      # Min LR for cosine
    factor=0.1,                                       # Factor for plateau
    patience=10                                       # Patience for plateau
)
```

## Early Stopping Strategies

### 1. Patience Strategy (Default)

Stop after N epochs without improvement:

```python
config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.PATIENCE,
    mode=EarlyStoppingMode.MIN,
    patience=10,
    monitor="val_loss"
)
```

**Use when:**
- Standard training scenarios
- Need to prevent overfitting
- Want to save training time

### 2. Delta Strategy

Stop when improvement is less than a threshold:

```python
config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.DELTA,
    mode=EarlyStoppingMode.MIN,
    min_delta=0.001,
    monitor="val_loss"
)
```

**Use when:**
- Need precise control over improvement threshold
- Training with small improvements
- Want to continue until convergence

### 3. Percentage Strategy

Stop when improvement percentage is below threshold:

```python
config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.PERCENTAGE,
    mode=EarlyStoppingMode.MIN,
    min_percentage=0.01,  # 1% improvement
    monitor="val_loss"
)
```

**Use when:**
- Relative improvement is more important than absolute
- Training with varying metric scales
- Need adaptive stopping criteria

### 4. Moving Average Strategy

Use moving average for stability:

```python
config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.MOVING_AVERAGE,
    mode=EarlyStoppingMode.MIN,
    moving_average_window=5,
    monitor="val_loss"
)
```

**Use when:**
- Noisy training metrics
- Need stable stopping decisions
- Training with high variance

### 5. Custom Strategy

Custom early stopping logic:

```python
def custom_stopping_function(epoch, metric, state):
    # Custom logic
    return epoch >= 100 or metric < 0.01

config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.CUSTOM,
    custom_stopping_function=custom_stopping_function,
    monitor="val_loss"
)
```

**Use when:**
- Domain-specific requirements
- Complex stopping conditions
- Need complete control

## Learning Rate Scheduling

### 1. Step LR

Reduce LR by gamma every step_size epochs:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.STEP,
    initial_lr=1e-3,
    step_size=30,
    gamma=0.1
)
```

### 2. Multi-Step LR

Reduce LR at specific milestones:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.MULTI_STEP,
    initial_lr=1e-3,
    milestones=[30, 60, 90],
    gamma=0.1
)
```

### 3. Exponential LR

Exponential decay:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.EXPONENTIAL,
    initial_lr=1e-3,
    gamma=0.95
)
```

### 4. Cosine Annealing (Default)

Cosine annealing schedule:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.COSINE_ANNEALING,
    initial_lr=1e-3,
    T_max=100,
    eta_min=1e-6
)
```

### 5. Cosine Annealing with Warm Restarts

Cosine annealing with periodic restarts:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.COSINE_ANNEALING_WARM_RESTARTS,
    initial_lr=1e-3,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)
```

### 6. Reduce LR on Plateau

Reduce LR when metric plateaus:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.REDUCE_LR_ON_PLATEAU,
    initial_lr=1e-3,
    factor=0.1,
    patience=10,
    threshold=1e-4,
    min_lr_plateau=1e-6
)
```

### 7. One Cycle

One cycle policy:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.ONE_CYCLE,
    initial_lr=1e-4,
    max_lr=1e-2,
    epochs=100,
    steps_per_epoch=100,
    pct_start=0.3,
    anneal_strategy="cos"
)
```

### 8. Cyclic LR

Cyclic learning rate:

```python
config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.CYCLIC,
    base_lr=1e-4,
    max_lr=1e-2,
    step_size_up=2000,
    step_size_down=2000,
    mode="triangular"
)
```

### 9. Custom LR

Custom learning rate function:

```python
def custom_lr_function(epoch):
    return 0.9 ** epoch

config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType.LAMBDA,
    custom_lr_function=custom_lr_function,
    initial_lr=1e-3
)
```

## Usage Examples

### Example 1: Classification with Early Stopping

```python
import asyncio
from agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling import (
    create_training_monitor, create_early_stopping_config, create_lr_scheduler_config,
    EarlyStoppingStrategy, EarlyStoppingMode, LRSchedulerType
)
from agents.backend.onyx.server.features.blog_posts.production_transformers import DeviceManager

async def classification_training_example():
    # Initialize monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup early stopping
    es_config = create_early_stopping_config(
        enabled=True,
        strategy=EarlyStoppingStrategy.PATIENCE,
        mode=EarlyStoppingMode.MIN,
        patience=5,
        monitor="val_loss",
        min_epochs=10
    )
    monitor.setup_early_stopping(es_config)
    
    # Setup LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.COSINE_ANNEALING,
        initial_lr=1e-3,
        min_lr=1e-6,
        max_lr=1e-2
    )
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Linear(100, 3)  # 3-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    # Training loop
    for epoch in range(50):
        # Simulate training metrics
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        val_loss = 1.2 / (epoch + 1) + np.random.normal(0, 0.02)
        train_acc = 0.5 + epoch * 0.01 + np.random.normal(0, 0.005)
        val_acc = 0.48 + epoch * 0.008 + np.random.normal(0, 0.01)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        # Update monitor
        should_stop = monitor.update(epoch, metrics, model)
        
        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Get training summary
    summary = monitor.get_training_summary()
    print(f"Training completed: {summary['total_epochs']} epochs")
    print(f"Best validation loss: {summary['early_stopping_state']['best_score']:.4f}")
    
    # Plot training curves
    monitor.plot_training_curves("training_curves.png")
    
    return summary

asyncio.run(classification_training_example())
```

### Example 2: Regression with Advanced Scheduling

```python
async def regression_training_example():
    # Initialize monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup early stopping with moving average
    es_config = create_early_stopping_config(
        enabled=True,
        strategy=EarlyStoppingStrategy.MOVING_AVERAGE,
        mode=EarlyStoppingMode.MIN,
        moving_average_window=5,
        monitor="val_loss",
        min_epochs=20
    )
    monitor.setup_early_stopping(es_config)
    
    # Setup one cycle LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.ONE_CYCLE,
        initial_lr=1e-4,
        max_lr=1e-2,
        epochs=100,
        steps_per_epoch=50,
        pct_start=0.3
    )
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    # Training loop
    for epoch in range(100):
        # Simulate regression metrics
        train_loss = 0.5 * np.exp(-epoch / 20) + np.random.normal(0, 0.01)
        val_loss = 0.6 * np.exp(-epoch / 25) + np.random.normal(0, 0.02)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        # Update monitor
        should_stop = monitor.update(epoch, metrics, model)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    return monitor.get_training_summary()

asyncio.run(regression_training_example())
```

### Example 3: Custom Early Stopping Logic

```python
async def custom_early_stopping_example():
    # Define custom stopping function
    def custom_stopping(epoch, metric, state):
        # Stop if validation loss is very low or if we've trained for too long
        if metric < 0.01:
            return True
        if epoch >= 200:
            return True
        # Stop if no improvement for 20 epochs and we're past epoch 50
        if epoch > 50 and state.counter >= 20:
            return True
        return False
    
    # Initialize monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup custom early stopping
    es_config = EarlyStoppingConfig(
        enabled=True,
        strategy=EarlyStoppingStrategy.CUSTOM,
        custom_stopping_function=custom_stopping,
        monitor="val_loss"
    )
    monitor.setup_early_stopping(es_config)
    
    # Setup cyclic LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.CYCLIC,
        base_lr=1e-4,
        max_lr=1e-2,
        step_size_up=1000,
        step_size_down=1000,
        mode="triangular"
    )
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Linear(100, 10)
    optimizer = optim.Adam(model.parameters(), lr=lr_config.base_lr)
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    # Training loop
    for epoch in range(300):
        # Simulate training
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.01)
        val_loss = 1.2 / (epoch + 1) + np.random.normal(0, 0.02)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        should_stop = monitor.update(epoch, metrics, model)
        
        if should_stop:
            print(f"Custom early stopping triggered at epoch {epoch}")
            break
    
    return monitor.get_training_summary()

asyncio.run(custom_early_stopping_example())
```

### Example 4: Integration with Model Training

```python
from agents.backend.onyx.server.features.blog_posts.model_training import ModelTrainer, TrainingConfig, ModelType, TrainingMode

async def integrated_training_example():
    # Create trainer
    device_manager = DeviceManager()
    trainer = ModelTrainer(device_manager)
    
    # Configure training with early stopping and LR scheduling
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="my_model",
        dataset_path="path/to/dataset",
        learning_rate=1e-3,
        num_epochs=100,
        early_stopping_patience=10,  # Enable early stopping
        eval_split=0.15,
        test_split=0.15
    )
    
    # Train with automatic early stopping and LR scheduling
    results = await trainer.train(config)
    
    # Access results
    print(f"Training completed in {results['total_training_time']:.2f} seconds")
    print(f"Best model saved at: {results['best_model_path']}")
    
    # Early stopping results
    if 'training_summary' in results:
        summary = results['training_summary']
        es_state = summary['early_stopping_state']
        print(f"Early stopping triggered: {es_state['stopped']}")
        print(f"Best epoch: {es_state['best_epoch']}")
        print(f"Best metric: {es_state['best_score']:.4f}")
    
    # LR scheduling results
    if 'training_summary' in results:
        lr_state = summary['lr_scheduler_state']
        print(f"Final LR: {lr_state['current_lr']:.2e}")
        print(f"Best LR: {lr_state['best_lr']:.2e}")
    
    return results

asyncio.run(integrated_training_example())
```

## Advanced Features

### Training Monitor

The `TrainingMonitor` class provides unified management:

```python
async def advanced_monitor_example():
    # Create monitor
    device_manager = DeviceManager()
    monitor = await create_training_monitor(device_manager)
    
    # Setup early stopping
    es_config = create_early_stopping_config(
        enabled=True,
        strategy=EarlyStoppingStrategy.PATIENCE,
        mode=EarlyStoppingMode.MIN,
        patience=5,
        monitor="val_loss"
    )
    monitor.setup_early_stopping(es_config)
    
    # Setup LR scheduler
    lr_config = create_lr_scheduler_config(
        scheduler_type=LRSchedulerType.COSINE_ANNEALING,
        initial_lr=1e-3
    )
    
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=lr_config.initial_lr)
    scheduler = monitor.setup_lr_scheduler(lr_config, optimizer)
    
    # Training loop
    for epoch in range(20):
        metrics = {
            'train_loss': 1.0 / (epoch + 1),
            'val_loss': 1.2 / (epoch + 1),
            'train_accuracy': 0.5 + epoch * 0.02,
            'val_accuracy': 0.48 + epoch * 0.018,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        # Update monitor
        should_stop = monitor.update(epoch, metrics, model)
        
        if should_stop:
            break
    
    # Get comprehensive summary
    summary = monitor.get_training_summary()
    
    # Plot training curves
    monitor.plot_training_curves("training_curves.png")
    
    # Restore best model
    monitor.restore_best_model(model)
    
    return summary

asyncio.run(advanced_monitor_example())
```

### Custom Early Stopping Functions

```python
def adaptive_patience_stopping(epoch, metric, state):
    """Adaptive patience based on training progress."""
    if epoch < 10:
        return False  # Don't stop in first 10 epochs
    
    # Increase patience as training progresses
    base_patience = 5
    adaptive_patience = base_patience + epoch // 20
    
    return state.counter >= adaptive_patience

def multi_metric_stopping(epoch, metric, state):
    """Stop based on multiple metrics."""
    # This would need access to multiple metrics
    # For demonstration, we'll use a simple condition
    return metric < 0.01 or epoch >= 100

def convergence_stopping(epoch, metric, state):
    """Stop when convergence is detected."""
    if len(state.history) < 10:
        return False
    
    # Check if recent improvements are very small
    recent_improvements = [
        abs(state.history[i] - state.history[i-1]) 
        for i in range(max(1, len(state.history)-5), len(state.history))
    ]
    
    avg_improvement = np.mean(recent_improvements)
    return avg_improvement < 1e-5
```

### Custom Learning Rate Functions

```python
def warmup_cosine_decay(epoch):
    """Warmup followed by cosine decay."""
    warmup_epochs = 10
    total_epochs = 100
    
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

def step_decay_with_plateau(epoch):
    """Step decay with plateau detection."""
    if epoch < 30:
        return 1.0
    elif epoch < 60:
        return 0.1
    elif epoch < 90:
        return 0.01
    else:
        return 0.001

def cyclical_lr(epoch):
    """Custom cyclical learning rate."""
    cycle_length = 20
    cycle = epoch // cycle_length
    position = epoch % cycle_length
    
    # Triangular cycle
    if position < cycle_length // 2:
        return 0.1 + 0.9 * (2 * position / cycle_length)
    else:
        return 0.1 + 0.9 * (2 * (cycle_length - position) / cycle_length)
```

## Best Practices

### 1. Choose Appropriate Early Stopping Strategy

- **Patience**: Default choice for most scenarios
- **Delta**: When you need precise control
- **Percentage**: For relative improvement tracking
- **Moving Average**: For noisy metrics
- **Custom**: For domain-specific requirements

### 2. Set Proper Patience Values

```python
# For fast convergence
config = EarlyStoppingConfig(patience=5)

# For stable training
config = EarlyStoppingConfig(patience=15)

# For complex models
config = EarlyStoppingConfig(patience=25)
```

### 3. Monitor the Right Metric

```python
# For classification
config = EarlyStoppingConfig(monitor="val_loss")  # or "val_accuracy"

# For regression
config = EarlyStoppingConfig(monitor="val_loss")  # or "val_mse"

# For custom metrics
config = EarlyStoppingConfig(monitor="val_f1_score")
```

### 4. Choose Appropriate LR Scheduler

```python
# For stable training
config = LRSchedulerConfig(scheduler_type=LRSchedulerType.COSINE_ANNEALING)

# For fast convergence
config = LRSchedulerConfig(scheduler_type=LRSchedulerType.ONE_CYCLE)

# For plateau detection
config = LRSchedulerConfig(scheduler_type=LRSchedulerType.REDUCE_LR_ON_PLATEAU)

# For exploration
config = LRSchedulerConfig(scheduler_type=LRSchedulerType.CYCLIC)
```

### 5. Set Minimum Epochs

```python
# Ensure minimum training time
config = EarlyStoppingConfig(
    patience=10,
    min_epochs=20  # Train for at least 20 epochs
)
```

### 6. Use Model Restoration

```python
# Always restore best model
config = EarlyStoppingConfig(
    restore_best_weights=True,
    monitor="val_loss"
)
```

### 7. Monitor Training Progress

```python
# Get training summary
summary = monitor.get_training_summary()

# Check early stopping state
es_state = summary['early_stopping_state']
if es_state['stopped']:
    print(f"Early stopping triggered at epoch {es_state['best_epoch']}")
    print(f"Best metric: {es_state['best_score']:.4f}")

# Check LR scheduling
lr_state = summary['lr_scheduler_state']
print(f"LR range: {min(lr_state['history']):.2e} - {max(lr_state['history']):.2e}")
```

## Performance Optimization

### 1. Efficient Monitoring

```python
# Use efficient update frequency
for epoch in range(num_epochs):
    # Update every N epochs for efficiency
    if epoch % update_frequency == 0:
        should_stop = monitor.update(epoch, metrics, model)
        if should_stop:
            break
```

### 2. Memory Management

```python
# Limit history size for large datasets
config = EarlyStoppingConfig(
    moving_average_window=5,  # Smaller window for memory efficiency
    patience=10
)
```

### 3. Parallel Processing

```python
# Use async operations
async def training_loop():
    for epoch in range(num_epochs):
        metrics = await train_epoch()
        should_stop = monitor.update(epoch, metrics, model)
        if should_stop:
            break
```

## Production Deployment

### 1. Configuration Management

```python
import os

# Environment-based configuration
es_config = EarlyStoppingConfig(
    enabled=os.getenv('EARLY_STOPPING_ENABLED', 'true').lower() == 'true',
    patience=int(os.getenv('EARLY_STOPPING_PATIENCE', '10')),
    monitor=os.getenv('EARLY_STOPPING_MONITOR', 'val_loss')
)

lr_config = LRSchedulerConfig(
    scheduler_type=LRSchedulerType[os.getenv('LR_SCHEDULER_TYPE', 'COSINE_ANNEALING')],
    initial_lr=float(os.getenv('INITIAL_LR', '1e-3')),
    min_lr=float(os.getenv('MIN_LR', '1e-6'))
)
```

### 2. Logging and Monitoring

```python
import logging

# Comprehensive logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log early stopping events
if should_stop:
    logger.info(f"Early stopping triggered at epoch {epoch}")
    logger.info(f"Best metric: {monitor.early_stopping.state.best_score:.4f}")

# Log LR changes
current_lr = monitor.lr_scheduler.get_lr()
logger.info(f"Learning rate: {current_lr:.2e}")
```

### 3. Error Handling

```python
try:
    should_stop = monitor.update(epoch, metrics, model)
except Exception as e:
    logger.error(f"Error in training monitor: {e}")
    # Fall back to default behavior
    should_stop = False
```

### 4. Testing

```python
# Run comprehensive tests
from agents.backend.onyx.server.features.blog_posts.test_early_stopping_lr_scheduling import run_all_tests

success = run_all_tests()
if not success:
    raise RuntimeError("Early stopping and LR scheduling tests failed")
```

## Troubleshooting

### Common Issues

#### 1. Early Stopping Too Aggressive

**Problem**: Training stops too early

**Solution**:
```python
# Increase patience
config = EarlyStoppingConfig(patience=20)

# Use moving average for stability
config = EarlyStoppingConfig(
    strategy=EarlyStoppingStrategy.MOVING_AVERAGE,
    moving_average_window=10
)

# Set minimum epochs
config = EarlyStoppingConfig(min_epochs=50)
```

#### 2. Early Stopping Not Triggering

**Problem**: Training continues indefinitely

**Solution**:
```python
# Check metric direction
config = EarlyStoppingConfig(
    mode=EarlyStoppingMode.MIN,  # or MAX for accuracy
    monitor="val_loss"  # Ensure correct metric
)

# Reduce patience
config = EarlyStoppingConfig(patience=5)
```

#### 3. LR Scheduler Not Working

**Problem**: Learning rate not changing

**Solution**:
```python
# Check scheduler type
config = LRSchedulerConfig(scheduler_type=LRSchedulerType.STEP)

# Ensure proper stepping
scheduler.step()  # Call step() after optimizer.step()

# Check LR range
config = LRSchedulerConfig(
    initial_lr=1e-3,
    min_lr=1e-6,
    max_lr=1e-2
)
```

#### 4. Memory Issues

**Problem**: Out of memory during training

**Solution**:
```python
# Limit history size
config = EarlyStoppingConfig(moving_average_window=3)

# Use efficient monitoring
if epoch % 5 == 0:  # Update every 5 epochs
    should_stop = monitor.update(epoch, metrics, model)
```

### Debug Mode

```python
# Enable debug logging
logging.getLogger('agents.backend.onyx.server.features.blog_posts.early_stopping_lr_scheduling').setLevel(logging.DEBUG)

# Use verbose mode
config = EarlyStoppingConfig(verbose=True)

# Check state manually
state = monitor.early_stopping.get_state()
print(f"Early stopping state: {state}")
```

## API Reference

### EarlyStoppingConfig

```python
@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    strategy: EarlyStoppingStrategy = EarlyStoppingStrategy.PATIENCE
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN
    patience: int = 10
    min_delta: float = 0.0
    min_percentage: float = 0.01
    moving_average_window: int = 5
    restore_best_weights: bool = True
    verbose: bool = True
    custom_stopping_function: Optional[Callable] = None
    monitor: str = "val_loss"
    min_epochs: int = 0
```

### LRSchedulerConfig

```python
@dataclass
class LRSchedulerConfig:
    scheduler_type: LRSchedulerType = LRSchedulerType.COSINE_ANNEALING
    initial_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [30, 60, 90])
    T_max: int = 100
    eta_min: float = 0.0
    T_0: int = 10
    T_mult: int = 2
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr_plateau: float = 0.0
    epochs: int = 100
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    base_lr: float = 1e-4
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    scale_fn: Optional[Callable] = None
    scale_mode: str = "cycle"
    custom_scheduler_function: Optional[Callable] = None
    custom_lr_function: Optional[Callable] = None
```

### TrainingMonitor

```python
class TrainingMonitor:
    def __init__(self, device_manager: DeviceManager)
    
    def setup_early_stopping(self, config: EarlyStoppingConfig)
    
    def setup_lr_scheduler(self, config: LRSchedulerConfig, optimizer: optim.Optimizer, num_training_steps: int = None) -> optim.lr_scheduler._LRScheduler
    
    def update(self, epoch: int, metrics: Dict[str, float], model: nn.Module) -> bool
    
    def restore_best_model(self, model: nn.Module)
    
    def get_training_summary(self) -> Dict[str, Any]
    
    def plot_training_curves(self, save_path: Optional[str] = None)
    
    def reset(self)
```

### Quick Functions

```python
async def create_training_monitor(device_manager: DeviceManager) -> TrainingMonitor

def create_early_stopping_config(
    enabled: bool = True,
    strategy: EarlyStoppingStrategy = EarlyStoppingStrategy.PATIENCE,
    mode: EarlyStoppingMode = EarlyStoppingMode.MIN,
    patience: int = 10,
    monitor: str = "val_loss",
    min_epochs: int = 0
) -> EarlyStoppingConfig

def create_lr_scheduler_config(
    scheduler_type: LRSchedulerType = LRSchedulerType.COSINE_ANNEALING,
    initial_lr: float = 1e-3,
    min_lr: float = 1e-6,
    max_lr: float = 1e-2
) -> LRSchedulerConfig
```

## Conclusion

This early stopping and learning rate scheduling system provides enterprise-grade functionality for intelligent training termination and dynamic learning rate adjustment. By following the best practices outlined in this guide, you can ensure robust, efficient, and high-quality model training with automatic optimization.

For additional support or questions, refer to the test suite and examples provided in the codebase. 