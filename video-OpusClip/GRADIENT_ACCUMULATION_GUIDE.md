# Gradient Accumulation Guide for Video-OpusClip

This guide provides comprehensive instructions for implementing and using gradient accumulation in the Video-OpusClip system to train with large effective batch sizes while managing memory constraints.

## Table of Contents

1. [Overview](#overview)
2. [Understanding Gradient Accumulation](#understanding-gradient-accumulation)
3. [Configuration](#configuration)
4. [Implementation](#implementation)
5. [Multi-GPU Integration](#multi-gpu-integration)
6. [Memory Management](#memory-management)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

Gradient accumulation is a technique that allows training with large effective batch sizes by accumulating gradients over multiple forward/backward passes before updating the model parameters. This is particularly useful for:

- Training with large models that don't fit in GPU memory
- Achieving stable training with large effective batch sizes
- Managing memory constraints in multi-GPU setups
- Implementing advanced training strategies

### Key Benefits

- **Memory Efficiency**: Train with large effective batch sizes using limited memory
- **Stability**: Large batch sizes often lead to more stable training
- **Flexibility**: Adjust effective batch size without changing data loading
- **Compatibility**: Works with mixed precision, multi-GPU, and various optimizers

## Understanding Gradient Accumulation

### How It Works

1. **Forward Pass**: Process a small batch of data
2. **Backward Pass**: Compute gradients and accumulate them
3. **Repeat**: Process more batches, accumulating gradients
4. **Update**: After N accumulation steps, update model parameters
5. **Reset**: Zero gradients and repeat

### Mathematical Foundation

```python
# Standard training (batch_size = 32)
loss = loss_fn(model(batch), targets)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# Gradient accumulation (effective_batch_size = 128, accumulation_steps = 4)
for i in range(4):
    loss = loss_fn(model(batch), targets)
    (loss / 4).backward()  # Scale loss by accumulation steps
    # Don't call optimizer.step() yet
    
optimizer.step()  # Update with accumulated gradients
optimizer.zero_grad()
```

### Effective Batch Size

```python
effective_batch_size = physical_batch_size × accumulation_steps
```

## Configuration

### Basic Configuration

```python
from gradient_accumulation import GradientAccumulationConfig

config = GradientAccumulationConfig(
    accumulation_steps=4,           # Number of steps to accumulate
    effective_batch_size=128,       # Target effective batch size
    max_batch_size=32,             # Maximum physical batch size
    strategy='standard',            # Accumulation strategy
    use_amp=True,                  # Use mixed precision
    auto_adjust=True               # Auto-adjust based on memory
)
```

### Advanced Configuration

```python
config = GradientAccumulationConfig(
    # Basic settings
    accumulation_steps=8,
    effective_batch_size=256,
    max_batch_size=32,
    
    # Strategy
    strategy='adaptive',  # 'standard', 'dynamic', 'adaptive'
    
    # Memory management
    memory_threshold=0.8,  # 80% memory usage threshold
    auto_adjust=True,
    
    # Mixed precision
    use_amp=True,
    amp_dtype=torch.float16,
    
    # Gradient handling
    gradient_clip_norm=1.0,
    gradient_clip_value=None,
    
    # Multi-GPU
    sync_across_devices=True,
    reduce_gradients=True,
    
    # Monitoring
    log_accumulation=True,
    log_frequency=10
)
```

### Configuration Strategies

#### 1. Standard Strategy
```python
# Fixed accumulation steps
config = GradientAccumulationConfig(
    strategy='standard',
    accumulation_steps=4
)
```

#### 2. Dynamic Strategy
```python
# Adjust based on memory usage
config = GradientAccumulationConfig(
    strategy='dynamic',
    memory_threshold=0.8,
    auto_adjust=True
)
```

#### 3. Adaptive Strategy
```python
# Adaptive based on performance and memory
config = GradientAccumulationConfig(
    strategy='adaptive',
    memory_threshold=0.8,
    auto_adjust=True
)
```

## Implementation

### Basic Implementation

```python
from gradient_accumulation import GradientAccumulationTrainer, create_accumulation_config

# Create configuration
config = create_accumulation_config(
    target_batch_size=128,
    max_batch_size=32,
    use_amp=True
)

# Create trainer
trainer = GradientAccumulationTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Training loop
for epoch in range(num_epochs):
    metrics = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
```

### Custom Implementation

```python
from gradient_accumulation import GradientAccumulationManager

# Create manager
manager = GradientAccumulationManager(config)

# Custom training loop
for batch_idx, batch in enumerate(train_loader):
    # Forward pass
    outputs = model(batch)
    loss = loss_fn(outputs, targets)
    
    # Accumulate gradients
    manager.accumulate_gradients(loss, model)
    
    # Check if we should update
    if manager.should_update(batch_idx):
        # Update optimizer
        update_info = manager.update_optimizer(optimizer, model)
        
        # Update scheduler
        scheduler.step()
        
        # Log progress
        print(f"Updated: {update_info}")
```

### Integration with Existing Training

```python
# Enhance existing trainer with gradient accumulation
class EnhancedTrainer:
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.accumulation_manager = GradientAccumulationManager(config)
    
    def train_epoch(self, epoch):
        for batch_idx, batch in enumerate(self.train_loader):
            # Standard forward pass
            loss = self.forward_pass(batch)
            
            # Accumulate gradients
            self.accumulation_manager.accumulate_gradients(loss, self.model)
            
            # Update if needed
            if self.accumulation_manager.should_update(batch_idx):
                self.accumulation_manager.update_optimizer(self.optimizer, self.model)
                self.scheduler.step()
```

## Multi-GPU Integration

### DataParallel Integration

```python
from multi_gpu_training import DataParallelTrainer

# Create DataParallel trainer with gradient accumulation
config = GradientAccumulationConfig(
    accumulation_steps=4,
    effective_batch_size=128,
    sync_across_devices=True
)

trainer = DataParallelTrainer(
    model=model,
    config=multi_gpu_config,
    train_loader=train_loader,
    accumulation_config=config  # Add accumulation config
)
```

### DistributedDataParallel Integration

```python
from multi_gpu_training import DistributedDataParallelTrainer

# Create DDP trainer with gradient accumulation
config = GradientAccumulationConfig(
    accumulation_steps=4,
    effective_batch_size=128,
    reduce_gradients=True
)

trainer = DistributedDataParallelTrainer(
    model=model,
    config=ddp_config,
    train_dataset=train_dataset,
    accumulation_config=config
)
```

### Synchronization Across Devices

```python
# Ensure gradients are synchronized across devices
config = GradientAccumulationConfig(
    sync_across_devices=True,
    reduce_gradients=True
)

# In training loop
if config.sync_across_devices:
    # Synchronize gradients across devices
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()
```

## Memory Management

### Automatic Memory Adjustment

```python
# Enable automatic memory adjustment
config = GradientAccumulationConfig(
    auto_adjust=True,
    memory_threshold=0.8
)

# The system will automatically:
# 1. Monitor memory usage
# 2. Increase accumulation steps if memory usage is high
# 3. Provide recommendations for optimization
```

### Memory Monitoring

```python
from gradient_accumulation import MemoryMonitor

monitor = MemoryMonitor()

# Get current memory usage
memory_usage = monitor.get_memory_usage()
print(f"GPU Utilization: {memory_usage['gpu_utilization']:.2f}")
print(f"Memory Allocated: {memory_usage['memory_allocated']:.2f}GB")

# Get memory trend
trend = monitor.get_memory_trend()
print(f"Memory Trend: {trend['trend']}")

# Get recommendations
recommendations = monitor.get_memory_recommendations()
for rec in recommendations:
    print(f"Recommendation: {rec}")
```

### Memory Optimization Strategies

#### 1. Gradient Checkpointing
```python
# Enable gradient checkpointing for memory efficiency
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

config = GradientAccumulationConfig(
    accumulation_steps=8,  # Increase accumulation steps
    use_amp=True          # Use mixed precision
)
```

#### 2. Dynamic Batch Sizing
```python
def calculate_optimal_batch_size(available_memory, model_memory):
    """Calculate optimal batch size based on available memory."""
    sample_memory = 0.1  # GB per sample (estimate)
    max_samples = (available_memory - model_memory) / sample_memory
    return max(1, int(max_samples))

# Use in configuration
optimal_batch_size = calculate_optimal_batch_size(8.0, 2.0)  # 8GB available, 2GB model
config = GradientAccumulationConfig(
    max_batch_size=optimal_batch_size,
    effective_batch_size=128
)
```

## Performance Optimization

### Performance Tracking

```python
from gradient_accumulation import PerformanceTracker

tracker = PerformanceTracker()

# Record metrics during training
tracker.record_accumulation_time(0.5)  # seconds
tracker.record_update_time(0.1)        # seconds
tracker.record_memory_peak(4.2)        # GB
tracker.record_gradient_norm(1.5)      # L2 norm

# Get performance metrics
metrics = tracker.get_metrics()
print(f"Average accumulation time: {metrics['accumulation_time_mean']:.3f}s")
print(f"Average update time: {metrics['update_time_mean']:.3f}s")
```

### Optimization Strategies

#### 1. Efficient Data Loading
```python
# Optimize data loading for gradient accumulation
train_loader = DataLoader(
    dataset,
    batch_size=config.max_batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

#### 2. Mixed Precision Training
```python
# Use mixed precision for better performance
config = GradientAccumulationConfig(
    use_amp=True,
    amp_dtype=torch.float16
)

# The system automatically handles:
# - Loss scaling
# - Gradient unscaling
# - Mixed precision backward pass
```

#### 3. Gradient Clipping
```python
# Configure gradient clipping
config = GradientAccumulationConfig(
    gradient_clip_norm=1.0,
    gradient_clip_value=None  # Use norm clipping
)

# Or use value clipping
config = GradientAccumulationConfig(
    gradient_clip_norm=None,
    gradient_clip_value=0.5
)
```

## Best Practices

### 1. Configuration Guidelines

```python
# Good practices for configuration
config = GradientAccumulationConfig(
    # Set reasonable accumulation steps
    accumulation_steps=4,  # 2-8 is usually good
    
    # Enable automatic adjustment
    auto_adjust=True,
    memory_threshold=0.8,
    
    # Use mixed precision
    use_amp=True,
    
    # Enable monitoring
    log_accumulation=True,
    log_frequency=10
)
```

### 2. Learning Rate Scaling

```python
# Scale learning rate for large effective batch sizes
def scale_learning_rate(base_lr, effective_batch_size, base_batch_size=32):
    """Scale learning rate for large effective batch sizes."""
    return base_lr * (effective_batch_size / base_batch_size) ** 0.5

# Usage
effective_batch_size = 128
scaled_lr = scale_learning_rate(1e-3, effective_batch_size)
optimizer = optim.Adam(model.parameters(), lr=scaled_lr)
```

### 3. Scheduler Integration

```python
# Integrate with learning rate schedulers
config = GradientAccumulationConfig(accumulation_steps=4)

# Update scheduler every accumulation step
for batch_idx, batch in enumerate(train_loader):
    # ... training code ...
    
    if manager.should_update(batch_idx):
        manager.update_optimizer(optimizer, model)
        scheduler.step()  # Update scheduler
```

### 4. Checkpointing

```python
# Save checkpoints with accumulation state
def save_checkpoint(trainer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'accumulation_state': trainer.accumulation_manager.get_status(),
        'config': trainer.config
    }
    torch.save(checkpoint, path)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # Restore accumulation state if needed
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues

```python
# Problem: Out of memory during accumulation
# Solution: Reduce batch size or increase accumulation steps

config = GradientAccumulationConfig(
    max_batch_size=16,  # Reduce from 32
    accumulation_steps=8,  # Increase from 4
    auto_adjust=True
)
```

#### 2. Gradient Explosion

```python
# Problem: Gradients become too large
# Solution: Use gradient clipping

config = GradientAccumulationConfig(
    gradient_clip_norm=1.0,
    gradient_clip_value=0.5
)
```

#### 3. Slow Training

```python
# Problem: Training is too slow
# Solution: Optimize data loading and use mixed precision

config = GradientAccumulationConfig(
    use_amp=True,
    accumulation_steps=2  # Reduce accumulation steps
)

# Optimize data loader
train_loader = DataLoader(
    dataset,
    batch_size=config.max_batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

#### 4. Inconsistent Results

```python
# Problem: Results vary between runs
# Solution: Set random seeds and ensure proper gradient scaling

import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)

config = GradientAccumulationConfig(
    accumulation_steps=4,
    use_amp=True  # Ensure proper loss scaling
)
```

### Debugging Tools

```python
# Enable debugging for gradient accumulation
config = GradientAccumulationConfig(
    log_accumulation=True,
    log_frequency=1  # Log every step for debugging
)

# Monitor gradients
def debug_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")
```

## Examples

### Example 1: Basic Gradient Accumulation

```python
from gradient_accumulation import GradientAccumulationTrainer, create_accumulation_config

# Create simple configuration
config = create_accumulation_config(
    target_batch_size=128,
    max_batch_size=32
)

# Create trainer
trainer = GradientAccumulationTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer
)

# Train
for epoch in range(10):
    metrics = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}: Loss={metrics['train_loss']:.4f}")
```

### Example 2: Advanced Configuration

```python
# Advanced configuration with memory management
config = GradientAccumulationConfig(
    accumulation_steps=8,
    effective_batch_size=256,
    max_batch_size=32,
    strategy='adaptive',
    memory_threshold=0.8,
    auto_adjust=True,
    use_amp=True,
    gradient_clip_norm=1.0,
    log_accumulation=True
)

trainer = GradientAccumulationTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### Example 3: Multi-GPU with Gradient Accumulation

```python
from multi_gpu_training import MultiGPUConfig
from gradient_accumulation import GradientAccumulationConfig

# Multi-GPU configuration
multi_gpu_config = MultiGPUConfig(
    strategy='dataparallel',
    num_gpus=4
)

# Gradient accumulation configuration
accumulation_config = GradientAccumulationConfig(
    accumulation_steps=4,
    effective_batch_size=512,  # 4 GPUs × 32 batch × 4 accumulation
    max_batch_size=32,
    sync_across_devices=True
)

# Create trainer
trainer = DataParallelTrainer(
    model=model,
    config=multi_gpu_config,
    train_loader=train_loader,
    accumulation_config=accumulation_config
)
```

### Example 4: Custom Training Loop

```python
from gradient_accumulation import GradientAccumulationManager

# Create manager
manager = GradientAccumulationManager(config)

# Custom training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Forward pass
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
        
        # Accumulate gradients
        manager.accumulate_gradients(loss, model)
        
        # Check memory and adjust if needed
        manager.check_memory_and_adjust(model)
        
        # Update if needed
        if manager.should_update(batch_idx):
            update_info = manager.update_optimizer(optimizer, model)
            scheduler.step()
            
            # Log status
            status = manager.get_status()
            print(f"Effective batch size: {status['effective_batch_size']}")
```

### Example 5: Memory-Efficient Training

```python
# Memory-efficient configuration
config = GradientAccumulationConfig(
    accumulation_steps=16,  # Large accumulation for memory efficiency
    max_batch_size=8,      # Small physical batch size
    effective_batch_size=128,
    use_amp=True,          # Use mixed precision
    auto_adjust=True,      # Auto-adjust based on memory
    memory_threshold=0.7   # Conservative memory threshold
)

# Enable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

trainer = GradientAccumulationTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer
)
```

## Summary

This guide provides comprehensive coverage of gradient accumulation implementation in the Video-OpusClip system. Key takeaways:

1. **Choose appropriate accumulation steps**: 2-8 steps usually work well
2. **Use mixed precision**: Improves performance and reduces memory usage
3. **Monitor memory usage**: Enable auto-adjustment for optimal performance
4. **Scale learning rate**: Adjust for large effective batch sizes
5. **Integrate with schedulers**: Update schedulers every accumulation step
6. **Use gradient clipping**: Prevent gradient explosion
7. **Optimize data loading**: Use multiple workers and pin memory

The gradient accumulation system is designed to be flexible, efficient, and easy to integrate with existing training pipelines while providing advanced features for memory management and performance optimization. 