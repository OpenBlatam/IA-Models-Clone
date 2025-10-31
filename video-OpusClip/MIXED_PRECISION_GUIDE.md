# Mixed Precision Training Guide for Video-OpusClip

This guide provides comprehensive instructions for implementing and using mixed precision training in the Video-OpusClip system using PyTorch's Automatic Mixed Precision (AMP) with `torch.cuda.amp`.

## Table of Contents

1. [Overview](#overview)
2. [Understanding Mixed Precision](#understanding-mixed-precision)
3. [Configuration](#configuration)
4. [Implementation](#implementation)
5. [Performance Optimization](#performance-optimization)
6. [Memory Management](#memory-management)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

Mixed precision training uses both FP16 (16-bit) and FP32 (32-bit) floating-point formats to accelerate training while maintaining numerical stability. This is particularly beneficial for:

- **Performance**: 1.5-3x speedup on modern GPUs
- **Memory Efficiency**: Up to 50% reduction in memory usage
- **Training Stability**: Automatic handling of numerical issues
- **Compatibility**: Works with existing models and optimizers

### Key Benefits

- **Speed**: Faster training on Tensor Core-enabled GPUs
- **Memory**: Reduced memory usage for larger batch sizes
- **Stability**: Automatic gradient scaling prevents underflow
- **Compatibility**: Drop-in replacement for existing training code

## Understanding Mixed Precision

### How It Works

1. **Forward Pass**: Use FP16 for most operations
2. **Backward Pass**: Compute gradients in FP16
3. **Gradient Scaling**: Scale loss to prevent underflow
4. **Optimizer Step**: Unscale gradients before optimization
5. **Master Weights**: Keep optimizer states in FP32

### Mathematical Foundation

```python
# Standard training (FP32)
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()

# Mixed precision training (FP16 + FP32)
with autocast():
    loss = loss_fn(model(inputs), targets)

scaled_loss = scaler.scale(loss)  # Scale to prevent underflow
scaled_loss.backward()
scaler.step(optimizer)  # Unscale and step
scaler.update()  # Update scaling factor
```

### Gradient Scaling

```python
# Loss scaling prevents underflow in FP16
original_loss = 1e-6  # Very small loss
scaled_loss = original_loss * 2**16  # Scale up
# Now in FP16: scaled_loss = 0.065536 (representable)
```

## Configuration

### Basic Configuration

```python
from mixed_precision_training import MixedPrecisionConfig

config = MixedPrecisionConfig(
    enabled=True,           # Enable mixed precision
    dtype=torch.float16,    # Use FP16
    init_scale=2**16,       # Initial scaling factor
    growth_factor=2.0,      # Scale growth factor
    backoff_factor=0.5      # Scale backoff factor
)
```

### Advanced Configuration

```python
config = MixedPrecisionConfig(
    # Basic settings
    enabled=True,
    dtype=torch.float16,
    
    # Gradient scaling
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    
    # Performance optimization
    cache_enabled=True,
    autocast_enabled=True,
    
    # Memory optimization
    memory_efficient=True,
    pin_memory=True,
    
    # Monitoring
    log_scaling=True,
    log_frequency=100,
    save_scaler_state=True,
    
    # Error handling
    handle_overflow=True,
    overflow_threshold=float('inf'),
    
    # Multi-GPU settings
    sync_scaler=True,
    broadcast_scaler=True
)
```

### Configuration Strategies

#### 1. Conservative Strategy
```python
# Safe configuration for stability
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**15,  # Lower initial scale
    growth_factor=1.5,  # Slower growth
    backoff_factor=0.7,  # Slower backoff
    handle_overflow=True
)
```

#### 2. Aggressive Strategy
```python
# Performance-focused configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16,  # Higher initial scale
    growth_factor=2.0,  # Faster growth
    backoff_factor=0.5,  # Faster backoff
    cache_enabled=True
)
```

#### 3. Memory-Efficient Strategy
```python
# Memory-optimized configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    memory_efficient=True,
    pin_memory=True,
    cache_enabled=False  # Disable cache for memory
)
```

## Implementation

### Basic Implementation

```python
from mixed_precision_training import MixedPrecisionTrainer, create_mixed_precision_config

# Create configuration
config = create_mixed_precision_config(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)

# Create trainer
trainer = MixedPrecisionTrainer(
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
from mixed_precision_training import MixedPrecisionManager

# Create manager
manager = MixedPrecisionManager(config)

# Custom training loop
for batch_idx, batch in enumerate(train_loader):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with manager.autocast_context():
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
    
    # Scale loss and backward pass
    scaled_loss = manager.scale_loss(loss)
    scaled_loss.backward()
    
    # Handle overflow
    if manager.handle_overflow(optimizer):
        continue
    
    # Unscale and step
    manager.unscale_optimizer(optimizer)
    success = manager.step_optimizer(optimizer)
    
    if success:
        manager.update_scaler()
    
    # Update scheduler
    scheduler.step()
```

### Integration with Existing Training

```python
# Enhance existing trainer with mixed precision
class EnhancedTrainer:
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.mp_manager = MixedPrecisionManager(config)
    
    def train_epoch(self, epoch):
        for batch_idx, batch in enumerate(self.train_loader):
            # Standard forward pass with mixed precision
            with self.mp_manager.autocast_context():
                loss = self.forward_pass(batch)
            
            # Mixed precision backward pass
            scaled_loss = self.mp_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # Mixed precision optimizer step
            self.mp_manager.step_optimizer(self.optimizer)
            self.mp_manager.update_scaler()
```

## Performance Optimization

### Benchmarking Mixed Precision

```python
from mixed_precision_training import benchmark_mixed_precision

# Benchmark mixed precision vs full precision
results = benchmark_mixed_precision(
    model=model,
    train_loader=train_loader,
    config=config,
    num_steps=100
)

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Memory savings: {results['memory_savings']:.1f}%")
print(f"FP32 time: {results['fp32_time']:.2f}s")
print(f"MP time: {results['mp_time']:.2f}s")
```

### Performance Monitoring

```python
# Monitor performance during training
def monitor_performance(trainer):
    stats = trainer.get_status()
    
    print(f"Scaler scale: {stats['scaler_state']['scale']:.2e}")
    print(f"Training time: {stats['performance_metrics']['training_time_mean']:.3f}s")
    print(f"Memory usage: {stats['memory_usage']['gpu_memory_allocated']:.2f}GB")
    print(f"Overflow count: {stats['performance_metrics']['overflow_count']}")
```

### Optimization Strategies

#### 1. Gradient Scaling Optimization
```python
# Optimize gradient scaling parameters
config = MixedPrecisionConfig(
    init_scale=2**16,      # Start with 2^16
    growth_factor=2.0,     # Double scale when no overflow
    backoff_factor=0.5,    # Halve scale on overflow
    growth_interval=2000   # Check every 2000 steps
)
```

#### 2. Memory Optimization
```python
# Memory-efficient configuration
config = MixedPrecisionConfig(
    memory_efficient=True,
    pin_memory=True,
    cache_enabled=False,  # Disable cache for memory
    dtype=torch.float16
)
```

#### 3. Performance Optimization
```python
# Performance-focused configuration
config = MixedPrecisionConfig(
    cache_enabled=True,
    autocast_enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)
```

## Memory Management

### Memory Monitoring

```python
from mixed_precision_training import MixedPrecisionMemoryMonitor

monitor = MixedPrecisionMemoryMonitor()

# Get current memory usage
memory_usage = monitor.get_memory_usage()
print(f"GPU Memory: {memory_usage['gpu_memory_allocated']:.2f}GB allocated")
print(f"GPU Utilization: {memory_usage['gpu_utilization']:.2f}")

# Estimate memory savings
fp32_memory = 8.0  # GB (estimated FP32 memory)
savings = monitor.estimate_memory_savings(fp32_memory)
print(f"Memory savings: {savings:.1f}%")
```

### Memory Optimization Strategies

#### 1. Gradient Checkpointing
```python
# Enable gradient checkpointing for memory efficiency
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

config = MixedPrecisionConfig(
    enabled=True,
    memory_efficient=True
)
```

#### 2. Dynamic Memory Allocation
```python
# Monitor and adjust based on memory usage
def adjust_batch_size_based_on_memory(model, base_batch_size):
    memory_usage = monitor.get_memory_usage()
    
    if memory_usage['gpu_utilization'] > 0.9:
        return base_batch_size // 2
    elif memory_usage['gpu_utilization'] < 0.5:
        return base_batch_size * 2
    
    return base_batch_size
```

#### 3. Memory-Efficient Data Loading
```python
# Optimize data loading for mixed precision
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2
)
```

## Error Handling

### Gradient Overflow Handling

```python
# Automatic overflow detection and handling
config = MixedPrecisionConfig(
    handle_overflow=True,
    overflow_threshold=float('inf')
)

# In training loop
if manager.handle_overflow(optimizer):
    logger.warning("Gradient overflow detected, skipping step")
    continue
```

### Numerical Stability

```python
# Monitor numerical stability
def check_numerical_stability(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN detected in {name}")
            return False
        if torch.isinf(param).any():
            logger.error(f"Inf detected in {name}")
            return False
    return True

# Use in training loop
if not check_numerical_stability(model):
    logger.error("Numerical instability detected")
    break
```

### Recovery Mechanisms

```python
# Automatic recovery from overflow
def handle_overflow_recovery(manager, optimizer, loss):
    # Skip this step
    optimizer.zero_grad()
    
    # Reduce learning rate temporarily
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5
    
    # Log overflow
    manager.performance_tracker.record_overflow()
    
    return True
```

## Best Practices

### 1. Configuration Guidelines

```python
# Good practices for mixed precision
config = MixedPrecisionConfig(
    # Enable mixed precision
    enabled=True,
    dtype=torch.float16,
    
    # Conservative scaling
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    
    # Enable monitoring
    log_scaling=True,
    log_frequency=100,
    
    # Enable error handling
    handle_overflow=True
)
```

### 2. Learning Rate Scaling

```python
# Scale learning rate for mixed precision
def scale_learning_rate_for_mp(base_lr, scale_factor=1.0):
    """Scale learning rate for mixed precision training."""
    return base_lr * scale_factor

# Usage
mp_lr = scale_learning_rate_for_mp(1e-3, scale_factor=1.0)
optimizer = optim.Adam(model.parameters(), lr=mp_lr)
```

### 3. Model Preparation

```python
# Prepare model for mixed precision
def prepare_model_for_mixed_precision(model):
    """Prepare model for mixed precision training."""
    # Convert model to appropriate dtype
    model = model.half()  # Convert to FP16
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model

# Usage
model = prepare_model_for_mixed_precision(model)
```

### 4. Checkpointing

```python
# Save checkpoints with mixed precision state
def save_mixed_precision_checkpoint(trainer, epoch, path):
    trainer.save_checkpoint(epoch, path)
    
    # Scaler state is automatically saved if enabled
    if trainer.config.save_scaler_state:
        print(f"Scaler state saved with checkpoint")

def load_mixed_precision_checkpoint(trainer, path):
    trainer.load_checkpoint(path)
    
    # Scaler state is automatically loaded
    print(f"Scaler state loaded from checkpoint")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Gradient Overflow

```python
# Problem: Frequent gradient overflow
# Solution: Adjust scaling parameters

config = MixedPrecisionConfig(
    init_scale=2**15,  # Lower initial scale
    growth_factor=1.5,  # Slower growth
    backoff_factor=0.7,  # Slower backoff
    handle_overflow=True
)
```

#### 2. Numerical Instability

```python
# Problem: NaN or Inf values
# Solution: Check model and data

def debug_numerical_issues(model, data):
    # Check model parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN in {name}")
        if torch.isinf(param).any():
            print(f"Inf in {name}")
    
    # Check input data
    if torch.isnan(data).any():
        print("NaN in input data")
    if torch.isinf(data).any():
        print("Inf in input data")
```

#### 3. Performance Degradation

```python
# Problem: Mixed precision is slower
# Solution: Check GPU compatibility and configuration

def check_gpu_compatibility():
    if not torch.cuda.is_available():
        return False
    
    # Check for Tensor Cores (Volta+)
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:  # Pre-Volta
        print("Warning: GPU may not support efficient mixed precision")
        return False
    
    return True

# Use appropriate configuration
if check_gpu_compatibility():
    config = MixedPrecisionConfig(enabled=True, dtype=torch.float16)
else:
    config = MixedPrecisionConfig(enabled=False)  # Fallback to FP32
```

#### 4. Memory Issues

```python
# Problem: Still running out of memory
# Solution: Optimize memory usage

config = MixedPrecisionConfig(
    enabled=True,
    memory_efficient=True,
    cache_enabled=False,  # Disable cache
    pin_memory=False      # Disable pin memory
)

# Reduce batch size
batch_size = batch_size // 2

# Enable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
```

### Debugging Tools

```python
# Enable debugging for mixed precision
config = MixedPrecisionConfig(
    log_scaling=True,
    log_frequency=1,  # Log every step
    handle_overflow=True
)

# Monitor scaler state
def monitor_scaler_state(trainer):
    scaler_state = trainer.mp_manager.get_scaler_state()
    print(f"Scale: {scaler_state['scale']:.2e}")
    print(f"Growth tracker: {scaler_state['growth_tracker']}")
    print(f"Enabled: {scaler_state['enabled']}")
```

## Examples

### Example 1: Basic Mixed Precision Training

```python
from mixed_precision_training import MixedPrecisionTrainer, create_mixed_precision_config

# Create configuration
config = create_mixed_precision_config(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)

# Create trainer
trainer = MixedPrecisionTrainer(
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

### Example 2: Advanced Configuration

```python
# Advanced configuration with monitoring
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    log_scaling=True,
    log_frequency=100,
    save_scaler_state=True,
    handle_overflow=True
)

trainer = MixedPrecisionTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### Example 3: Custom Training Loop

```python
from mixed_precision_training import MixedPrecisionManager

# Create manager
manager = MixedPrecisionManager(config)

# Custom training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with manager.autocast_context():
            outputs = model(batch)
            loss = loss_fn(outputs, targets)
        
        # Scale loss and backward pass
        scaled_loss = manager.scale_loss(loss)
        scaled_loss.backward()
        
        # Handle overflow
        if manager.handle_overflow(optimizer):
            continue
        
        # Unscale and step
        manager.unscale_optimizer(optimizer)
        success = manager.step_optimizer(optimizer)
        
        if success:
            manager.update_scaler()
        
        # Update scheduler
        scheduler.step()
        
        # Log progress
        if batch_idx % 100 == 0:
            scaler_state = manager.get_scaler_state()
            print(f"Batch {batch_idx}: Scale={scaler_state['scale']:.2e}")
```

### Example 4: Performance Benchmarking

```python
from mixed_precision_training import benchmark_mixed_precision

# Benchmark different configurations
configs = [
    ("FP32", MixedPrecisionConfig(enabled=False)),
    ("FP16", MixedPrecisionConfig(enabled=True, dtype=torch.float16)),
    ("BF16", MixedPrecisionConfig(enabled=True, dtype=torch.bfloat16))
]

for name, config in configs:
    print(f"\nBenchmarking {name}:")
    results = benchmark_mixed_precision(model, train_loader, config)
    print(f"  Speedup: {results['speedup']:.2f}x")
    print(f"  Memory savings: {results['memory_savings']:.1f}%")
    print(f"  Training time: {results['mp_time']:.2f}s")
```

### Example 5: Memory-Efficient Training

```python
# Memory-efficient configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    memory_efficient=True,
    cache_enabled=False,
    pin_memory=False
)

# Enable gradient checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Use smaller batch size
train_loader = DataLoader(
    dataset,
    batch_size=16,  # Smaller batch size
    num_workers=2,
    pin_memory=False
)

trainer = MixedPrecisionTrainer(
    model=model,
    train_loader=train_loader,
    config=config,
    optimizer=optimizer
)
```

## Summary

This guide provides comprehensive coverage of mixed precision training implementation in the Video-OpusClip system. Key takeaways:

1. **Enable mixed precision**: Use FP16 for performance and memory savings
2. **Configure gradient scaling**: Set appropriate scaling parameters
3. **Monitor performance**: Track speedup and memory savings
4. **Handle errors**: Implement overflow detection and recovery
5. **Optimize memory**: Use memory-efficient configurations
6. **Benchmark results**: Compare FP32 vs mixed precision performance
7. **Save checkpoints**: Include scaler state in checkpoints

The mixed precision training system is designed to maximize performance while maintaining numerical stability and providing excellent developer experience. It supports both simple mixed precision setups and complex configurations with advanced monitoring and error handling. 