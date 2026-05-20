# Mixed Precision Training System

Comprehensive mixed precision training implementation using PyTorch's Automatic Mixed Precision (AMP) with `torch.cuda.amp` for improved training speed and reduced memory usage.

## Overview

The Mixed Precision Training System provides a complete solution for training deep learning models with automatic mixed precision, offering significant performance improvements while maintaining model accuracy.

### Key Features

- **Automatic Mixed Precision (AMP)**: Uses `torch.cuda.amp` for automatic precision management
- **Gradient Scaling**: Automatic gradient scaling to prevent underflow
- **Memory Optimization**: Reduced memory usage with float16/bfloat16 precision
- **Performance Monitoring**: Comprehensive tracking of training metrics
- **Compatibility Checking**: Automatic model compatibility validation
- **Integration Ready**: Seamless integration with existing training pipelines

## Benefits

### Performance Improvements
- **Training Speed**: 1.5x to 3x faster training on modern GPUs
- **Memory Usage**: 30-50% reduction in GPU memory consumption
- **Batch Size**: Ability to use larger batch sizes due to memory savings
- **Throughput**: Higher training throughput for faster experimentation

### Accuracy Preservation
- **Automatic Scaling**: Gradient scaling prevents numerical underflow
- **Overflow Detection**: Automatic detection and handling of gradient overflow
- **Fallback Support**: Graceful fallback to standard precision if needed
- **Validation**: Comprehensive validation of mixed precision compatibility

## Architecture

### Core Components

1. **MixedPrecisionConfig**: Configuration management for AMP settings
2. **MixedPrecisionTrainer**: Main training manager with AMP support
3. **MixedPrecisionOptimizer**: High-level optimizer wrapper
4. **Utility Functions**: Helper functions for setup and compatibility checking

### Integration Points

- **Optimized Training Optimizer**: Full integration with the main training pipeline
- **Multi-GPU Training**: Compatible with distributed training
- **Gradient Accumulation**: Works with gradient accumulation strategies
- **Performance Optimization**: Integrates with performance optimization tools

## Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from core.mixed_precision_training import (
    MixedPrecisionConfig, create_mixed_precision_trainer
)

# Create model and optimizer
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# Create mixed precision trainer
mp_trainer = create_mixed_precision_trainer(
    model=model,
    optimizer=optimizer,
    enable_amp=True,
    dtype=torch.float16,
    enable_grad_scaler=True
)

# Training loop
for batch in dataloader:
    inputs, targets = batch
    
    # Training step with mixed precision
    metrics = mp_trainer.train_step(
        (inputs, targets),
        loss_fn,
        device
    )
    
    print(f"Loss: {metrics['loss']:.6f}, "
          f"Accuracy: {metrics['accuracy']:.4f}, "
          f"Time: {metrics['step_time']:.4f}s")
```

### Advanced Configuration

```python
from core.mixed_precision_training import MixedPrecisionConfig

# Custom configuration
config = MixedPrecisionConfig(
    enable_amp=True,
    dtype=torch.float16,  # or torch.bfloat16
    enable_grad_scaler=True,
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enable_monitoring=True,
    log_amp_stats=True,
    track_memory_usage=True,
    validate_amp=True,
    check_compatibility=True
)

# Create trainer with custom config
mp_trainer = MixedPrecisionTrainer(
    config=config,
    model=model,
    optimizer=optimizer,
    logger=logger
)
```

### Integration with Optimized Training

```python
from core.optimized_training_optimizer import create_optimized_training_optimizer

# Create optimizer with mixed precision
optimizer = create_optimized_training_optimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    # Mixed precision configuration
    enable_amp=True,
    amp_dtype=torch.float16,
    enable_grad_scaler=True,
    amp_init_scale=2**16,
    amp_growth_factor=2.0,
    amp_backoff_factor=0.5,
    amp_growth_interval=2000,
    amp_monitoring=True,
    amp_log_stats=True,
    amp_track_memory=True,
    amp_validate=True,
    amp_check_compatibility=True
)

# Train with mixed precision
results = await optimizer.train()
```

## Configuration Options

### MixedPrecisionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_amp` | bool | True | Enable Automatic Mixed Precision |
| `dtype` | torch.dtype | torch.float16 | Precision for mixed precision |
| `enabled` | bool | True | Whether AMP is enabled (can be toggled) |
| `enable_grad_scaler` | bool | True | Enable gradient scaling |
| `init_scale` | float | 2^16 | Initial scale for gradient scaling |
| `growth_factor` | float | 2.0 | Factor to increase scale when no overflow |
| `backoff_factor` | float | 0.5 | Factor to decrease scale when overflow occurs |
| `growth_interval` | int | 2000 | Steps between scale increases |
| `cache_enabled` | bool | True | Enable AMP cache for performance |
| `autocast_enabled` | bool | True | Enable autocast context |
| `memory_efficient` | bool | False | Use memory-efficient mixed precision |
| `clear_cache` | bool | True | Clear CUDA cache periodically |
| `enable_monitoring` | bool | True | Enable mixed precision monitoring |
| `log_amp_stats` | bool | True | Log AMP statistics |
| `track_memory_usage` | bool | True | Track memory usage changes |
| `validate_amp` | bool | True | Validate AMP configuration |
| `check_compatibility` | bool | True | Check model compatibility with AMP |

## Monitoring and Statistics

### Available Metrics

The mixed precision trainer provides comprehensive statistics:

```python
# Get AMP statistics
stats = mp_trainer.get_amp_stats()

# Available metrics
print(f"Total steps: {stats['total_steps']}")
print(f"AMP steps: {stats['amp_steps']}")
print(f"Scaler updates: {stats['scaler_updates']}")
print(f"Overflow count: {stats['overflow_count']}")
print(f"Current scale: {stats['current_scale']}")
print(f"AMP usage ratio: {stats['amp_usage_ratio']:.3f}")
print(f"Overflow ratio: {stats['overflow_ratio']:.3f}")
print(f"Average memory savings: {stats['avg_memory_savings']:.3f} GB")
print(f"Average step time: {stats['avg_step_time']:.6f}s")
```

### Logging and Visualization

```python
# Save statistics to file
mp_trainer.save_amp_stats("logs/amp_stats.json")

# Enable detailed logging
config = MixedPrecisionConfig(
    log_amp_stats=True,
    track_memory_usage=True,
    enable_monitoring=True
)
```

## Compatibility and Validation

### Model Compatibility

```python
from core.mixed_precision_training import check_amp_compatibility

# Check if model is compatible
compatibility = check_amp_compatibility(model)

if compatibility["compatible"]:
    print("Model is compatible with mixed precision")
else:
    print(f"Compatibility issues: {compatibility['warnings']}")
```

### Common Compatibility Issues

1. **BatchNorm Layers**: May need attention for running statistics
2. **Custom Operations**: Some custom operations may not support mixed precision
3. **CUDA Availability**: Mixed precision works best with CUDA GPUs
4. **Model Architecture**: Some architectures may have compatibility issues

### Fallback Strategy

The system includes automatic fallback to standard precision:

```python
try:
    mp_trainer = create_mixed_precision_trainer(...)
except Exception as e:
    print("Mixed precision failed, falling back to standard precision")
    # Continue with standard precision training
```

## Performance Optimization

### Best Practices

1. **Use Appropriate Data Types**
   - `torch.float16` for most cases
   - `torch.bfloat16` for better numerical stability

2. **Monitor Gradient Scaling**
   - Watch for frequent scale reductions (indicates overflow)
   - Adjust `init_scale` and `growth_factor` if needed

3. **Memory Management**
   - Enable `clear_cache` for periodic memory cleanup
   - Monitor memory usage with `track_memory_usage=True`

4. **Batch Size Optimization**
   - Use larger batch sizes due to memory savings
   - Balance between memory usage and training stability

### Performance Tuning

```python
# Optimize for speed
config = MixedPrecisionConfig(
    dtype=torch.float16,
    cache_enabled=True,
    clear_cache=False,  # Disable for maximum speed
    memory_efficient=False
)

# Optimize for memory
config = MixedPrecisionConfig(
    dtype=torch.bfloat16,
    memory_efficient=True,
    clear_cache=True,
    track_memory_usage=True
)
```

## Troubleshooting

### Common Issues

1. **Gradient Overflow**
   ```
   Warning: Gradient overflow detected, scale reduced
   ```
   - Solution: Reduce learning rate or adjust scaler parameters

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - Solution: Enable `memory_efficient=True` or reduce batch size

3. **Compatibility Warnings**
   ```
   Mixed precision compatibility issues
   ```
   - Solution: Review model architecture and consider fallback

4. **Performance Degradation**
   - Solution: Check if AMP is actually enabled and monitor statistics

### Debug Mode

```python
# Enable detailed debugging
config = MixedPrecisionConfig(
    validate_amp=True,
    check_compatibility=True,
    log_amp_stats=True,
    track_memory_usage=True
)

# Monitor during training
stats = mp_trainer.get_amp_stats()
if stats['overflow_ratio'] > 0.1:
    print("High overflow ratio detected - consider adjusting parameters")
```

## Examples

### Complete Training Example

See `examples/mixed_precision_demo.py` for a comprehensive demonstration including:

- Speed benchmarking
- Memory usage comparison
- Detailed training monitoring
- Visualization generation
- Integration examples

### Running the Demo

```bash
# Run the comprehensive demo
python run_mixed_precision_demo.py

# Or run specific components
python -c "
from examples.mixed_precision_demo import demonstrate_mixed_precision_training
import asyncio
asyncio.run(demonstrate_mixed_precision_training())
"
```

## Integration with Other Systems

### Multi-GPU Training

```python
from core.multi_gpu_training import MultiGPUConfig

# Combine with multi-GPU training
multi_gpu_config = MultiGPUConfig(enable_multi_gpu=True)
mp_config = MixedPrecisionConfig(enable_amp=True)

# Both work together seamlessly
optimizer = create_optimized_training_optimizer(
    model=model,
    train_loader=train_loader,
    multi_gpu_config=multi_gpu_config,
    enable_amp=True,
    amp_dtype=torch.float16
)
```

### Gradient Accumulation

```python
from core.gradient_accumulation import GradientAccumulationConfig

# Works with gradient accumulation
grad_accum_config = GradientAccumulationConfig(accumulation_steps=4)
mp_config = MixedPrecisionConfig(enable_amp=True)

# Mixed precision handles accumulation internally
optimizer = create_optimized_training_optimizer(
    model=model,
    train_loader=train_loader,
    gradient_accumulation_config=grad_accum_config,
    enable_amp=True
)
```

## API Reference

### MixedPrecisionConfig

Configuration class for mixed precision training settings.

### MixedPrecisionTrainer

Main training manager with mixed precision support.

**Methods:**
- `train_step()`: Perform training step with mixed precision
- `validate_step()`: Perform validation step with mixed precision
- `get_amp_stats()`: Get comprehensive AMP statistics
- `enable_amp()`: Enable mixed precision training
- `disable_amp()`: Disable mixed precision training
- `toggle_amp()`: Toggle mixed precision on/off
- `reset_stats()`: Reset AMP statistics
- `save_amp_stats()`: Save statistics to file
- `cleanup()`: Cleanup resources

### MixedPrecisionOptimizer

High-level optimizer wrapper with mixed precision support.

**Methods:**
- `train_epoch()`: Train for one epoch with mixed precision
- `validate_epoch()`: Validate for one epoch with mixed precision
- `get_training_stats()`: Get training statistics
- `cleanup()`: Cleanup resources

### Utility Functions

- `create_mixed_precision_trainer()`: Create trainer with default settings
- `create_mixed_precision_optimizer()`: Create optimizer with default settings
- `check_amp_compatibility()`: Check model compatibility

## Performance Benchmarks

Typical performance improvements observed:

| Metric | Standard Precision | Mixed Precision | Improvement |
|--------|-------------------|-----------------|-------------|
| Training Time | 100% | 60-70% | 1.4-1.7x faster |
| Memory Usage | 100% | 50-70% | 30-50% reduction |
| Batch Size | 32 | 48-64 | 1.5-2x larger |
| Throughput | 100% | 140-170% | 1.4-1.7x higher |

*Results may vary depending on model architecture, hardware, and configuration.*

## Conclusion

The Mixed Precision Training System provides a robust, efficient, and easy-to-use solution for training deep learning models with automatic mixed precision. It offers significant performance improvements while maintaining model accuracy and providing comprehensive monitoring and debugging capabilities.

For more information, see the example demonstrations and API documentation. 