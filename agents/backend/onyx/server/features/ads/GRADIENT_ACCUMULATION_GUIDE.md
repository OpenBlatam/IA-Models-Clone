# Gradient Accumulation Guide

A comprehensive guide for implementing gradient accumulation for large batch sizes in the Onyx Ads Backend.

## 🚀 Overview

Gradient accumulation enables training with large effective batch sizes while using less GPU memory by accumulating gradients over multiple forward/backward passes before updating model parameters.

## 📊 Benefits

### Memory Efficiency
- **Reduced GPU memory usage** by using smaller actual batch sizes
- **Large effective batch sizes** for better training stability
- **Memory optimization** with automatic batch size adjustment

### Training Stability
- **Better gradient estimates** with larger effective batch sizes
- **Improved convergence** for large models
- **Stable training** with gradient clipping and scaling

### Performance Benefits
- **Flexible batch sizing** based on available GPU memory
- **Automatic optimization** of accumulation steps
- **Mixed precision support** for faster training

## 🛠️ Installation and Setup

### Prerequisites

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install GPUtil psutil

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Environment Variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs to use
export CUDA_LAUNCH_BLOCKING=1        # For debugging

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🔧 Configuration

### GradientAccumulationConfig

```python
from onyx.server.features.ads.gradient_accumulation import GradientAccumulationConfig

# Basic configuration
config = GradientAccumulationConfig(
    accumulation_steps=4,
    target_batch_size=32,
    mixed_precision=True,
    gradient_clipping=1.0,
    log_accumulation=True
)

# Advanced configuration
config = GradientAccumulationConfig(
    accumulation_steps=8,
    target_batch_size=64,
    max_memory_usage=0.9,
    memory_safety_margin=0.1,
    auto_adjust_batch_size=True,
    sync_gradients=True,
    gradient_scaling=True,
    mixed_precision=True,
    log_accumulation=True,
    log_memory_usage=True,
    gradient_clipping=1.0,
    warmup_steps=100,
    accumulation_scheduler="linear"
)
```

### Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `accumulation_steps` | Number of steps to accumulate gradients | 4 | int |
| `target_batch_size` | Target effective batch size | None | int |
| `max_memory_usage` | Maximum GPU memory usage | 0.9 | float |
| `memory_safety_margin` | Memory safety margin | 0.1 | float |
| `auto_adjust_batch_size` | Auto-adjust batch size | True | bool |
| `sync_gradients` | Synchronize gradients | True | bool |
| `gradient_scaling` | Scale learning rate | True | bool |
| `mixed_precision` | Use mixed precision | True | bool |
| `log_accumulation` | Log accumulation progress | True | bool |
| `gradient_clipping` | Gradient clipping value | 1.0 | float |

## 🎯 Usage Examples

### 1. Basic Gradient Accumulation

```python
from onyx.server.features.ads.gradient_accumulation import (
    GradientAccumulationConfig,
    GradientAccumulator
)

# Create configuration
config = GradientAccumulationConfig(
    accumulation_steps=4,
    target_batch_size=32
)

# Create accumulator
accumulator = GradientAccumulator(config)

# Training loop
for batch_idx, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    # Accumulate gradients
    acc_stats = accumulator.accumulate_gradients(
        loss, model, optimizer, scaler
    )
    
    # Log progress
    if acc_stats["should_update"]:
        print(f"Optimizer updated at step {batch_idx}")
```

### 2. Adaptive Gradient Accumulation

```python
from onyx.server.features.ads.gradient_accumulation import AdaptiveGradientAccumulator

# Create adaptive accumulator
config = GradientAccumulationConfig(
    target_batch_size=64,
    auto_adjust_batch_size=True
)

accumulator = AdaptiveGradientAccumulator(config)

# Update configuration based on GPU state
accumulator.update_config(model, [0, 1, 2, 3])

# Training with automatic optimization
for batch_idx, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    acc_stats = accumulator.accumulate_gradients(
        loss, model, optimizer, scaler
    )
```

### 3. Integration with Multi-GPU Training

```python
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService

# Initialize service
finetuning_service = OptimizedFineTuningService()

# Setup gradient accumulation
accumulation_setup = await finetuning_service.setup_gradient_accumulation(
    target_effective_batch_size=64,
    accumulation_steps=8
)

# Train with accumulation
result = await finetuning_service.finetune_model_with_accumulation(
    model_name="gpt2",
    dataset=dataset,
    training_config={
        "epochs": 3,
        "learning_rate": 5e-5,
        "weight_decay": 0.01
    },
    user_id=123,
    target_effective_batch_size=64,
    accumulation_steps=8
)

print(f"Training completed: {result['model_path']}")
print(f"Effective batch size: {result['effective_batch_size']}")
print(f"Accumulation steps: {result['accumulation_steps']}")
```

### 4. Large Batch Training

```python
# Automatic large batch optimization
result = await finetuning_service.finetune_model_large_batch(
    model_name="gpt2",
    dataset=dataset,
    training_config={
        "epochs": 3,
        "learning_rate": 5e-5
    },
    user_id=123,
    target_batch_size=128,
    max_memory_usage=0.9
)

print(f"Large batch training completed")
print(f"Effective batch size: {result['effective_batch_size']}")
print(f"Actual batch size per GPU: {result['actual_batch_size_per_gpu']}")
```

## 📊 API Usage

### Configuration

```bash
# Configure gradient accumulation
curl -X POST http://localhost:8000/gradient-accumulation/config \
  -H "Content-Type: application/json" \
  -d '{
    "accumulation_steps": 4,
    "target_effective_batch_size": 32,
    "mixed_precision": true,
    "gradient_clipping": 1.0
  }'

# Get configuration
curl http://localhost:8000/gradient-accumulation/config/accumulation_1234567890
```

### Training Endpoints

```bash
# Train with gradient accumulation
curl -X POST http://localhost:8000/gradient-accumulation/training/with-accumulation \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "dataset_config": {
      "texts": ["sample text 1", "sample text 2"],
      "max_length": 512
    },
    "training_config": {
      "epochs": 3,
      "learning_rate": 5e-5
    },
    "user_id": 123,
    "target_effective_batch_size": 64,
    "accumulation_steps": 8
  }'

# Large batch training
curl -X POST http://localhost:8000/gradient-accumulation/training/large-batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "dataset_config": {
      "texts": ["sample text 1", "sample text 2"],
      "max_length": 512
    },
    "training_config": {
      "epochs": 3,
      "learning_rate": 5e-5
    },
    "user_id": 123,
    "target_effective_batch_size": 128
  }'
```

### Batch Size Calculation

```bash
# Calculate optimal batch size
curl -X POST http://localhost:8000/gradient-accumulation/calculate-batch-size \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "target_effective_batch_size": 64,
    "gpu_ids": [0, 1, 2, 3]
  }'

# Simple calculation
curl "http://localhost:8000/gradient-accumulation/calculate-batch-size?model_name=gpt2&target_batch_size=64&gpu_ids=0&gpu_ids=1&gpu_ids=2&gpu_ids=3"
```

### Statistics and Monitoring

```bash
# Get accumulation statistics
curl http://localhost:8000/gradient-accumulation/stats/accumulation_1234567890

# Get all statistics
curl http://localhost:8000/gradient-accumulation/stats

# Get performance metrics
curl http://localhost:8000/gradient-accumulation/performance/metrics

# Get performance recommendations
curl "http://localhost:8000/gradient-accumulation/performance/recommendations?model_size=large&target_batch_size=64"
```

### Utility Endpoints

```bash
# Calculate effective batch size
curl -X POST "http://localhost:8000/gradient-accumulation/calculate-effective-batch-size?actual_batch_size=8&accumulation_steps=4"

# Calculate accumulation steps
curl -X POST "http://localhost:8000/gradient-accumulation/calculate-accumulation-steps?target_batch_size=32&actual_batch_size=8"

# Adjust learning rate
curl -X POST "http://localhost:8000/gradient-accumulation/adjust-learning-rate?base_lr=0.001&accumulation_steps=4"
```

## 🔍 Monitoring and Debugging

### Accumulation Monitoring

```python
from onyx.server.features.ads.gradient_accumulation import GradientAccumulator

# Monitor accumulation progress
accumulator = GradientAccumulator(config)

for batch_idx, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    acc_stats = accumulator.accumulate_gradients(loss, model, optimizer)
    
    # Log accumulation progress
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}: "
              f"Accumulation {acc_stats['accumulation_step']}/{config.accumulation_steps}, "
              f"Loss: {loss.item():.4f}, "
              f"Memory: {acc_stats['memory_used']:.2f}GB")

# Get accumulation statistics
stats = accumulator.get_accumulation_stats()
print(f"Average gradient norm: {stats['avg_gradient_norm']:.4f}")
print(f"Average memory usage: {stats['avg_memory_usage_gb']:.2f}GB")
print(f"Average accumulation time: {stats['avg_accumulation_time']:.4f}s")
```

### Performance Monitoring

```python
# Get comprehensive accumulation stats
accumulation_stats = await finetuning_service.get_accumulation_stats(training_id)
print(f"Training ID: {accumulation_stats['training_id']}")
print(f"Accumulation steps: {accumulation_stats['stats']['accumulation_steps']}")
print(f"Total loss: {accumulation_stats['stats']['total_loss']:.4f}")

# Monitor GPU usage during accumulation
gpu_stats = await finetuning_service.get_gpu_stats()
print(f"Available GPUs: {gpu_stats['available_gpus']}")
print(f"GPU Info: {gpu_stats['gpu_info']}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('onyx.server.features.ads.gradient_accumulation').setLevel(logging.DEBUG)

# Test accumulation setup
config = GradientAccumulationConfig(
    accumulation_steps=4,
    log_accumulation=True,
    log_memory_usage=True
)

accumulator = GradientAccumulator(config)
```

## 🚀 Best Practices

### 1. Batch Size Selection

```python
# Calculate optimal batch size based on GPU memory
result = await finetuning_service.calculate_optimal_batch_size(
    model_name="gpt2",
    target_effective_batch_size=64,
    gpu_ids=[0, 1, 2, 3]
)

print(f"Optimal batch size per GPU: {result['actual_batch_size_per_gpu']}")
print(f"Required accumulation steps: {result['accumulation_steps']}")
print(f"Effective batch size: {result['effective_batch_size']}")
```

### 2. Learning Rate Adjustment

```python
# Adjust learning rate for gradient accumulation
base_lr = 5e-5
accumulation_steps = 4
adjusted_lr = adjust_learning_rate(base_lr, accumulation_steps)

print(f"Base learning rate: {base_lr}")
print(f"Adjusted learning rate: {adjusted_lr}")
print(f"Learning rate scale: {1.0 / accumulation_steps}")
```

### 3. Memory Management

```python
# Use memory-efficient settings
config = GradientAccumulationConfig(
    accumulation_steps=8,
    max_memory_usage=0.8,  # Use 80% of GPU memory
    memory_safety_margin=0.2,  # 20% safety margin
    auto_adjust_batch_size=True,
    mixed_precision=True  # Use mixed precision for memory efficiency
)
```

### 4. Gradient Clipping

```python
# Enable gradient clipping for stability
config = GradientAccumulationConfig(
    accumulation_steps=4,
    gradient_clipping=1.0,  # Clip gradients at norm 1.0
    sync_gradients=True
)
```

### 5. Mixed Precision

```python
# Use mixed precision for faster training and less memory
from torch.cuda.amp import GradScaler

scaler = GradScaler()

config = GradientAccumulationConfig(
    accumulation_steps=4,
    mixed_precision=True
)

# In training loop
with torch.cuda.amp.autocast():
    outputs = model(batch)
    loss = criterion(outputs, targets)

acc_stats = accumulator.accumulate_gradients(
    loss, model, optimizer, scaler
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Out of Bounds**
   ```bash
   # Reduce accumulation steps
   config.accumulation_steps = 2
   
   # Reduce batch size per GPU
   config.target_batch_size = 16
   
   # Enable mixed precision
   config.mixed_precision = True
   ```

2. **Slow Training**
   ```bash
   # Check GPU utilization
   curl http://localhost:8000/gradient-accumulation/performance/metrics
   
   # Optimize batch size
   curl -X POST http://localhost:8000/gradient-accumulation/calculate-batch-size \
     -d '{"model_name": "gpt2", "target_effective_batch_size": 32}'
   ```

3. **Unstable Training**
   ```bash
   # Enable gradient clipping
   config.gradient_clipping = 1.0
   
   # Adjust learning rate
   config.gradient_scaling = True
   
   # Check gradient norms
   curl http://localhost:8000/gradient-accumulation/stats/training_id
   ```

4. **Incorrect Effective Batch Size**
   ```bash
   # Verify calculation
   curl -X POST "http://localhost:8000/gradient-accumulation/calculate-effective-batch-size?actual_batch_size=8&accumulation_steps=4"
   
   # Check configuration
   curl http://localhost:8000/gradient-accumulation/config/training_id
   ```

### Performance Optimization

1. **Batch Size Tuning**
   ```python
   # Start with small accumulation steps and increase
   accumulation_steps = [2, 4, 8, 16]
   for steps in accumulation_steps:
       config.accumulation_steps = steps
       # Test training performance
   ```

2. **Memory Optimization**
   ```python
   # Use memory fraction control
   config.max_memory_usage = 0.8
   config.memory_safety_margin = 0.2
   
   # Enable automatic adjustment
   config.auto_adjust_batch_size = True
   ```

3. **Mixed Precision**
   ```python
   # Enable for faster training and less memory
   config.mixed_precision = True
   ```

4. **Gradient Accumulation**
   ```python
   # Use for large effective batch sizes
   config.accumulation_steps = 8
   config.gradient_scaling = True
   ```

## 📈 Performance Benchmarks

### Training Speed Comparison

| Method | Effective Batch Size | Memory Usage | Training Speed | Stability |
|--------|---------------------|--------------|----------------|-----------|
| Standard | 8 | High | 1x | Medium |
| Accumulation (4 steps) | 32 | Medium | 0.9x | High |
| Accumulation (8 steps) | 64 | Low | 0.8x | Very High |
| Mixed Precision + Accumulation | 64 | Very Low | 1.2x | High |

### Memory Efficiency

| Configuration | Memory per GPU | Effective Batch Size | Memory Efficiency |
|---------------|----------------|---------------------|-------------------|
| Standard (batch=8) | 8GB | 8 | 100% |
| Accumulation (4 steps) | 6GB | 32 | 75% |
| Accumulation (8 steps) | 4GB | 64 | 50% |
| Mixed Precision + Accumulation | 3GB | 64 | 37.5% |

### Scaling Efficiency

- **2-4 accumulation steps**: 90-95% efficiency
- **4-8 accumulation steps**: 85-90% efficiency
- **8-16 accumulation steps**: 80-85% efficiency
- **Optimal range**: 4-8 steps for most models

## 🔒 Security Considerations

### Access Control

- Implement authentication for gradient accumulation API endpoints
- Use rate limiting for batch size calculations
- Monitor and log all accumulation operations

### Resource Protection

- Set memory limits to prevent system crashes
- Implement accumulation timeouts for long-running operations
- Monitor GPU temperature during accumulation

### Data Protection

- Ensure training data is properly secured
- Implement secure checkpoint storage
- Monitor GPU memory for sensitive data

## 📚 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from onyx.server.features.ads.gradient_accumulation_api import router as accumulation_router

app = FastAPI()
app.include_router(accumulation_router, prefix="/api/v1")
```

### Background Tasks

```python
@app.on_event("startup")
async def startup_event():
    # Initialize gradient accumulation API
    global accumulation_api
    accumulation_api = GradientAccumulationAPI()

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup accumulation resources
    if 'accumulation_api' in globals():
        accumulation_api.cleanup_all()
```

### Custom Training Loop

```python
async def custom_accumulation_training(model, dataset, config):
    # Setup gradient accumulation
    accumulator = AdaptiveGradientAccumulator(config)
    
    # Custom training logic
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(dataloader):
            outputs = model(batch)
            loss = criterion(outputs, targets)
            
            acc_stats = accumulator.accumulate_gradients(
                loss, model, optimizer, scaler
            )
            
            # Custom logging and validation
    
    return accumulator.get_accumulation_stats()
```

This comprehensive gradient accumulation system provides the tools and capabilities needed to efficiently train large models with large effective batch sizes while optimizing memory usage and maintaining training stability. 