# Mixed Precision Training Guide

A comprehensive guide for implementing mixed precision training with `torch.cuda.amp` in the Onyx Ads Backend.

## 🚀 Overview

Mixed precision training uses both FP16 (16-bit) and FP32 (32-bit) floating-point formats to accelerate training while maintaining numerical stability. This reduces memory usage and increases training speed.

## 📊 Benefits

### Performance Improvements
- **2x faster training** on modern GPUs with Tensor Cores
- **Reduced memory usage** by up to 50%
- **Larger batch sizes** possible with same memory
- **Faster model loading** and checkpointing

### Memory Efficiency
- **50% memory reduction** for model parameters
- **Reduced GPU memory bandwidth** usage
- **More efficient memory allocation**
- **Better cache utilization**

### Training Stability
- **Automatic gradient scaling** prevents underflow
- **Numerical stability** maintained
- **Compatible with existing optimizers**
- **Seamless integration** with gradient accumulation

## 🛠️ Installation and Setup

### Prerequisites

```bash
# PyTorch with CUDA support (required for AMP)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify AMP support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'AMP available: {torch.cuda.amp.autocast}')"
```

### Environment Variables

```bash
# Enable AMP optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_LAUNCH_BLOCKING=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🔧 Configuration

### MixedPrecisionConfig

```python
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionConfig

# Basic configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16,
    memory_efficient=True
)

# Advanced configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    autocast_enabled=True,
    scaler_enabled=True,
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    memory_efficient=True,
    cache_enabled=True,
    deterministic=False,
    log_precision=True,
    log_memory_usage=True,
    min_loss_scale=1e-4,
    max_loss_scale=2**16,
    loss_scale_window=1000,
    hysteresis=2
)
```

### Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `enabled` | Enable mixed precision | True | bool |
| `dtype` | Precision dtype | torch.float16 | torch.dtype |
| `autocast_enabled` | Enable autocast | True | bool |
| `scaler_enabled` | Enable gradient scaler | True | bool |
| `init_scale` | Initial loss scale | 2^16 | float |
| `growth_factor` | Scale growth factor | 2.0 | float |
| `backoff_factor` | Scale backoff factor | 0.5 | float |
| `memory_efficient` | Memory efficient mode | True | bool |
| `cache_enabled` | Enable autocast cache | True | bool |
| `log_precision` | Log precision info | True | bool |

## 🎯 Usage Examples

### 1. Basic Mixed Precision Training

```python
from onyx.server.features.ads.mixed_precision_training import (
    MixedPrecisionConfig,
    MixedPrecisionTrainer
)

# Create configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)

# Create trainer
trainer = MixedPrecisionTrainer(config)

# Training loop
for batch_idx, batch in enumerate(dataloader):
    # Forward pass with mixed precision
    outputs = trainer.forward_pass(model, batch)
    loss = criterion(outputs, targets)
    
    # Backward pass with mixed precision
    backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
    
    # Optimizer step with mixed precision
    optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
    
    # Log progress
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx}: Loss: {loss.item():.4f}, "
              f"Scale: {backward_stats['scaler_scale']:.2f}")
```

### 2. Adaptive Mixed Precision Training

```python
from onyx.server.features.ads.mixed_precision_training import AdaptiveMixedPrecisionTrainer

# Create adaptive trainer
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)

trainer = AdaptiveMixedPrecisionTrainer(config)

# Update configuration based on model and hardware
trainer.update_config_adaptive(model)

# Training with automatic optimization
for batch_idx, batch in enumerate(dataloader):
    outputs = trainer.forward_pass(model, batch)
    loss = criterion(outputs, targets)
    
    backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
    optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
```

### 3. Integration with Gradient Accumulation

```python
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionGradientAccumulator

# Create mixed precision accumulator
mp_config = MixedPrecisionConfig(enabled=True, dtype=torch.float16)
acc_config = GradientAccumulationConfig(accumulation_steps=4, mixed_precision=True)

accumulator = MixedPrecisionGradientAccumulator(mp_config, acc_config)

# Training with accumulation and mixed precision
for batch_idx, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    # Accumulate gradients with mixed precision
    acc_stats = accumulator.accumulate_gradients_mp(
        loss, model, optimizer, accumulator.mp_trainer.scaler
    )
    
    if acc_stats["should_update"]:
        print(f"Optimizer updated at step {batch_idx}")
```

### 4. Integration with Fine-tuning Service

```python
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService

# Initialize service
finetuning_service = OptimizedFineTuningService()

# Setup mixed precision
mp_setup = await finetuning_service.setup_mixed_precision(
    model_name="gpt2",
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16,
    memory_efficient=True
)

# Train with mixed precision
result = await finetuning_service.finetune_model_with_mixed_precision(
    model_name="gpt2",
    dataset=dataset,
    training_config={
        "epochs": 3,
        "learning_rate": 5e-5,
        "weight_decay": 0.01
    },
    user_id=123
)

print(f"Training completed: {result['model_path']}")
print(f"Memory savings: {result['memory_savings_gb']:.2f}GB")
print(f"Scaler scale: {result['scaler_scale']:.2f}")
```

### 5. Optimized Training (Mixed Precision + Gradient Accumulation)

```python
# Train with both optimizations
result = await finetuning_service.finetune_model_optimized(
    model_name="gpt2",
    dataset=dataset,
    training_config={
        "epochs": 3,
        "learning_rate": 5e-5
    },
    user_id=123,
    use_mixed_precision=True,
    use_gradient_accumulation=True,
    target_effective_batch_size=64
)

print(f"Optimized training completed")
print(f"Mixed precision: {result['mixed_precision_enabled']}")
print(f"Gradient accumulation: {result['gradient_accumulation_enabled']}")
print(f"Effective batch size: {result['effective_batch_size']}")
print(f"Memory savings: {result['memory_savings_gb']:.2f}GB")
```

## 📊 API Usage

### Setup Mixed Precision

```bash
# Setup mixed precision
curl -X POST http://localhost:8000/finetuning/setup-mixed-precision \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "enabled": true,
    "dtype": "float16",
    "init_scale": 65536,
    "memory_efficient": true
  }'

# Get mixed precision stats
curl http://localhost:8000/finetuning/mixed-precision-stats
```

### Training Endpoints

```bash
# Train with mixed precision
curl -X POST http://localhost:8000/finetuning/mixed-precision \
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
    "mixed_precision_config": {
      "enabled": true,
      "dtype": "float16",
      "init_scale": 65536
    }
  }'

# Optimized training
curl -X POST http://localhost:8000/finetuning/optimized \
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
    "use_mixed_precision": true,
    "use_gradient_accumulation": true,
    "target_effective_batch_size": 64
  }'
```

### Optimization Endpoints

```bash
# Optimize mixed precision settings
curl -X POST http://localhost:8000/finetuning/optimize-mixed-precision \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt2",
    "gpu_memory_gb": 8.0,
    "batch_size": 16
  }'
```

## 🔍 Monitoring and Debugging

### Mixed Precision Monitoring

```python
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionTrainer

# Monitor mixed precision training
trainer = MixedPrecisionTrainer(config)

for batch_idx, batch in enumerate(dataloader):
    outputs = trainer.forward_pass(model, batch)
    loss = criterion(outputs, targets)
    
    backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
    optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
    
    # Log mixed precision stats
    if batch_idx % 10 == 0:
        mp_stats = trainer.get_training_stats()
        print(f"Batch {batch_idx}: "
              f"Loss: {loss.item():.4f}, "
              f"Scale: {mp_stats['scaler_scale']:.2f}, "
              f"Memory saved: {mp_stats['memory_saved']:.2f}GB, "
              f"Overflow: {mp_stats['overflow_count']}, "
              f"Underflow: {mp_stats['underflow_count']}")
```

### Performance Monitoring

```python
# Get comprehensive mixed precision stats
mp_stats = await finetuning_service.get_mixed_precision_stats()
print(f"Mixed precision enabled: {mp_stats['enabled']}")
print(f"Scaler scale: {mp_stats['scaler_scale']:.2f}")
print(f"Memory savings: {mp_stats['memory_savings_gb']:.2f}GB")
print(f"Overflow count: {mp_stats['overflow_count']}")
print(f"Underflow count: {mp_stats['underflow_count']}")

# Optimize settings
optimization = await finetuning_service.optimize_mixed_precision_settings(
    model_name="gpt2",
    gpu_memory_gb=8.0,
    batch_size=16
)
print(f"Should use mixed precision: {optimization['should_use_mixed_precision']}")
print(f"Recommendations: {optimization['recommendations']}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('onyx.server.features.ads.mixed_precision_training').setLevel(logging.DEBUG)

# Test mixed precision setup
config = MixedPrecisionConfig(
    enabled=True,
    log_precision=True,
    log_memory_usage=True
)

trainer = MixedPrecisionTrainer(config)
```

## 🚀 Best Practices

### 1. Model Compatibility

```python
# Check if model supports mixed precision
def check_model_compatibility(model):
    """Check if model is compatible with mixed precision."""
    # Most models work with mixed precision
    # Some layers may need special handling
    incompatible_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # BatchNorm layers work better in FP32
            incompatible_layers.append(name)
    
    return len(incompatible_layers) == 0, incompatible_layers

# Use mixed precision only if compatible
is_compatible, incompatible = check_model_compatibility(model)
if is_compatible:
    config = MixedPrecisionConfig(enabled=True)
else:
    print(f"Model has incompatible layers: {incompatible}")
    config = MixedPrecisionConfig(enabled=False)
```

### 2. Loss Scale Management

```python
# Monitor loss scale
scaler = torch.cuda.amp.GradScaler()

for batch_idx, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    
    # Check scale
    current_scale = scaler.get_scale()
    if current_scale < 1e-4:
        print(f"Warning: Low loss scale detected: {current_scale}")
    elif current_scale > 2**16:
        print(f"Warning: High loss scale detected: {current_scale}")
    
    scaler.step(optimizer)
    scaler.update()
```

### 3. Memory Management

```python
# Use memory-efficient settings
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    memory_efficient=True,
    cache_enabled=False  # Disable cache for memory efficiency
)

# Monitor memory usage
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    print(f"Memory allocated: {memory_allocated:.2f}GB")
    print(f"Memory reserved: {memory_reserved:.2f}GB")
```

### 4. Gradient Clipping

```python
# Gradient clipping with mixed precision
scaler = torch.cuda.amp.GradScaler()

for batch_idx, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    
    # Unscale for gradient clipping
    scaler.unscale_(optimizer)
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step optimizer
    scaler.step(optimizer)
    scaler.update()
```

### 5. Checkpointing

```python
# Save checkpoint with mixed precision info
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'mixed_precision_enabled': True,
    'scaler_scale': scaler.get_scale()
}

torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
```

## 🔧 Troubleshooting

### Common Issues

1. **Loss Scale Too Low**
   ```python
   # Increase initial scale
   config = MixedPrecisionConfig(
       enabled=True,
       init_scale=2**20,  # Higher initial scale
       growth_factor=2.0,
       backoff_factor=0.5
   )
   ```

2. **Memory Issues**
   ```python
   # Use memory-efficient settings
   config = MixedPrecisionConfig(
       enabled=True,
       memory_efficient=True,
       cache_enabled=False
   )
   ```

3. **Training Instability**
   ```python
   # Reduce learning rate with mixed precision
   base_lr = 5e-5
   mp_lr = base_lr * 0.5  # Reduce learning rate
   
   optimizer = torch.optim.AdamW(model.parameters(), lr=mp_lr)
   ```

4. **Gradient Overflow**
   ```python
   # Monitor and handle overflow
   scaler = torch.cuda.amp.GradScaler()
   
   for batch_idx, batch in enumerate(dataloader):
       scaler.scale(loss).backward()
       
       if scaler.get_scale() < 1e-4:
           print("Gradient overflow detected, skipping step")
           scaler.update()
           continue
       
       scaler.step(optimizer)
       scaler.update()
   ```

### Performance Optimization

1. **Batch Size Tuning**
   ```python
   # Increase batch size with mixed precision
   base_batch_size = 8
   mp_batch_size = base_batch_size * 2  # Double batch size
   
   dataloader = DataLoader(dataset, batch_size=mp_batch_size)
   ```

2. **Model Optimization**
   ```python
   # Use mixed precision for specific layers
   for name, module in model.named_modules():
       if isinstance(module, nn.Linear):
           module.half()  # Convert to FP16
   ```

3. **Memory Optimization**
   ```python
   # Clear cache periodically
   if batch_idx % 100 == 0:
       torch.cuda.empty_cache()
   ```

## 📈 Performance Benchmarks

### Training Speed Comparison

| Method | Training Speed | Memory Usage | Batch Size | Stability |
|--------|----------------|--------------|------------|-----------|
| FP32 | 1x | 100% | 8 | High |
| FP16 (no scaling) | 1.5x | 50% | 16 | Low |
| Mixed Precision | 2x | 50% | 16 | High |
| Mixed Precision + Accumulation | 1.8x | 50% | 64 | Very High |

### Memory Efficiency

| Configuration | Memory per GPU | Batch Size | Memory Efficiency |
|---------------|----------------|------------|-------------------|
| FP32 | 8GB | 8 | 100% |
| Mixed Precision | 4GB | 16 | 50% |
| Mixed Precision + Accumulation | 4GB | 64 | 12.5% |

### Scaling Efficiency

- **Small models (<10M params)**: 1.5-2x speedup
- **Medium models (10M-100M params)**: 1.8-2.2x speedup
- **Large models (>100M params)**: 2-2.5x speedup
- **Optimal range**: Models with >10M parameters

## 🔒 Security Considerations

### Access Control

- Implement authentication for mixed precision API endpoints
- Use rate limiting for optimization requests
- Monitor and log all mixed precision operations

### Resource Protection

- Set memory limits to prevent system crashes
- Implement training timeouts for long-running operations
- Monitor GPU temperature during mixed precision training

### Data Protection

- Ensure training data is properly secured
- Implement secure checkpoint storage
- Monitor GPU memory for sensitive data

## 📚 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from onyx.server.features.ads.mixed_precision_training import MixedPrecisionTrainer

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Initialize mixed precision trainer
    global mp_trainer
    config = MixedPrecisionConfig(enabled=True)
    mp_trainer = MixedPrecisionTrainer(config)

@app.post("/train/mixed-precision")
async def train_with_mixed_precision(request: TrainingRequest):
    # Use mixed precision trainer
    result = await mp_trainer.train_epoch(...)
    return result
```

### Custom Training Loop

```python
async def custom_mixed_precision_training(model, dataset, config):
    # Setup mixed precision
    trainer = MixedPrecisionTrainer(config)
    
    # Custom training logic
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(dataloader):
            outputs = trainer.forward_pass(model, batch)
            loss = criterion(outputs, targets)
            
            backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
            optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
            
            # Custom logging and validation
    
    return trainer.get_training_stats()
```

This comprehensive mixed precision training system provides the tools and capabilities needed to accelerate training while maintaining numerical stability and reducing memory usage in the Onyx Ads Backend. 