# PyTorch Quick Reference for Video-OpusClip

A comprehensive guide to PyTorch features and optimizations available in the Video-OpusClip system.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Basic PyTorch Operations](#basic-pytorch-operations)
3. [Model Development](#model-development)
4. [Training Optimization](#training-optimization)
5. [Multi-GPU Training](#multi-gpu-training)
6. [Mixed Precision Training](#mixed-precision-training)
7. [Gradient Accumulation](#gradient-accumulation)
8. [Debugging & Monitoring](#debugging--monitoring)
9. [Performance Profiling](#performance-profiling)
10. [Production Deployment](#production-deployment)
11. [Video-OpusClip Integration](#video-opusclip-integration)

## Installation & Setup

### Basic PyTorch Installation

```bash
# CPU only
pip install torch torchvision torchaudio

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Video-OpusClip Dependencies

```bash
# Basic dependencies
pip install -r requirements_basic.txt

# Complete dependencies (includes all optimizations)
pip install -r requirements_complete.txt

# Automated installation
python install_dependencies.py
```

### Verify Installation

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
```

## Basic PyTorch Operations

### Tensor Operations

```python
import torch

# Create tensors
x = torch.randn(3, 3)
y = torch.zeros(3, 3)
z = torch.ones(3, 3)

# Basic operations
result = x + y
result = torch.mm(x, y)  # Matrix multiplication
result = x.mean()
result = x.sum()

# Move to GPU
if torch.cuda.is_available():
    x = x.cuda()
    # or
    x = x.to('cuda')
```

### Neural Network Basics

```python
import torch.nn as nn

# Simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Forward pass
input_tensor = torch.randn(5, 10)
output = model(input_tensor)

# Loss function
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# Backward pass
loss.backward()
```

## Model Development

### Custom Model

```python
class VideoModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Usage
model = VideoModel(input_size=100, hidden_size=50, num_classes=10)
```

### Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataloader
dataset = VideoDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Training Optimization

### Basic Training Loop

```python
import torch.optim as optim

# Setup
model = YourModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Using Video-OpusClip Optimized Trainer

```python
from optimized_training import OptimizedTrainer, TrainingConfig

# Configuration
config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    use_amp=True,  # Mixed precision
    gradient_accumulation_steps=4,
    early_stopping_patience=10
)

# Create trainer
trainer = OptimizedTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# Train
results = trainer.train()
```

## Multi-GPU Training

### DataParallel (Single Machine, Multiple GPUs)

```python
from multi_gpu_training import DataParallelTrainer, MultiGPUConfig

# Configuration
config = MultiGPUConfig(
    strategy='dataparallel',
    gpu_ids=[0, 1, 2, 3],
    batch_size=32
)

# Create trainer
trainer = DataParallelTrainer(
    model=model,
    config=config,
    train_loader=train_loader
)

# Train
results = trainer.train()
```

### DistributedDataParallel (Multi-Machine)

```python
from multi_gpu_training import DistributedDataParallelTrainer

# Configuration
config = MultiGPUConfig(
    strategy='distributed',
    world_size=4,
    backend='nccl'
)

# Create trainer
trainer = DistributedDataParallelTrainer(
    model=model,
    config=config,
    train_loader=train_loader
)

# Train
results = trainer.train()
```

## Mixed Precision Training

### Manual Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

# Setup
scaler = GradScaler()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(input)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Using Video-OpusClip Mixed Precision Trainer

```python
from mixed_precision_training import MixedPrecisionTrainer, MixedPrecisionConfig

# Configuration
config = MixedPrecisionConfig(
    enabled=True,
    dtype=torch.float16,
    init_scale=2**16
)

# Create trainer
trainer = MixedPrecisionTrainer(
    model=model,
    train_loader=train_loader,
    config=config
)

# Train
results = trainer.train()
```

## Gradient Accumulation

### Manual Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Using Video-OpusClip Gradient Accumulation

```python
from gradient_accumulation import GradientAccumulationTrainer, GradientAccumulationConfig

# Configuration
config = GradientAccumulationConfig(
    accumulation_steps=4,
    max_batch_size=128,
    use_amp=True
)

# Create trainer
trainer = GradientAccumulationTrainer(
    model=model,
    train_loader=train_loader,
    config=config
)

# Train
results = trainer.train()
```

## Debugging & Monitoring

### Autograd Anomaly Detection

```python
from pytorch_debug_tools import PyTorchDebugManager, PyTorchDebugConfig

# Configuration
config = PyTorchDebugConfig(
    enable_autograd_anomaly=True,
    enable_gradient_debugging=True,
    enable_memory_debugging=True
)

# Create debug manager
debug_manager = PyTorchDebugManager(config)

# Use in training
with debug_manager.anomaly_detector.detect_anomaly():
    loss = model(input)
    loss.backward()
```

### Memory Monitoring

```python
# Check GPU memory
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

# Clear cache
torch.cuda.empty_cache()
```

### Training Monitoring

```python
from training_logger import TrainingLogger

# Create logger
logger = TrainingLogger(
    log_dir="logs",
    experiment_name="video_training"
)

# Log metrics
logger.log_metrics({
    'loss': loss.item(),
    'accuracy': accuracy,
    'learning_rate': optimizer.param_groups[0]['lr']
})
```

## Performance Profiling

### Code Profiling

```python
from code_profiler import CodeProfiler, ProfilingConfig

# Configuration
config = ProfilingConfig(
    enable_performance_profiling=True,
    enable_memory_profiling=True,
    enable_gpu_profiling=True
)

# Create profiler
profiler = CodeProfiler(config)

# Profile training
with profiler.profile_training():
    trainer.train()

# Get results
results = profiler.get_profiling_results()
```

### GPU Profiling

```python
# Profile GPU operations
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Production Deployment

### Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model
torch.save(model, 'model_full.pth')
model = torch.load('model_full.pth')
```

### Model Optimization

```python
# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

### TorchServe Deployment

```python
# Create model archive
torch-model-archiver --model-name video_model \
                    --version 1.0 \
                    --model-file model.py \
                    --serialized-file model.pth \
                    --handler handler.py

# Start TorchServe
torchserve --start --model-store model-store --models video_model=video_model.mar
```

## Video-OpusClip Integration

### Using Video-OpusClip Components

```python
# Import Video-OpusClip components
from optimized_training import OptimizedTrainer
from mixed_precision_training import MixedPrecisionTrainer
from multi_gpu_training import DataParallelTrainer
from gradient_accumulation import GradientAccumulationTrainer
from pytorch_debug_tools import PyTorchDebugManager
from code_profiler import CodeProfiler
from training_logger import TrainingLogger

# Create comprehensive training setup
def create_video_opusclip_training(
    model,
    train_loader,
    val_loader=None,
    use_mixed_precision=True,
    use_multi_gpu=True,
    use_gradient_accumulation=True,
    enable_debugging=False
):
    """Create a comprehensive training setup with all Video-OpusClip optimizations."""
    
    # Choose trainer based on requirements
    if use_multi_gpu and torch.cuda.device_count() > 1:
        from multi_gpu_training import MultiGPUConfig
        config = MultiGPUConfig(
            strategy='dataparallel',
            gpu_ids=list(range(torch.cuda.device_count()))
        )
        trainer = DataParallelTrainer(model, config, train_loader, val_loader)
    elif use_mixed_precision:
        from mixed_precision_training import MixedPrecisionConfig
        config = MixedPrecisionConfig(enabled=True)
        trainer = MixedPrecisionTrainer(model, train_loader, val_loader, config)
    elif use_gradient_accumulation:
        from gradient_accumulation import GradientAccumulationConfig
        config = GradientAccumulationConfig(accumulation_steps=4)
        trainer = GradientAccumulationTrainer(model, train_loader, val_loader, config)
    else:
        from optimized_training import TrainingConfig
        config = TrainingConfig()
        trainer = OptimizedTrainer(model, train_loader, val_loader, config)
    
    # Add debugging if enabled
    if enable_debugging:
        debug_manager = PyTorchDebugManager()
        trainer.debug_manager = debug_manager
    
    # Add profiling
    profiler = CodeProfiler()
    trainer.profiler = profiler
    
    # Add logging
    logger = TrainingLogger()
    trainer.logger = logger
    
    return trainer

# Usage
trainer = create_video_opusclip_training(
    model=your_model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_mixed_precision=True,
    use_multi_gpu=True,
    use_gradient_accumulation=True,
    enable_debugging=True
)

# Train
results = trainer.train()
```

### Quick Start Scripts

```bash
# Run setup check
python torch_setup_check.py

# Quick start mixed precision
python quick_start_mixed_precision.py

# Quick start multi-GPU
python quick_start_multi_gpu.py

# Quick start gradient accumulation
python quick_start_gradient_accumulation.py

# Quick start PyTorch debugging
python quick_start_pytorch_debugging.py

# Quick start profiling
python quick_start_profiling.py
```

## Best Practices

### Performance Optimization

1. **Use Mixed Precision**: Always enable mixed precision for GPU training
2. **Gradient Accumulation**: Use for large effective batch sizes
3. **Multi-GPU**: Utilize all available GPUs
4. **Memory Management**: Monitor and optimize GPU memory usage
5. **Data Loading**: Use multiple workers and pin memory

### Debugging

1. **Enable Anomaly Detection**: Use `autograd.detect_anomaly()` during development
2. **Monitor Gradients**: Check for NaN or infinite gradients
3. **Memory Profiling**: Track memory usage patterns
4. **Performance Profiling**: Identify bottlenecks

### Production

1. **Model Optimization**: Use quantization and TorchScript
2. **Error Handling**: Implement robust error handling
3. **Logging**: Comprehensive logging for monitoring
4. **Testing**: Thorough testing before deployment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size, use gradient accumulation
2. **NaN Loss**: Check data, enable anomaly detection
3. **Slow Training**: Enable mixed precision, use multiple GPUs
4. **Import Errors**: Install missing dependencies

### Getting Help

1. Check the detailed guides in this directory
2. Run the setup checker: `python torch_setup_check.py`
3. Review the examples in the `examples/` directory
4. Check PyTorch documentation: https://pytorch.org/docs/

---

This quick reference covers the essential PyTorch features available in your Video-OpusClip system. For detailed information, refer to the specific guides and examples in this directory. 