# Multi-GPU Training Guide for Video-OpusClip

This guide provides comprehensive instructions for implementing and using multi-GPU training in the Video-OpusClip system using PyTorch's DataParallel and DistributedDataParallel.

## Table of Contents

1. [Overview](#overview)
2. [GPU Detection and Selection](#gpu-detection-and-selection)
3. [DataParallel Training](#dataparallel-training)
4. [DistributedDataParallel Training](#distributeddataparallel-training)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## Overview

The Video-OpusClip system supports two main multi-GPU training strategies:

- **DataParallel**: Simple multi-GPU training for single-node setups
- **DistributedDataParallel**: Advanced multi-GPU training for single/multi-node setups

### Key Features

- Automatic GPU detection and selection
- Memory-aware GPU allocation
- Mixed precision training support
- Comprehensive performance monitoring
- Error handling and recovery
- Checkpointing and resuming

## GPU Detection and Selection

### Automatic GPU Detection

```python
from multi_gpu_training import get_gpu_info, select_optimal_gpus

# Get comprehensive GPU information
gpu_info = get_gpu_info()
print(f"Available GPUs: {gpu_info['count']}")

# Select optimal GPUs based on memory requirements
optimal_gpus = select_optimal_gpus(num_gpus=4, min_memory_gb=8.0)
print(f"Selected GPUs: {optimal_gpus}")
```

### GPU Information Structure

```python
gpu_info = {
    'count': 4,  # Number of available GPUs
    'current': 0,  # Current GPU device
    'devices': [
        {
            'id': 0,
            'name': 'NVIDIA GeForce RTX 4090',
            'memory_total': 25769803776,  # 24GB
            'memory_allocated': 1073741824,  # 1GB
            'memory_cached': 2147483648,  # 2GB
            'compute_capability': '8.9',
            'multi_processor_count': 128
        }
    ],
    'memory': {
        0: {
            'total': 25769803776,
            'allocated': 1073741824,
            'cached': 2147483648,
            'free': 23622320128
        }
    },
    'capabilities': {
        0: {
            'compute_capability': '8.9',
            'multi_processor_count': 128,
            'max_threads_per_block': 1024,
            'max_shared_memory_per_block': 49152
        }
    }
}
```

## DataParallel Training

DataParallel is suitable for single-node multi-GPU training with 2-4 GPUs.

### Basic Setup

```python
from multi_gpu_training import MultiGPUConfig, DataParallelTrainer

# Configure DataParallel training
config = MultiGPUConfig(
    strategy='dataparallel',
    num_gpus=4,
    batch_size=32,
    num_workers=8
)

# Create trainer
trainer = DataParallelTrainer(
    model=your_model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_function
)

# Training loop
for epoch in range(num_epochs):
    train_metrics = trainer.train_epoch(epoch)
    val_metrics = trainer.validate(epoch)
    
    print(f"Epoch {epoch}: {train_metrics}, {val_metrics}")
```

### Enhanced DataParallel Features

```python
class OptimizedDataParallel(nn.DataParallel):
    """Enhanced DataParallel with monitoring and error handling."""
    
    def get_device_memory_usage(self):
        """Get memory usage for each device."""
        memory_info = {}
        for device_id in self.device_ids:
            memory_info[device_id] = {
                'allocated': torch.cuda.memory_allocated(device_id) / 1024**3,
                'cached': torch.cuda.memory_reserved(device_id) / 1024**3,
                'total': torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            }
        return memory_info

# Usage
model = OptimizedDataParallel(model, device_ids=[0, 1, 2, 3])
memory_stats = model.get_device_memory_usage()
print(f"Memory usage: {memory_stats}")
```

### DataParallel Best Practices

1. **Batch Size**: Increase batch size proportionally to GPU count
2. **Data Loading**: Use multiple workers for data loading
3. **Memory Management**: Monitor memory usage across devices
4. **Gradient Synchronization**: DataParallel handles this automatically

```python
# Optimal batch size calculation
def get_optimal_batch_size(model, num_gpus, memory_per_sample=0.1):
    gpu_info = get_gpu_info()
    min_memory = min(gpu_info['memory'][gpu_id]['free'] for gpu_id in range(num_gpus))
    available_memory = min_memory * 0.7  # 70% of free memory
    batch_size = int(available_memory / memory_per_sample)
    return max(1, min(batch_size, 512))

# Usage
optimal_batch_size = get_optimal_batch_size(model, 4)
print(f"Optimal batch size: {optimal_batch_size}")
```

## DistributedDataParallel Training

DistributedDataParallel is suitable for both single-node and multi-node training.

### Single-Node DDP Setup

```python
from multi_gpu_training import (
    MultiGPUConfig, DistributedDataParallelTrainer,
    launch_distributed_training
)

# Configure DDP training
config = MultiGPUConfig(
    strategy='distributed',
    world_size=4,  # Number of GPUs
    backend='nccl',
    master_addr='localhost',
    master_port='12355'
)

# Define model and dataset functions
def create_model():
    return YourModel()

def create_datasets():
    train_dataset = YourTrainDataset()
    val_dataset = YourValDataset()
    return train_dataset, val_dataset

# Launch distributed training
launch_distributed_training(
    rank=0,  # Will be set automatically by mp.spawn
    world_size=4,
    model_fn=create_model,
    dataset_fn=create_datasets,
    config=config,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3
)
```

### Multi-Node DDP Setup

```python
# Node 0 (Master)
config = MultiGPUConfig(
    strategy='distributed',
    world_size=8,  # Total GPUs across all nodes
    rank=0,
    local_rank=0,
    backend='nccl',
    master_addr='192.168.1.100',  # Master node IP
    master_port='12355'
)

# Node 1 (Worker)
config = MultiGPUConfig(
    strategy='distributed',
    world_size=8,
    rank=4,  # Global rank
    local_rank=0,
    backend='nccl',
    master_addr='192.168.1.100',
    master_port='12355'
)
```

### DDP Advanced Configuration

```python
config = MultiGPUConfig(
    strategy='distributed',
    world_size=4,
    backend='nccl',
    
    # Performance optimizations
    find_unused_parameters=False,
    broadcast_buffers=True,
    bucket_cap_mb=25,
    gradient_as_bucket_view=False,
    static_graph=False,
    
    # Memory optimization
    memory_efficient_find_unused_parameters=False,
    
    # Data loading
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)
```

## Performance Optimization

### Memory Optimization

```python
# Memory-efficient training
def optimize_memory_usage(model, config):
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimize data loading
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return scaler, train_loader
```

### Load Balancing

```python
# Automatic load balancing
def balance_workload(num_gpus, dataset_size):
    samples_per_gpu = dataset_size // num_gpus
    remainder = dataset_size % num_gpus
    
    distribution = [samples_per_gpu] * num_gpus
    for i in range(remainder):
        distribution[i] += 1
    
    return distribution

# Usage
distribution = balance_workload(4, 10000)
print(f"Workload distribution: {distribution}")
```

### Communication Optimization

```python
# Optimize DDP communication
def optimize_ddp_communication(model, config):
    # Use gradient bucketing
    model = DDP(
        model,
        bucket_cap_mb=config.bucket_cap_mb,
        gradient_as_bucket_view=config.gradient_as_bucket_view
    )
    
    # Enable static graph optimization
    if config.static_graph:
        model._set_static_graph()
    
    return model
```

## Best Practices

### 1. GPU Selection

```python
# Always check GPU availability
gpu_info = get_gpu_info()
if gpu_info['count'] == 0:
    raise RuntimeError("No CUDA devices available")

# Select GPUs with sufficient memory
optimal_gpus = select_optimal_gpus(
    num_gpus=4,
    min_memory_gb=8.0
)

if len(optimal_gpus) < 4:
    logger.warning(f"Only {len(optimal_gpus)} GPUs meet memory requirements")
```

### 2. Batch Size Optimization

```python
# Calculate optimal batch size
def calculate_optimal_batch_size(model, num_gpus):
    # Estimate memory per sample
    sample_memory = estimate_sample_memory(model)
    
    # Get available memory
    gpu_info = get_gpu_info()
    available_memory = min(
        gpu_info['memory'][gpu_id]['free'] 
        for gpu_id in range(num_gpus)
    )
    
    # Calculate batch size
    batch_size = int(available_memory * 0.7 / sample_memory)
    
    # Ensure reasonable bounds
    batch_size = max(1, min(batch_size, 512))
    
    return batch_size
```

### 3. Data Loading Optimization

```python
# Optimize data loading for multi-GPU
def create_optimized_dataloader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=2,
        drop_last=True  # Important for DDP
    )
```

### 4. Mixed Precision Training

```python
# Enable mixed precision for better performance
def setup_mixed_precision_training():
    scaler = torch.cuda.amp.GradScaler()
    
    def training_step(model, data, target, optimizer):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_fn(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return loss
    
    return training_step, scaler
```

### 5. Checkpointing

```python
# Save and load checkpoints for multi-GPU training
def save_checkpoint(trainer, epoch, path):
    if hasattr(trainer, 'rank') and trainer.rank == 0:
        # Only save on rank 0 for DDP
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.module.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'scaler_state_dict': trainer.scaler.state_dict(),
            'config': trainer.config
        }
        torch.save(checkpoint, path)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=f'cuda:{trainer.rank}')
    trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if trainer.scheduler and checkpoint['scheduler_state_dict']:
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch']
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```python
# Solution: Reduce batch size or enable gradient checkpointing
def handle_oom_error(model, config):
    # Reduce batch size
    config.batch_size = config.batch_size // 2
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    return config
```

#### 2. NCCL Communication Errors

```python
# Solution: Check network configuration and firewall settings
def troubleshoot_nccl_errors():
    # Check if NCCL is available
    if not torch.distributed.is_nccl_available():
        logger.error("NCCL not available, falling back to Gloo")
        return 'gloo'
    
    # Check network connectivity
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 12355))
        sock.close()
    except:
        logger.error("Cannot connect to master port")
        return 'gloo'
    
    return 'nccl'
```

#### 3. DataParallel Synchronization Issues

```python
# Solution: Ensure proper data distribution
def fix_dataparallel_sync(model, data):
    # Ensure data is on the correct device
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Ensure batch size is consistent
    if data.size(0) % torch.cuda.device_count() != 0:
        # Pad or truncate to make divisible
        pad_size = torch.cuda.device_count() - (data.size(0) % torch.cuda.device_count())
        data = torch.cat([data, data[:pad_size]], dim=0)
    
    return data
```

#### 4. Performance Degradation

```python
# Solution: Profile and optimize bottlenecks
def profile_multi_gpu_performance(trainer):
    import time
    
    # Profile training step
    start_time = time.time()
    metrics = trainer.train_epoch(0)
    end_time = time.time()
    
    training_time = end_time - start_time
    samples_per_second = len(trainer.train_loader.dataset) / training_time
    
    logger.info(f"Training time: {training_time:.2f}s")
    logger.info(f"Samples per second: {samples_per_second:.2f}")
    
    # Check GPU utilization
    memory_stats = trainer.get_memory_stats()
    logger.info(f"Memory usage: {memory_stats}")
    
    return {
        'training_time': training_time,
        'samples_per_second': samples_per_second,
        'memory_stats': memory_stats
    }
```

## Examples

### Complete Multi-GPU Training Example

```python
from multi_gpu_training import (
    MultiGPUConfig, MultiGPUTrainingManager,
    launch_multi_gpu_training
)

# Define your model and dataset
class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Configuration
config = MultiGPUConfig(
    strategy='auto',  # Will choose DataParallel for 2-4 GPUs, DDP for more
    num_gpus=4,
    batch_size=32,
    num_workers=8,
    pin_memory=True
)

# Model and dataset functions
def create_model():
    return VideoModel()

def create_datasets():
    # Create your datasets here
    train_dataset = YourVideoDataset(train=True)
    val_dataset = YourVideoDataset(train=False)
    return train_dataset, val_dataset

# Launch training
launch_multi_gpu_training(
    model_fn=create_model,
    dataset_fn=create_datasets,
    config=config,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5
)
```

### Benchmarking Multi-GPU Performance

```python
from multi_gpu_training import benchmark_multi_gpu_performance

# Benchmark different configurations
configs = [
    MultiGPUConfig(strategy='dataparallel', num_gpus=2),
    MultiGPUConfig(strategy='dataparallel', num_gpus=4),
    MultiGPUConfig(strategy='distributed', world_size=4)
]

for config in configs:
    print(f"\nBenchmarking {config.strategy} with {config.num_gpus} GPUs:")
    
    model = create_model()
    dataset = create_datasets()[0]  # Training dataset
    
    results = benchmark_multi_gpu_performance(model, dataset, config)
    
    print(f"Training time: {results['training_time']:.2f}s")
    print(f"Samples per second: {results['samples_per_second']:.2f}")
    print(f"GPU utilization: {results['gpu_utilization']}")
```

### Custom Training Loop with Multi-GPU

```python
from multi_gpu_training import MultiGPUTrainingManager

# Create manager
manager = MultiGPUTrainingManager(config)

# Create trainer
trainer = manager.create_trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# Custom training loop
for epoch in range(num_epochs):
    # Training
    train_metrics = trainer.train_epoch(epoch)
    
    # Validation
    val_metrics = trainer.validate(epoch)
    
    # Logging
    if hasattr(trainer, 'rank') and trainer.rank == 0:
        logger.info(f"Epoch {epoch}: {train_metrics}, {val_metrics}")
    
    # Save checkpoint
    if epoch % 10 == 0:
        trainer.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
```

## Summary

This guide provides comprehensive coverage of multi-GPU training implementation in the Video-OpusClip system. Key takeaways:

1. **Choose the right strategy**: DataParallel for 2-4 GPUs, DDP for more
2. **Optimize memory usage**: Use mixed precision and gradient checkpointing
3. **Monitor performance**: Track GPU utilization and memory usage
4. **Handle errors gracefully**: Implement proper error handling and recovery
5. **Benchmark configurations**: Test different setups for optimal performance

The multi-GPU training system is designed to be production-ready, scalable, and easy to use while providing maximum performance for video processing tasks. 