# Multi-GPU Training System

Comprehensive multi-GPU training support using DataParallel and DistributedDataParallel for both single-machine multi-GPU and distributed multi-node training.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Training Modes](#training-modes)
8. [GPU Monitoring](#gpu-monitoring)
9. [Performance Optimization](#performance-optimization)
10. [Distributed Training](#distributed-training)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

## Overview

The Multi-GPU Training System provides comprehensive support for training email sequence models across multiple GPUs using PyTorch's DataParallel and DistributedDataParallel. It includes automatic detection of available GPUs, performance monitoring, and seamless integration with the existing training pipeline.

### Key Benefits

- **Automatic GPU Detection**: Automatically detects and utilizes available GPUs
- **Multiple Training Modes**: Support for DataParallel, DistributedDataParallel, and single GPU training
- **Performance Monitoring**: Real-time GPU utilization and memory tracking
- **Seamless Integration**: Works with existing training optimizers and logging systems
- **Error Handling**: Comprehensive error handling and recovery mechanisms

## Features

### Core Features

- **DataParallel Support**: Single-machine multi-GPU training
- **DistributedDataParallel Support**: Multi-node distributed training
- **Automatic Mode Detection**: Automatically selects the best training mode
- **GPU Monitoring**: Real-time GPU metrics tracking
- **Performance Optimization**: Integrated with performance optimization system
- **Checkpoint Management**: Proper handling of model checkpoints for multi-GPU setups

### Advanced Features

- **Synchronized Batch Normalization**: Support for SyncBatchNorm in distributed training
- **Gradient Bucketing**: Optimized gradient communication in distributed training
- **Memory Optimization**: Efficient memory usage across multiple GPUs
- **Load Balancing**: Automatic load balancing across available GPUs
- **Fault Tolerance**: Error recovery and graceful degradation

## Architecture

### Core Components

```
Multi-GPU Training System
├── MultiGPUTrainer (Main orchestrator)
├── DataParallelManager (DataParallel handling)
├── DistributedManager (DistributedDataParallel handling)
├── GPUMonitor (GPU monitoring and metrics)
└── MultiGPUConfig (Configuration management)
```

### Integration Points

- **Training Optimizer**: Integrated with OptimizedTrainingOptimizer
- **Performance System**: Works with PerformanceOptimizer
- **Logging System**: Integrated with TrainingLogger
- **Error Handling**: Uses ErrorHandler for robust error management

## Installation

### Prerequisites

```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies
pip install numpy psutil
```

### System Requirements

- **CUDA**: Version 11.0 or higher
- **PyTorch**: Version 1.12 or higher
- **Python**: Version 3.8 or higher
- **GPUs**: Multiple CUDA-compatible GPUs for multi-GPU training

## Quick Start

### Basic Usage

```python
import torch
from core.multi_gpu_training import create_multi_gpu_trainer, MultiGPUConfig
from core.optimized_training_optimizer import create_optimized_training_optimizer

# Create model and data loaders
model = YourModel()
train_loader = DataLoader(train_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)

# Configure multi-GPU training
multi_gpu_config = MultiGPUConfig(
    training_mode="auto",
    enable_data_parallel=True,
    enable_distributed=False,
    enable_gpu_monitoring=True
)

# Create optimized training optimizer with multi-GPU support
optimizer = create_optimized_training_optimizer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    multi_gpu_config=multi_gpu_config,
    max_epochs=10,
    learning_rate=0.001
)

# Train model
results = await optimizer.train()
```

### Automatic GPU Detection

```python
from core.multi_gpu_training import optimize_model_for_multi_gpu

# Automatically optimize model for multi-GPU training
optimized_model, trainer = optimize_model_for_multi_gpu(
    model=model,
    training_mode="auto",
    enable_data_parallel=True,
    enable_distributed=False
)

# Get training information
training_info = trainer.get_training_info()
print(f"Training mode: {training_info['training_mode']}")
print(f"GPU count: {training_info['gpu_info']['device_count']}")
```

## Configuration

### MultiGPUConfig

```python
@dataclass
class MultiGPUConfig:
    # Training mode
    training_mode: str = "auto"  # "auto", "single_gpu", "data_parallel", "distributed"
    
    # DataParallel settings
    enable_data_parallel: bool = True
    device_ids: Optional[List[int]] = None  # None for all available GPUs
    
    # Distributed settings
    enable_distributed: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    init_method: str = "env://"
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1
    
    # Communication settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = False
    
    # Performance settings
    enable_gradient_as_bucket_view: bool = False
    enable_find_unused_parameters: bool = False
    
    # Monitoring
    enable_gpu_monitoring: bool = True
    sync_bn: bool = False  # Synchronize batch normalization
```

### Configuration Examples

#### DataParallel Configuration

```python
config = MultiGPUConfig(
    training_mode="data_parallel",
    enable_data_parallel=True,
    enable_distributed=False,
    device_ids=[0, 1, 2],  # Use specific GPUs
    enable_gpu_monitoring=True
)
```

#### DistributedDataParallel Configuration

```python
config = MultiGPUConfig(
    training_mode="distributed",
    enable_data_parallel=False,
    enable_distributed=True,
    backend="nccl",
    sync_bn=True,
    find_unused_parameters=False,
    broadcast_buffers=True
)
```

#### Auto Detection Configuration

```python
config = MultiGPUConfig(
    training_mode="auto",
    enable_data_parallel=True,
    enable_distributed=True,
    enable_gpu_monitoring=True
)
```

## Training Modes

### 1. Single GPU Training

Used when only one GPU is available or explicitly configured.

```python
config = MultiGPUConfig(
    training_mode="single_gpu",
    enable_data_parallel=False,
    enable_distributed=False
)
```

### 2. DataParallel Training

Single-machine multi-GPU training using PyTorch's DataParallel.

**Advantages:**
- Simple setup
- Automatic data distribution
- No additional processes required

**Disadvantages:**
- Single process bottleneck
- Limited scalability
- No gradient accumulation across GPUs

```python
config = MultiGPUConfig(
    training_mode="data_parallel",
    enable_data_parallel=True,
    enable_distributed=False,
    device_ids=None  # Use all available GPUs
)
```

### 3. DistributedDataParallel Training

Multi-node distributed training using PyTorch's DistributedDataParallel.

**Advantages:**
- True distributed training
- Better scalability
- Gradient accumulation across nodes
- Better memory efficiency

**Disadvantages:**
- More complex setup
- Requires process management
- Network communication overhead

```python
config = MultiGPUConfig(
    training_mode="distributed",
    enable_data_parallel=False,
    enable_distributed=True,
    backend="nccl",
    sync_bn=True
)
```

### 4. Auto Detection

Automatically selects the best training mode based on available resources.

```python
config = MultiGPUConfig(
    training_mode="auto",
    enable_data_parallel=True,
    enable_distributed=True
)
```

**Auto Detection Logic:**
1. If no CUDA available → Single GPU (CPU)
2. If 1 GPU available → Single GPU
3. If multiple GPUs available and distributed enabled → DistributedDataParallel
4. If multiple GPUs available and data parallel enabled → DataParallel
5. Otherwise → Single GPU

## GPU Monitoring

### Real-time Monitoring

The system provides comprehensive GPU monitoring capabilities:

```python
# Get GPU information
gpu_info = trainer.gpu_monitor.get_gpu_info()
print(f"GPU count: {gpu_info['device_count']}")
print(f"Current device: {gpu_info['current_device']}")

# Record GPU metrics during training
trainer.record_gpu_metrics()

# Get monitoring summary
summary = trainer.gpu_monitor.get_gpu_summary()
print(f"Average memory usage: {summary['memory_allocated_mean']}")
print(f"Peak memory usage: {summary['memory_allocated_max']}")
```

### Monitored Metrics

- **Memory Usage**: Allocated and reserved memory
- **GPU Utilization**: GPU compute utilization
- **Temperature**: GPU temperature (if available)
- **Power Usage**: GPU power consumption (if available)

### Monitoring Integration

GPU monitoring is automatically integrated into the training loop:

```python
# During training, metrics are automatically recorded
async def train_epoch(self, epoch: int):
    for batch in self.train_loader:
        # Train batch
        metrics = self._train_batch(batch)
        
        # GPU metrics are automatically recorded
        self.multi_gpu_trainer.record_gpu_metrics()
        
        # Log metrics
        self.logger.log_batch(metrics)
```

## Performance Optimization

### Memory Optimization

The system includes several memory optimization techniques:

1. **Gradient Checkpointing**: Reduces memory usage at the cost of computation
2. **Mixed Precision Training**: Uses FP16 to reduce memory usage
3. **Dynamic Batching**: Adjusts batch size based on available memory
4. **Memory Pinning**: Optimizes CPU-GPU data transfer

### Computational Optimization

1. **Model Compilation**: Uses PyTorch 2.0 compilation for faster execution
2. **Fused Optimizers**: Uses fused optimizers for better performance
3. **Efficient Data Loading**: Multi-worker data loading with memory pinning

### Performance Monitoring

```python
# Get performance summary
performance_summary = trainer.get_performance_summary()
print(f"Training throughput: {performance_summary['throughput']} samples/s")
print(f"Memory efficiency: {performance_summary['memory_efficiency']}")

# Benchmark performance
benchmark_results = optimizer.benchmark_performance(num_iterations=100)
print(f"Average forward time: {benchmark_results['average_forward_time']}")
print(f"Average backward time: {benchmark_results['average_backward_time']}")
```

## Distributed Training

### Setup

For distributed training, you need to set up the environment:

```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2
export RANK=0

# Launch distributed training
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 your_training_script.py
```

### Distributed Training Script

```python
import os
import torch.distributed as dist
from core.multi_gpu_training import MultiGPUConfig

def setup_distributed():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    # Set device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return local_rank

def main():
    # Setup distributed training
    local_rank = setup_distributed()
    
    # Configure multi-GPU training
    config = MultiGPUConfig(
        training_mode="distributed",
        enable_distributed=True,
        backend="nccl",
        sync_bn=True
    )
    
    # Create trainer and train
    trainer = create_multi_gpu_trainer(config=config)
    model = YourModel()
    optimized_model = trainer.initialize_training(model)
    
    # Train model
    results = await trainer.train()
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Multi-Node Training

For multi-node training, you need to configure the network:

```bash
# Node 0
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=0

# Node 1
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=1

# Launch on each node
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=12355 your_training_script.py
```

## Examples

### Basic Multi-GPU Training

```python
import asyncio
import torch
from torch.utils.data import DataLoader
from core.multi_gpu_training import MultiGPUConfig
from core.optimized_training_optimizer import create_optimized_training_optimizer

async def train_email_sequence_model():
    # Create model and data
    model = EmailSequenceModel()
    train_dataset = EmailSequenceDataset(num_samples=10000)
    val_dataset = EmailSequenceDataset(num_samples=2000)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Configure multi-GPU training
    multi_gpu_config = MultiGPUConfig(
        training_mode="auto",
        enable_data_parallel=True,
        enable_distributed=False,
        enable_gpu_monitoring=True
    )
    
    # Create optimizer
    optimizer = create_optimized_training_optimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="email_sequence_multi_gpu",
        multi_gpu_config=multi_gpu_config,
        max_epochs=10,
        learning_rate=0.001,
        early_stopping_patience=5
    )
    
    # Train model
    results = await optimizer.train()
    
    # Get performance summary
    performance_summary = optimizer.multi_gpu_trainer.get_performance_summary()
    print(f"Training completed: {results}")
    print(f"Performance summary: {performance_summary}")
    
    return results

# Run training
asyncio.run(train_email_sequence_model())
```

### Advanced Distributed Training

```python
import os
import torch.distributed as dist
from core.multi_gpu_training import MultiGPUConfig, create_multi_gpu_trainer

def main_worker(local_rank, world_size):
    # Setup distributed training
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # Configure distributed training
    config = MultiGPUConfig(
        training_mode="distributed",
        enable_distributed=True,
        backend="nccl",
        sync_bn=True,
        find_unused_parameters=False,
        broadcast_buffers=True
    )
    
    # Create trainer
    trainer = create_multi_gpu_trainer(config=config)
    
    # Create model and data
    model = EmailSequenceModel()
    train_dataset = EmailSequenceDataset(num_samples=10000)
    
    # Setup distributed data loader
    train_loader = trainer.setup_dataloader(
        train_dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize training
    optimized_model = trainer.initialize_training(model)
    
    # Train model
    results = await trainer.train()
    
    # Cleanup
    dist.destroy_process_group()
    
    return results

def main():
    world_size = torch.cuda.device_count()
    
    # Launch distributed training
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

### GPU Monitoring Example

```python
from core.multi_gpu_training import create_multi_gpu_trainer, MultiGPUConfig

def monitor_gpu_usage():
    # Create trainer with GPU monitoring
    config = MultiGPUConfig(
        training_mode="auto",
        enable_gpu_monitoring=True
    )
    
    trainer = create_multi_gpu_trainer(config=config)
    
    # Get GPU information
    gpu_info = trainer.gpu_monitor.get_gpu_info()
    print(f"Available GPUs: {gpu_info['device_count']}")
    
    for device_id, device_info in gpu_info['devices'].items():
        print(f"GPU {device_id}: {device_info['name']}")
        print(f"  Memory: {device_info['total_memory'] / 1024**3:.1f} GB")
        print(f"  Compute Capability: {device_info['capability']}")
    
    # Monitor GPU usage during training
    model = EmailSequenceModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Simulate training
    for epoch in range(5):
        for batch in range(10):
            # Simulate forward pass
            dummy_input = torch.randn(32, 50).cuda()
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Record GPU metrics
            trainer.record_gpu_metrics()
        
        # Get epoch summary
        summary = trainer.gpu_monitor.get_gpu_summary()
        print(f"Epoch {epoch}: Average memory usage = {summary['memory_allocated_mean'] / 1024**3:.2f} GB")

# Run monitoring
monitor_gpu_usage()
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training
- Clear GPU cache between epochs

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller batch size
config = MultiGPUConfig(
    training_mode="data_parallel",
    enable_data_parallel=True
)

# Enable mixed precision
optimizer = create_optimized_training_optimizer(
    model=model,
    train_loader=train_loader,
    multi_gpu_config=config,
    enable_mixed_precision=True,
    batch_size=32  # Reduced batch size
)
```

#### 2. DataParallel Performance Issues

**Symptoms:**
- Slow training with multiple GPUs
- High memory usage

**Solutions:**
- Use DistributedDataParallel instead
- Optimize data loading
- Check for CPU bottlenecks

```python
# Switch to DistributedDataParallel
config = MultiGPUConfig(
    training_mode="distributed",
    enable_distributed=True,
    backend="nccl"
)
```

#### 3. Distributed Training Communication Issues

**Symptoms:**
```
RuntimeError: NCCL error
```

**Solutions:**
- Check network connectivity
- Verify environment variables
- Use appropriate backend

```python
# Use gloo backend for CPU-only training
config = MultiGPUConfig(
    training_mode="distributed",
    backend="gloo"  # Use gloo for CPU
)

# Check environment variables
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
print(f"RANK: {os.environ.get('RANK')}")
```

#### 4. GPU Monitoring Issues

**Symptoms:**
- No GPU metrics available
- Incorrect GPU information

**Solutions:**
- Check CUDA installation
- Verify GPU drivers
- Use alternative monitoring tools

```python
# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check GPU count
print(f"GPU count: {torch.cuda.device_count()}")

# Test GPU access
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    test_tensor = torch.randn(10, 10).to(device)
    print(f"GPU test successful: {test_tensor.device}")
```

### Performance Optimization Tips

1. **Batch Size Optimization**
   ```python
   # Find optimal batch size
   optimal_batch_size = optimizer.get_optimal_batch_size(target_memory_usage=0.8)
   print(f"Optimal batch size: {optimal_batch_size}")
   ```

2. **Memory Optimization**
   ```python
   # Enable memory optimizations
   config = MultiGPUConfig(
       training_mode="data_parallel",
       enable_gpu_monitoring=True
   )
   
   optimizer = create_optimized_training_optimizer(
       model=model,
       train_loader=train_loader,
       multi_gpu_config=config,
       enable_mixed_precision=True,
       enable_gradient_checkpointing=True
   )
   ```

3. **Data Loading Optimization**
   ```python
   # Optimize data loading
   train_loader = DataLoader(
       dataset,
       batch_size=64,
       num_workers=4,
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
   )
   ```

## API Reference

### MultiGPUTrainer

Main class for multi-GPU training management.

#### Methods

- `initialize_training(model)`: Initialize multi-GPU training
- `setup_dataloader(dataset, batch_size, **kwargs)`: Setup DataLoader for multi-GPU
- `record_gpu_metrics()`: Record GPU metrics
- `get_training_info()`: Get training information
- `get_performance_summary()`: Get performance summary
- `cleanup()`: Cleanup resources

### MultiGPUConfig

Configuration class for multi-GPU training.

#### Attributes

- `training_mode`: Training mode ("auto", "single_gpu", "data_parallel", "distributed")
- `enable_data_parallel`: Enable DataParallel
- `enable_distributed`: Enable DistributedDataParallel
- `device_ids`: Specific GPU device IDs
- `backend`: Distributed backend ("nccl", "gloo")
- `enable_gpu_monitoring`: Enable GPU monitoring
- `sync_bn`: Synchronize batch normalization

### GPUMonitor

GPU monitoring and metrics collection.

#### Methods

- `start_monitoring()`: Start GPU monitoring
- `record_gpu_metrics(device_ids)`: Record GPU metrics
- `get_gpu_summary()`: Get monitoring summary
- `get_gpu_info()`: Get GPU information

### Utility Functions

- `create_multi_gpu_trainer(**kwargs)`: Create multi-GPU trainer
- `optimize_model_for_multi_gpu(model, **kwargs)`: Optimize model for multi-GPU
- `setup_distributed_environment(world_size, backend, init_method)`: Setup distributed environment
- `launch_distributed_training(script_path, world_size, **kwargs)`: Launch distributed training
- `get_free_port()`: Get free port for distributed training

## Conclusion

The Multi-GPU Training System provides comprehensive support for training email sequence models across multiple GPUs. It offers automatic detection, performance monitoring, and seamless integration with existing training pipelines. Whether you're using DataParallel for single-machine multi-GPU training or DistributedDataParallel for distributed training, the system provides the tools and optimizations needed for efficient and scalable training.

For more information and examples, see the demonstration scripts and API documentation. 