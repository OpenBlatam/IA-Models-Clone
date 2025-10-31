# Multi-GPU Training Implementation

## Overview

This document describes the comprehensive multi-GPU training implementation that has been integrated into the numerical stability framework. The system provides support for both DataParallel and DistributedDataParallel training modes, with automatic mode selection and optimization.

## Features

### 🚀 Multi-GPU Training Modes

- **DataParallel**: Simple multi-GPU training for single-node setups
- **DistributedDataParallel**: Advanced distributed training for multi-node setups
- **Hybrid**: Combination of different parallelism strategies
- **Auto**: Automatic mode selection based on hardware configuration

### 🔧 Key Capabilities

- **Automatic Mode Detection**: Chooses optimal training mode based on available GPUs
- **Model Wrapping**: Seamlessly wraps models for multi-GPU training
- **DataLoader Optimization**: Automatically adds DistributedSampler for distributed training
- **Memory Management**: Per-GPU memory monitoring and optimization
- **Performance Monitoring**: Real-time GPU utilization and communication overhead tracking
- **Fault Tolerance**: Automatic retry mechanisms and error recovery
- **Batch Size Optimization**: Calculates optimal batch sizes for multi-GPU setups

## Architecture

### Core Components

#### 1. MultiGPUConfig
Configuration class for multi-GPU training settings:

```python
@dataclass
class MultiGPUConfig:
    mode: MultiGPUMode = MultiGPUMode.AUTO
    device_ids: Optional[List[int]] = None
    output_device: Optional[int] = None
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = True
    enable_hybrid: bool = False
    hybrid_strategy: str = "pipeline"
    enable_multi_gpu_monitoring: bool = True
    sync_bn: bool = True
    enable_gradient_synchronization: bool = True
    enable_fault_tolerance: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
```

#### 2. MultiGPUManager
Core class managing multi-GPU training operations:

```python
class MultiGPUManager:
    def __init__(self, config: PerformanceConfig)
    def _setup_multi_gpu(self)
    def wrap_model(self, model: nn.Module) -> nn.Module
    def optimize_dataloader(self, dataloader: DataLoader) -> DataLoader
    def synchronize(self)
    def get_gpu_stats(self) -> Dict[str, Any]
    def cleanup(self)
    def is_multi_gpu_enabled(self) -> bool
    def get_optimal_batch_size(self, base_batch_size: int) -> int
```

#### 3. Integration with PerformanceOptimizer
The MultiGPUManager is integrated into the main PerformanceOptimizer:

```python
class PerformanceOptimizer:
    def __init__(self, config: PerformanceConfig):
        self.multi_gpu_manager = MultiGPUManager(config)
    
    def wrap_model_for_multi_gpu(self, model: nn.Module) -> nn.Module
    def get_multi_gpu_status(self) -> Dict[str, Any]
    def get_optimal_batch_size(self, base_batch_size: int) -> int
    def synchronize_gpus(self)
```

## Usage Examples

### Basic Multi-GPU Setup

```python
from performance_optimization import (
    PerformanceConfig, MultiGPUConfig, MultiGPUMode, 
    create_performance_optimizer
)

# Create multi-GPU configuration
multi_gpu_config = MultiGPUConfig(
    mode=MultiGPUMode.DATAPARALLEL,
    device_ids=[0, 1, 2, 3],  # Use GPUs 0, 1, 2, 3
    output_device=0
)

# Create performance configuration
config = PerformanceConfig(
    optimization_level=OptimizationLevel.ADVANCED,
    multi_gpu_config=multi_gpu_config,
    enable_mixed_precision=True,
    enable_compile=True
)

# Create optimizer
optimizer = create_performance_optimizer(config)

# Wrap model for multi-GPU training
model = YourModel()
wrapped_model = optimizer.wrap_model_for_multi_gpu(model)

# Optimize DataLoader
dataloader = DataLoader(dataset, batch_size=32)
optimized_dataloader = optimizer.optimize_training_pipeline(
    wrapped_model, dataloader, torch_optimizer, criterion
)
```

### Distributed Training Setup

```python
# Create distributed configuration
multi_gpu_config = MultiGPUConfig(
    mode=MultiGPUMode.DISTRIBUTED,
    backend="nccl",
    world_size=4,  # Total number of processes
    rank=0,        # Current process rank
    local_rank=0,  # Local GPU rank
    find_unused_parameters=False,
    bucket_cap_mb=25,
    static_graph=True
)

config = PerformanceConfig(
    optimization_level=OptimizationLevel.ULTRA,
    multi_gpu_config=multi_gpu_config
)

optimizer = create_performance_optimizer(config)
```

### Hybrid Training Setup

```python
# Create hybrid configuration
multi_gpu_config = MultiGPUConfig(
    mode=MultiGPUMode.HYBRID,
    enable_hybrid=True,
    hybrid_strategy="pipeline",  # or "model_parallel", "data_parallel"
    backend="nccl"
)

config = PerformanceConfig(
    optimization_level=OptimizationLevel.ULTRA,
    multi_gpu_config=multi_gpu_config
)
```

### Auto Mode (Recommended)

```python
# Let the system automatically choose the best mode
multi_gpu_config = MultiGPUConfig(
    mode=MultiGPUMode.AUTO  # Will choose based on GPU count
)

config = PerformanceConfig(
    optimization_level=OptimizationLevel.ADVANCED,
    multi_gpu_config=multi_gpu_config
)
```

## Integration with Numerical Stability

### Enhanced PerformanceOptimizationConfig

The `PerformanceOptimizationConfig` in the numerical stability system now includes multi-GPU settings:

```python
@dataclass
class PerformanceOptimizationConfig:
    # ... existing fields ...
    
    # Multi-GPU training settings
    enable_multi_gpu: bool = True
    multi_gpu_mode: str = "auto"
    multi_gpu_device_ids: Optional[List[int]] = None
    multi_gpu_backend: str = "nccl"
    multi_gpu_find_unused_parameters: bool = False
    multi_gpu_bucket_cap_mb: int = 25
    multi_gpu_static_graph: bool = True
```

### NumericalStabilityManager Integration

The `NumericalStabilityManager` now provides multi-GPU capabilities:

```python
class NumericalStabilityManager:
    def wrap_model_for_multi_gpu(self, model: nn.Module) -> nn.Module
    def get_multi_gpu_status(self) -> Dict[str, Any]
    def get_optimal_batch_size(self, base_batch_size: int) -> int
    def synchronize_gpus(self)
```

### Training Wrapper Integration

The training wrapper automatically integrates multi-GPU capabilities:

```python
# Create wrapper with multi-GPU support
wrapper = create_training_wrapper(
    clipping_config,
    nan_config,
    debug_config,
    performance_config  # Includes multi-GPU settings
)

# Use in training loop
for batch in dataloader:
    # ... forward pass, loss calculation ...
    
    # Apply stability measures (includes multi-GPU optimization)
    stability_result = wrapper(model, loss, optimizer)
    
    # Get multi-GPU status
    multi_gpu_status = wrapper.get_multi_gpu_status()
    
    # Get optimal batch size for next iteration
    optimal_batch_size = wrapper.get_optimal_batch_size(32)
```

## Performance Benefits

### DataParallel Benefits
- **Simple Setup**: Minimal code changes required
- **Automatic Data Distribution**: Automatically splits batches across GPUs
- **Gradient Aggregation**: Automatically aggregates gradients from all GPUs
- **Memory Efficiency**: Better memory utilization across multiple GPUs

### DistributedDataParallel Benefits
- **Higher Performance**: More efficient than DataParallel
- **Better Memory Usage**: Each GPU maintains its own model copy
- **Scalability**: Can scale across multiple nodes
- **Advanced Features**: Bucket-based gradient communication, static graph optimization

### Hybrid Benefits
- **Flexibility**: Combine different parallelism strategies
- **Optimized Workloads**: Choose best strategy for specific operations
- **Pipeline Efficiency**: Overlap computation and communication

## Monitoring and Debugging

### GPU Statistics

```python
# Get comprehensive GPU statistics
gpu_stats = optimizer.get_multi_gpu_status()

print(f"Device Count: {gpu_stats['device_count']}")
print(f"Current Mode: {gpu_stats['current_mode']}")
print(f"Training Stats: {gpu_stats['training_stats']}")

# Per-GPU memory information
for gpu_info in gpu_stats['gpu_memory']:
    print(f"GPU {gpu_info['device']}: "
          f"Allocated: {gpu_info['allocated_mb']:.1f}MB, "
          f"Free: {gpu_info['free_mb']:.1f}MB")
```

### Performance Metrics

```python
# Get optimization status including multi-GPU info
status = optimizer.get_optimization_status()

print(f"Multi-GPU Mode: {status['config']['multi_gpu_mode']}")
print(f"Device Count: {status['config']['device_count']}")
print(f"Multi-GPU Status: {status['multi_gpu_status']}")
```

### Real-time Monitoring

```python
# Monitor during training
for step in range(num_steps):
    # ... training step ...
    
    # Get current GPU stats
    gpu_stats = optimizer.get_multi_gpu_status()
    
    # Log performance metrics
    if step % 100 == 0:
        print(f"Step {step}: GPU Memory Usage:")
        for gpu_info in gpu_stats['gpu_memory']:
            print(f"  GPU {gpu_info['device']}: {gpu_info['allocated_mb']:.1f}MB")
    
    # Synchronize if needed
    optimizer.synchronize_gpus()
```

## Best Practices

### 1. Mode Selection
- **1-4 GPUs**: Use `DataParallel` for simplicity
- **4+ GPUs**: Use `DistributedDataParallel` for better performance
- **Multi-node**: Always use `DistributedDataParallel`
- **Unknown setup**: Use `AUTO` mode for automatic selection

### 2. Batch Size Optimization
```python
# Let the system calculate optimal batch size
base_batch_size = 32
optimal_batch_size = optimizer.get_optimal_batch_size(base_batch_size)

# The system considers:
# - Number of available GPUs
# - GPU memory constraints
# - Current memory usage
# - Performance characteristics
```

### 3. Memory Management
```python
# Monitor memory usage
memory_info = optimizer.memory_manager.get_memory_usage()

# Clear cache when needed
if memory_info['system_percent'] > 80:
    optimizer.memory_manager.optimize_memory(force=True)

# Synchronize GPUs
optimizer.synchronize_gpus()
```

### 4. Error Handling
```python
try:
    wrapped_model = optimizer.wrap_model_for_multi_gpu(model)
except Exception as e:
    print(f"Multi-GPU setup failed: {e}")
    # Fall back to single GPU
    wrapped_model = model
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
optimal_batch_size = optimizer.get_optimal_batch_size(16)  # Reduced from 32

# Enable gradient checkpointing
config.enable_gradient_checkpointing = True

# Clear GPU cache
torch.cuda.empty_cache()
```

#### 2. Distributed Training Issues
```python
# Check if distributed is initialized
if dist.is_initialized():
    print("Distributed training is active")
else:
    print("Distributed training not initialized")

# Check process group
if optimizer.multi_gpu_manager.distributed_process_group:
    print("Process group exists")
```

#### 3. Performance Issues
```python
# Check GPU utilization
gpu_stats = optimizer.get_multi_gpu_status()
print(f"GPU utilization: {gpu_stats['training_stats']['gpu_utilization']}")

# Optimize communication
config.multi_gpu_bucket_cap_mb = 50  # Increase bucket size
config.multi_gpu_static_graph = True  # Enable static graph optimization
```

### Debug Mode

Enable debug mode for detailed logging:

```python
# Enable debug logging
import logging
logging.getLogger('multi_gpu_manager').setLevel(logging.DEBUG)

# Check detailed status
status = optimizer.get_multi_gpu_status()
print(f"Detailed status: {status}")
```

## Future Enhancements

### Planned Features

1. **Advanced Pipeline Parallelism**: Full implementation of pipeline parallelism
2. **Model Parallelism**: Complete model parallelism support
3. **Dynamic Load Balancing**: Automatic workload distribution
4. **Advanced Fault Tolerance**: More sophisticated error recovery
5. **Performance Profiling**: Detailed performance analysis tools
6. **Multi-Node Support**: Enhanced multi-node training capabilities

### Custom Extensions

The system is designed to be extensible:

```python
class CustomMultiGPUManager(MultiGPUManager):
    def __init__(self, config):
        super().__init__(config)
        # Add custom functionality
    
    def custom_optimization(self, model):
        # Implement custom optimization logic
        pass
```

## Conclusion

The multi-GPU training implementation provides a comprehensive solution for scaling deep learning training across multiple GPUs. With automatic mode selection, seamless integration with the numerical stability framework, and extensive monitoring capabilities, it enables efficient and robust multi-GPU training with minimal code changes.

The system automatically handles the complexity of multi-GPU training while providing the flexibility to customize behavior for specific use cases. Whether using DataParallel for simple multi-GPU setups or DistributedDataParallel for advanced distributed training, the implementation ensures optimal performance and stability.






