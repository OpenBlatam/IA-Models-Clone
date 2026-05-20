# Performance Optimizer Documentation

## Overview

The Performance Optimizer is a comprehensive module designed to optimize AI video processing performance using modern Python async/await patterns and functional programming principles. It provides GPU management, caching, profiling, and experiment tracking capabilities.

## Key Features

### 🚀 **Async-First Design**
- All operations use `async def` for asynchronous operations
- Pure functions use `def` for synchronous operations
- Proper error handling with retry mechanisms
- Concurrent task management with semaphores

### 🎯 **GPU Optimization**
- Multi-GPU support with automatic device management
- Mixed precision training with PyTorch AMP
- Memory management and cache clearing
- Gradient checkpointing and attention slicing

### 💾 **Intelligent Caching**
- Thread-safe async cache with TTL
- Automatic cache size management
- Memory-efficient storage
- Cache hit/miss statistics

### 📊 **Performance Monitoring**
- Real-time profiling with PyTorch Profiler
- Memory usage tracking
- Execution time measurement
- Metrics collection and reporting

### 🔬 **Experiment Tracking**
- WandB integration for experiment tracking
- TensorBoard support for visualization
- Configuration logging
- Performance metrics history

## Architecture

### Core Components

1. **PerformanceOptimizer** - Main orchestrator class
2. **GPUManager** - GPU resource management
3. **ModelManager** - Model loading and management
4. **AsyncTaskManager** - Concurrent task execution
5. **PerformanceCache** - Thread-safe caching
6. **PerformanceProfiler** - Profiling and monitoring
7. **ExperimentTracker** - Experiment tracking

### Configuration

```python
@dataclass
class OptimizationConfig:
    # GPU settings
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 4
    
    # Memory management
    max_memory_usage: float = 0.8
    enable_gradient_checkpointing: bool = True
    enable_attention_slicing: bool = True
    
    # Caching
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_cpu: bool = True
    
    # Experiment tracking
    enable_wandb: bool = False
    enable_tensorboard: bool = True
    experiment_name: str = "ai_video_optimization"
    
    # Async settings
    max_concurrent_tasks: int = 4
    task_timeout: float = 300.0
    
    # Model settings
    model_name: str = "stabilityai/stable-diffusion-2-1"
    tokenizer_name: str = "openai/clip-vit-large-patch14"
```

## Usage Examples

### Basic Usage

```python
import asyncio
from performance_optimizer import (
    create_performance_optimizer,
    OptimizationConfig
)

async def main():
    # Create configuration
    config = OptimizationConfig(
        use_gpu=True,
        mixed_precision=True,
        cache_enabled=True,
        enable_profiling=True
    )
    
    # Create optimizer
    optimizer = await create_performance_optimizer(config)
    
    try:
        # Optimize text processing
        text = "Generate a beautiful landscape video"
        processed_text = await optimizer.optimize_text_processing(text)
        
        # Optimize video generation
        video_data = await optimizer.optimize_video_generation(
            "A serene mountain landscape",
            num_inference_steps=20
        )
        
        # Get optimization stats
        stats = await optimizer.get_optimization_stats()
        print(f"Optimization stats: {stats}")
        
    finally:
        await optimizer.cleanup()

# Run
asyncio.run(main())
```

### Advanced Usage with Batch Processing

```python
async def batch_optimization_example():
    config = OptimizationConfig(
        max_concurrent_tasks=4,
        enable_profiling=True,
        enable_wandb=True
    )
    
    optimizer = await create_performance_optimizer(config)
    
    try:
        # Batch tasks
        tasks = [
            {"type": "text", "text": "Process text 1"},
            {"type": "text", "text": "Process text 2"},
            {"type": "video", "prompt": "Video 1", "kwargs": {"num_inference_steps": 15}},
            {"type": "video", "prompt": "Video 2", "kwargs": {"num_inference_steps": 20}}
        ]
        
        # Run batch optimization
        results = await optimizer.run_batch_optimization(tasks)
        
        for i, result in enumerate(results):
            print(f"Task {i} result: {type(result)}")
            
    finally:
        await optimizer.cleanup()
```

### Custom Configuration

```python
async def custom_config_example():
    # Custom configuration for specific use case
    config = OptimizationConfig(
        # GPU settings
        use_gpu=True,
        gpu_ids=[0, 1],  # Use multiple GPUs
        mixed_precision=True,
        gradient_accumulation_steps=8,
        
        # Memory optimization
        max_memory_usage=0.9,
        enable_gradient_checkpointing=True,
        enable_attention_slicing=True,
        
        # Caching
        cache_enabled=True,
        cache_size=2000,
        cache_ttl=7200,  # 2 hours
        
        # Profiling
        enable_profiling=True,
        profile_memory=True,
        profile_cpu=True,
        
        # Experiment tracking
        enable_wandb=True,
        enable_tensorboard=True,
        experiment_name="custom_optimization_experiment",
        
        # Async settings
        max_concurrent_tasks=8,
        task_timeout=600.0,  # 10 minutes
        
        # Model settings
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        tokenizer_name="openai/clip-vit-large-patch14"
    )
    
    return await create_performance_optimizer(config)
```

## Utility Functions

### Pure Functions (def)

```python
# Configuration management
def create_optimization_config(**kwargs) -> OptimizationConfig:
    """Create optimization configuration."""
    return OptimizationConfig(**kwargs)

def validate_optimization_config(config: OptimizationConfig) -> bool:
    """Validate optimization configuration."""
    return config.max_concurrent_tasks > 0 and config.task_timeout > 0

# Memory utilities
def calculate_memory_usage(tensor: torch.Tensor) -> int:
    """Calculate memory usage of tensor in bytes."""
    return tensor.element_size() * tensor.nelement()

def get_optimal_batch_size(available_memory: int, model_memory: int) -> int:
    """Calculate optimal batch size based on available memory."""
    return max(1, available_memory // (model_memory * 2))
```

### Async Functions (async def)

```python
# Execution time measurement
async def measure_execution_time(coro: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of coroutine."""
    start_time = time.time()
    result = await coro(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

# Retry mechanism
async def retry_operation(coro: Callable, max_retries: int = 3, delay: float = 1.0, *args, **kwargs) -> Any:
    """Retry operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay * (2 ** attempt))

# Parallel processing
async def parallel_map(func: Callable, items: List[Any], max_workers: int = 4) -> List[Any]:
    """Parallel map with async function."""
    semaphore = asyncio.Semaphore(max_workers)
    
    async def worker(item):
        async with semaphore:
            return await func(item)
    
    return await asyncio.gather(*[worker(item) for item in items])
```

## Best Practices

### 1. **Resource Management**
```python
# Always use try/finally for cleanup
optimizer = await create_performance_optimizer(config)
try:
    # Your operations here
    pass
finally:
    await optimizer.cleanup()
```

### 2. **Error Handling**
```python
# Use retry mechanism for transient failures
result = await retry_operation(
    optimizer.optimize_text_processing,
    max_retries=3,
    delay=1.0,
    text="Your text here"
)
```

### 3. **Performance Monitoring**
```python
# Enable profiling for performance analysis
config = OptimizationConfig(enable_profiling=True)
optimizer = await create_performance_optimizer(config)

# Get detailed stats
stats = await optimizer.get_optimization_stats()
print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")
```

### 4. **Memory Management**
```python
# Monitor memory usage
memory_info = await optimizer.gpu_manager.get_memory_info()
for device, info in memory_info.items():
    print(f"{device}: {info['allocated'] / info['total']:.2%} used")
```

### 5. **Configuration Validation**
```python
# Always validate configuration
config = create_optimization_config(max_concurrent_tasks=4)
if not validate_optimization_config(config):
    raise ValueError("Invalid configuration")
```

## Integration with Existing System

The Performance Optimizer integrates seamlessly with the existing AI Video system:

```python
# In your main.py or workflow
from .optimization.performance_optimizer import create_performance_optimizer

class AIVideoSystem:
    async def initialize(self):
        # Initialize performance optimizer
        self.performance_optimizer = await create_performance_optimizer()
    
    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        # Use optimizer for enhanced performance
        optimized_prompt = await self.performance_optimizer.optimize_text_processing(
            request.prompt
        )
        
        video_data = await self.performance_optimizer.optimize_video_generation(
            optimized_prompt,
            num_inference_steps=request.num_steps
        )
        
        return VideoResponse(video_data=video_data)
```

## Testing

Run the test suite to verify functionality:

```bash
python -m agents.backend.onyx.server.features.ai_video.optimization.test_performance_optimizer
```

## Performance Metrics

The optimizer tracks various performance metrics:

- **Cache Performance**: Hit/miss ratios, cache size
- **Processing Time**: Text processing, video generation, batch processing
- **Memory Usage**: GPU memory allocation, cache memory
- **Concurrency**: Active tasks, task completion rates
- **Error Rates**: Retry attempts, failure rates

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce `max_memory_usage`
   - Enable `enable_attention_slicing`
   - Use `enable_gradient_checkpointing`

2. **Slow Performance**
   - Enable `mixed_precision`
   - Increase `max_concurrent_tasks`
   - Check cache hit rates

3. **Timeout Errors**
   - Increase `task_timeout`
   - Reduce batch sizes
   - Check GPU utilization

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = OptimizationConfig(enable_profiling=True)
optimizer = await create_performance_optimizer(config)
```

## Future Enhancements

- **Distributed Training**: Multi-node GPU support
- **Model Quantization**: INT8/FP16 optimization
- **Dynamic Batching**: Adaptive batch size adjustment
- **Auto-scaling**: Automatic resource scaling
- **Custom Models**: Support for custom model architectures

## Contributing

When contributing to the Performance Optimizer:

1. Follow the async/await patterns
2. Use `def` for pure functions
3. Use `async def` for async operations
4. Add proper error handling
5. Include comprehensive tests
6. Update documentation

## License

This module is part of the AI Video System and follows the same licensing terms. 