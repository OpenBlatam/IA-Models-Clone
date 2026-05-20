# Profiling and Optimization Guide

A comprehensive guide for profiling and optimizing the Onyx Ads Backend to identify and resolve bottlenecks in data loading and preprocessing.

## 🚀 Overview

This guide covers comprehensive profiling and optimization techniques to identify performance bottlenecks and automatically apply optimizations for:
- **Data loading bottlenecks** - I/O operations, memory usage, worker configuration
- **Preprocessing bottlenecks** - CPU-bound operations, parallel processing, caching
- **Training bottlenecks** - GPU utilization, memory management, batch size optimization
- **System bottlenecks** - CPU, memory, GPU monitoring and optimization

## 📊 Profiling System Architecture

### Core Components

1. **ProfilingOptimizer** - Main profiling engine with CPU, memory, and GPU profiling
2. **DataLoadingOptimizer** - Specialized optimization for data loading operations
3. **PreprocessingOptimizer** - Optimization for data preprocessing pipelines
4. **MemoryOptimizer** - Memory usage optimization and monitoring
5. **IOOptimizer** - I/O operations optimization
6. **RealTimeProfiler** - Real-time monitoring and alerting

### Profiling Tools Integration

- **cProfile** - CPU profiling with function-level analysis
- **line_profiler** - Line-by-line profiling (when available)
- **memory_profiler** - Memory usage profiling (when available)
- **torch.profiler** - GPU profiling with detailed CUDA analysis
- **psutil** - System resource monitoring
- **Custom profilers** - Specialized profiling for data operations

## 🔧 Configuration

### ProfilingConfig

```python
from onyx.server.features.ads.profiling_optimizer import ProfilingConfig

config = ProfilingConfig(
    enabled=True,
    profile_cpu=True,
    profile_memory=True,
    profile_gpu=True,
    profile_data_loading=True,
    profile_preprocessing=True,
    profile_depth=10,
    min_time_threshold=0.001,  # 1ms
    min_memory_threshold=1024 * 1024,  # 1MB
    save_profiles=True,
    profile_dir="profiles",
    auto_optimize=True,
    optimization_threshold=0.1,  # 10% improvement required
    real_time_monitoring=True,
    alert_threshold=5.0,  # 5 seconds
    monitoring_interval=1.0  # 1 second
)
```

### DataOptimizationConfig

```python
from onyx.server.features.ads.data_optimization import DataOptimizationConfig

config = DataOptimizationConfig(
    optimize_loading=True,
    prefetch_factor=2,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    memory_efficient=True,
    max_memory_usage=0.8,  # 80% of available memory
    chunk_size=1000,
    enable_caching=True,
    cache_dir="cache",
    cache_size=1000,
    optimize_preprocessing=True,
    batch_preprocessing=True,
    parallel_preprocessing=True,
    preprocessing_workers=2,
    optimize_io=True,
    compression="gzip",
    buffer_size=8192,
    async_io=True,
    monitor_performance=True,
    log_metrics=True,
    alert_threshold=5.0
)
```

## 🎯 Usage Examples

### 1. Basic Profiling

```python
from onyx.server.features.ads.profiling_optimizer import ProfilingOptimizer, ProfilingConfig

# Create profiler
config = ProfilingConfig(enabled=True, profile_cpu=True, profile_memory=True)
profiler = ProfilingOptimizer(config)

# Profile a function
def sample_function():
    import time
    time.sleep(0.1)  # Simulate work
    return "result"

result = profiler.profile_code(sample_function)
print(f"Profiling result: {result}")
print(f"Bottlenecks: {result.bottleneck_functions}")
print(f"Recommendations: {result.recommendations}")
```

### 2. Data Loading Optimization

```python
from onyx.server.features.ads.data_optimization import (
    DataLoadingOptimizer, 
    DataOptimizationConfig,
    optimize_dataset
)

# Create optimizer
config = DataOptimizationConfig(
    optimize_loading=True,
    memory_efficient=True,
    enable_caching=True
)
optimizer = DataLoadingOptimizer(config)

# Optimize dataset
dataset = YourDataset()
optimized_dataset = optimize_dataset(dataset, config)

# Create optimized DataLoader
optimized_dataloader = optimizer.optimize_dataloader(
    optimized_dataset,
    batch_size=32,
    shuffle=True
)

# Identify bottlenecks
bottlenecks = optimizer.identify_bottlenecks()
print(f"Identified bottlenecks: {bottlenecks}")
```

### 3. Preprocessing Optimization

```python
from onyx.server.features.ads.data_optimization import PreprocessingOptimizer

# Create optimizer
optimizer = PreprocessingOptimizer(config)

# Define preprocessing functions
def normalize_text(text):
    return text.lower().strip()

def tokenize_text(text):
    return text.split()

def augment_text(text):
    return text + " [AUGMENTED]"

preprocessing_funcs = [normalize_text, tokenize_text, augment_text]

# Create optimized pipeline
optimized_pipeline = optimizer.optimize_preprocessing_pipeline(
    preprocessing_funcs, sample_data
)

# Use optimized pipeline
results = optimized_pipeline(data)
```

### 4. Memory Optimization

```python
from onyx.server.features.ads.data_optimization import MemoryOptimizer

# Create optimizer
optimizer = MemoryOptimizer(config)

# Use memory context
with optimizer.memory_context():
    # Your memory-intensive operations
    large_data = load_large_dataset()
    processed_data = process_data(large_data)
    del large_data  # Explicit cleanup

# Optimize data structures
optimized_data = optimizer.optimize_memory_usage(data)
```

### 5. I/O Optimization

```python
from onyx.server.features.ads.data_optimization import IOOptimizer

# Create optimizer
optimizer = IOOptimizer(config)

# Optimize file reading
optimized_reader = optimizer.optimize_file_reading("large_file.txt")
content = optimized_reader()

# Optimize data storage
optimizer.optimize_data_storage(data, "output_file")
```

### 6. Real-time Monitoring

```python
from onyx.server.features.ads.profiling_optimizer import RealTimeProfiler

# Create profiler
profiler = RealTimeProfiler(config)

# Add alert callback
def alert_callback(alert, metrics):
    print(f"Alert: {alert}")
    print(f"Metrics: {metrics}")

profiler.add_alert_callback(alert_callback)

# Start monitoring
profiler.start_monitoring()

# Your operations here
# ...

# Stop monitoring
profiler.stop_monitoring()

# Get performance summary
summary = profiler.get_performance_summary()
print(f"Performance summary: {summary}")
```

### 7. Integration with Fine-tuning Service

```python
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService

# Initialize service
service = OptimizedFineTuningService()

# Profile and optimize entire training pipeline
result = await service.profile_and_optimize_training(
    model_name="gpt2",
    dataset=dataset,
    training_config={
        "epochs": 3,
        "learning_rate": 5e-5
    },
    user_id=123,
    profile_data_loading=True,
    profile_preprocessing=True,
    profile_training=True
)

print(f"Profiling results: {result}")
print(f"Optimization plan: {result['optimization_plan']}")
print(f"Recommendations: {result['recommendations']}")

# Optimize specific components
optimized_dataloader = await service.optimize_dataset_loading(dataset, batch_size=32)

optimized_pipeline = await service.optimize_preprocessing_pipeline(
    preprocessing_funcs, sample_data
)

# Get performance report
report = await service.get_performance_report()
print(f"Performance report: {report}")
```

## 📊 API Usage

### Profiling Endpoints

```bash
# Profile and optimize training
curl -X POST http://localhost:8000/finetuning/profile-and-optimize \
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
    "profile_data_loading": true,
    "profile_preprocessing": true,
    "profile_training": true
  }'

# Optimize dataset loading
curl -X POST http://localhost:8000/finetuning/optimize-dataset-loading \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_config": {
      "texts": ["sample text 1", "sample text 2"],
      "max_length": 512
    },
    "batch_size": 32
  }'

# Get performance report
curl http://localhost:8000/finetuning/performance-report
```

### Optimization Endpoints

```bash
# Optimize preprocessing pipeline
curl -X POST http://localhost:8000/finetuning/optimize-preprocessing \
  -H "Content-Type: application/json" \
  -d '{
    "preprocessing_functions": [
      "normalize_text",
      "tokenize_text",
      "augment_text"
    ],
    "sample_data": ["sample text 1", "sample text 2"]
  }'

# Profile specific operation
curl -X POST http://localhost:8000/finetuning/profile-operation \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "data_loading",
    "config": {
      "profile_cpu": true,
      "profile_memory": true,
      "profile_gpu": true
    }
  }'
```

## 🔍 Monitoring and Debugging

### Performance Monitoring

```python
# Monitor specific operations
with profiler.profile_function("data_loading"):
    dataloader = DataLoader(dataset, batch_size=32)
    for batch in dataloader:
        process_batch(batch)

# Get detailed statistics
stats = profiler.get_training_stats()
print(f"CPU usage: {stats['cpu_percent']}%")
print(f"Memory usage: {stats['memory_percent']}%")
print(f"GPU utilization: {stats['gpu_utilization']}%")
```

### Bottleneck Identification

```python
# Identify bottlenecks in data loading
bottlenecks = data_loading_optimizer.identify_bottlenecks()
for bottleneck_type, details in bottlenecks.items():
    print(f"{bottleneck_type}: {details}")

# Identify bottlenecks in preprocessing
preprocessing_bottlenecks = preprocessing_optimizer.identify_bottlenecks()
for bottleneck in preprocessing_bottlenecks:
    print(f"Preprocessing bottleneck: {bottleneck}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('onyx.server.features.ads.profiling_optimizer').setLevel(logging.DEBUG)
logging.getLogger('onyx.server.features.ads.data_optimization').setLevel(logging.DEBUG)

# Test profiling setup
config = ProfilingConfig(
    enabled=True,
    log_precision=True,
    log_memory_usage=True
)
profiler = ProfilingOptimizer(config)
```

## 🚀 Best Practices

### 1. Profiling Strategy

```python
# Profile at different levels
def comprehensive_profiling():
    # System-level profiling
    with profiling_context(config):
        # Application-level profiling
        with profiler.profile_function("main_operation"):
            # Component-level profiling
            result = profiler.profile_code(specific_function)
            return result

# Use appropriate profiling depth
config = ProfilingConfig(
    profile_depth=5,  # Shallow for quick analysis
    min_time_threshold=0.01,  # 10ms threshold
    min_memory_threshold=1024 * 1024  # 1MB threshold
)
```

### 2. Data Loading Optimization

```python
# Optimize based on dataset characteristics
def optimize_for_dataset(dataset):
    analysis = data_loading_optimizer._analyze_dataset(dataset)
    
    if analysis["complexity"] == "complex":
        # Use fewer workers for complex datasets
        config.num_workers = min(4, os.cpu_count())
    else:
        # Use more workers for simple datasets
        config.num_workers = min(8, os.cpu_count())
    
    if analysis["size"] < 1000:
        # Disable multiprocessing for small datasets
        config.num_workers = 0
    
    return data_loading_optimizer.optimize_dataloader(dataset, config)
```

### 3. Memory Management

```python
# Use memory optimization context
with memory_optimization_context(config):
    # Load data in chunks
    for chunk in data_chunks:
        processed_chunk = process_data(chunk)
        yield processed_chunk
        del chunk  # Explicit cleanup

# Optimize data structures
def optimize_data_structures(data):
    if isinstance(data, torch.Tensor):
        # Use appropriate dtype
        if data.dtype == torch.float64:
            data = data.float()
        # Use contiguous memory layout
        if not data.is_contiguous():
            data = data.contiguous()
    return data
```

### 4. Preprocessing Optimization

```python
# Use appropriate optimization strategy
def choose_preprocessing_strategy(funcs, data_size):
    if data_size < 1000:
        return "sequential"  # Small datasets
    elif any("io" in f.__name__ for f in funcs):
        return "parallel_io"  # I/O bound operations
    else:
        return "parallel_cpu"  # CPU bound operations

# Cache expensive operations
@lru_cache(maxsize=1000)
def expensive_preprocessing(text):
    # Expensive operation
    return complex_processing(text)
```

### 5. Real-time Monitoring

```python
# Set up monitoring with appropriate thresholds
config = ProfilingConfig(
    real_time_monitoring=True,
    alert_threshold=5.0,  # 5 seconds
    monitoring_interval=1.0  # 1 second
)

# Add custom alert handlers
def custom_alert_handler(alert, metrics):
    if "High CPU usage" in alert:
        # Reduce workload
        reduce_parallel_workers()
    elif "High memory usage" in alert:
        # Trigger garbage collection
        gc.collect()

profiler.add_alert_callback(custom_alert_handler)
```

## 🔧 Troubleshooting

### Common Issues

1. **High CPU Usage**
   ```python
   # Reduce parallel workers
   config.num_workers = max(1, config.num_workers // 2)
   
   # Use sequential processing for small datasets
   if dataset_size < 1000:
       config.num_workers = 0
   ```

2. **High Memory Usage**
   ```python
   # Enable memory optimization
   config.memory_efficient = True
   config.chunk_size = 500  # Reduce chunk size
   
   # Use streaming for large datasets
   config.streaming = True
   ```

3. **Slow Data Loading**
   ```python
   # Increase workers
   config.num_workers = min(8, os.cpu_count())
   
   # Enable persistent workers
   config.persistent_workers = True
   
   # Increase prefetch factor
   config.prefetch_factor = 4
   ```

4. **GPU Memory Issues**
   ```python
   # Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Reduce batch size
   batch_size = batch_size // 2
   
   # Use mixed precision
   config.mixed_precision = True
   ```

### Performance Optimization

1. **Batch Size Tuning**
   ```python
   # Start with small batch size and increase
   batch_size = 8
   while gpu_memory_usage < 0.8:  # 80% threshold
       batch_size *= 2
       test_batch_size(batch_size)
   ```

2. **Worker Optimization**
   ```python
   # Calculate optimal workers
   cpu_count = os.cpu_count()
   memory_gb = psutil.virtual_memory().total / (1024**3)
   
   if memory_gb < 8:
       optimal_workers = min(cpu_count // 2, 2)
   else:
       optimal_workers = min(cpu_count, 8)
   ```

3. **Caching Strategy**
   ```python
   # Use appropriate cache size
   cache_size = min(1000, dataset_size // 10)
   
   # Clear cache periodically
   if cache_hits < cache_misses * 0.5:
       clear_cache()
   ```

## 📈 Performance Benchmarks

### Expected Improvements

| Optimization | Expected Improvement | Conditions |
|--------------|---------------------|------------|
| Data Loading | 2-4x faster | Proper worker configuration |
| Preprocessing | 2-3x faster | Parallel processing |
| Memory Usage | 30-50% reduction | Memory optimization |
| GPU Utilization | 20-40% increase | Batch size optimization |
| Overall Training | 1.5-2x faster | Combined optimizations |

### Monitoring Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU Usage | >80% | Reduce workers |
| Memory Usage | >80% | Enable memory optimization |
| GPU Memory | >80% | Reduce batch size |
| Data Loading Time | >100ms/batch | Increase workers |
| Preprocessing Time | >1ms/sample | Enable parallel processing |

## 🔒 Security Considerations

### Access Control

- Implement authentication for profiling endpoints
- Use rate limiting for optimization requests
- Monitor and log all profiling operations

### Resource Protection

- Set memory limits to prevent system crashes
- Implement profiling timeouts for long-running operations
- Monitor system resources during profiling

### Data Protection

- Ensure profiling data is properly secured
- Implement secure storage for profiling results
- Monitor memory for sensitive data during profiling

## 📚 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from onyx.server.features.ads.profiling_optimizer import ProfilingOptimizer

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global profiler
    config = ProfilingConfig(enabled=True)
    profiler = ProfilingOptimizer(config)

@app.post("/profile/operation")
async def profile_operation(request: ProfilingRequest):
    result = profiler.profile_code(request.function, *request.args)
    return result
```

### Custom Training Loop

```python
async def optimized_training_loop(model, dataset, config):
    # Setup profiling
    profiler = ProfilingOptimizer(config.profiling_config)
    
    # Optimize data loading
    data_optimizer = DataLoadingOptimizer(config.data_optimization_config)
    optimized_dataloader = data_optimizer.optimize_dataloader(dataset)
    
    # Profile training
    with profiler.profile_function("training_epoch"):
        for epoch in range(config.epochs):
            for batch in optimized_dataloader:
                # Training step
                pass
    
    return profiler.get_training_stats()
```

This comprehensive profiling and optimization system provides the tools and capabilities needed to identify and resolve bottlenecks in data loading and preprocessing, leading to significant performance improvements in the Onyx Ads Backend. 