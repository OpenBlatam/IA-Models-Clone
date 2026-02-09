# Code Profiling and Bottleneck Detection System

This comprehensive system provides advanced profiling capabilities to identify and optimize bottlenecks in data loading and preprocessing operations for deep learning workflows.

## Overview

The system consists of three main components:

1. **Advanced Bottleneck Profiler** (`advanced_bottleneck_profiler.py`) - Comprehensive profiling with real-time monitoring
2. **Data Loading Profiler** (`data_loading_profiler.py`) - Specialized profiling for data loading bottlenecks
3. **Preprocessing Profiler** (`preprocessing_profiler.py`) - Specialized profiling for preprocessing bottlenecks

## Features

### 🚀 **Advanced Bottleneck Profiler**
- **Real-time monitoring** of CPU, memory, GPU, and I/O usage
- **Automatic bottleneck detection** with severity scoring
- **Session-based profiling** with comprehensive reporting
- **Multiple profiling levels** (Basic, Detailed, Comprehensive, Production)
- **Real-time bottleneck alerts** and optimization suggestions

### 📊 **Data Loading Profiler**
- **Configuration optimization** for DataLoader parameters
- **Batch size analysis** for optimal memory usage
- **Worker count optimization** for parallel processing
- **Memory and GPU transfer profiling**
- **Automatic optimization suggestions**

### 🔧 **Preprocessing Profiler**
- **Function-level profiling** with detailed metrics
- **Batch size optimization** for preprocessing operations
- **Caching analysis** and recommendations
- **GPU utilization monitoring**
- **Algorithm efficiency detection**

## Quick Start

### 1. Basic Usage

```python
from advanced_bottleneck_profiler import BottleneckProfiler, BottleneckProfilerConfig

# Create configuration
config = BottleneckProfilerConfig(
    profiling_level=ProfilingLevel.DETAILED,
    enable_real_time_monitoring=True,
    auto_optimize=False
)

# Create profiler
profiler = BottleneckProfiler(config)

# Start profiling session
session_id = profiler.start_profiling_session("my_training_run")

# Profile your operations
with profiler.profile_operation("data_loading", BottleneckType.DATA_LOADING):
    # Your data loading code here
    pass

# Stop profiling and get results
session = profiler.stop_profiling_session()
summary = profiler.get_bottleneck_summary()
```

### 2. Data Loading Profiling

```python
from data_loading_profiler import DataLoadingProfiler, DataLoadingProfilerConfig

# Create profiler
config = DataLoadingProfilerConfig(
    batch_size_range=[16, 32, 64, 128],
    worker_range=[0, 2, 4, 8]
)
profiler = DataLoadingProfiler(config)

# Profile multiple configurations
profiles = profiler.profile_configuration_range(dataset)

# Get optimal configuration
optimal = profiler.get_optimal_configuration()
print(f"Optimal batch size: {optimal.batch_size}")
print(f"Optimal workers: {optimal.num_workers}")

# Generate report
report_path = profiler.generate_optimization_report()
```

### 3. Preprocessing Profiling

```python
from preprocessing_profiler import PreprocessingProfiler, PreprocessingProfilerConfig

# Create profiler
config = PreprocessingProfilerConfig(
    batch_size_range=[1, 8, 16, 32, 64],
    enable_caching=True
)
profiler = PreprocessingProfiler(config)

# Profile preprocessing function
profile = profiler.profile_preprocessing_function(
    normalize_function, sample_data, num_iterations=10
)

# Profile with different batch sizes
batch_profiles = profiler.profile_batch_preprocessing(
    normalize_function, sample_data
)

# Get optimal configuration
optimal = profiler.get_optimal_preprocessing_config()
```

## Configuration Options

### Bottleneck Profiler Configuration

```python
@dataclass
class BottleneckProfilerConfig:
    profiling_level: ProfilingLevel = ProfilingLevel.DETAILED
    enable_real_time_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    sampling_interval: float = 0.1  # seconds
    bottleneck_threshold: float = 0.05  # 5% of total time
    memory_threshold: float = 0.8  # 80% of available memory
    gpu_threshold: float = 0.9  # 90% of GPU memory
    auto_optimize: bool = False
    save_profiles: bool = True
    profile_output_dir: str = "bottleneck_profiles"
```

### Data Loading Profiler Configuration

```python
@dataclass
class DataLoadingProfilerConfig:
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    profiling_duration: int = 60  # seconds
    batch_size_range: List[int] = [16, 32, 64, 128]
    worker_range: List[int] = [0, 2, 4, 8]
    save_profiles: bool = True
    profile_output_dir: str = "data_loading_profiles"
    auto_optimize: bool = False
```

### Preprocessing Profiler Configuration

```python
@dataclass
class PreprocessingProfilerConfig:
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_cpu_tracking: bool = True
    enable_io_tracking: bool = True
    profiling_duration: int = 60  # seconds
    batch_size_range: List[int] = [1, 8, 16, 32, 64]
    enable_caching: bool = True
    cache_size: int = 1000
    save_profiles: bool = True
    profile_output_dir: str = "preprocessing_profiles"
    auto_optimize: bool = False
```

## Bottleneck Types

### Data Loading Bottlenecks
- **SLOW_DISK_IO** - Slow disk I/O operations
- **INSUFFICIENT_WORKERS** - Not enough parallel workers
- **MEMORY_PRESSURE** - High memory usage
- **GPU_TRANSFER_OVERHEAD** - GPU transfer bottlenecks
- **PREPROCESSING_BOTTLENECK** - Preprocessing operations
- **BATCH_SIZE_INEFFICIENCY** - Suboptimal batch sizes
- **SAMPLER_OVERHEAD** - Custom sampler performance
- **COLLATE_FUNCTION_SLOW** - Slow collate functions

### Preprocessing Bottlenecks
- **CPU_INTENSIVE** - CPU-bound operations
- **MEMORY_INTENSIVE** - Memory-heavy operations
- **GPU_UNDERUTILIZED** - GPU not being used effectively
- **I_O_BOUND** - I/O-bound operations
- **VECTORIZATION_INEFFICIENT** - Poor vectorization
- **CACHING_MISSED** - Missing caching opportunities
- **BATCH_SIZE_INEFFICIENT** - Suboptimal batch sizes
- **ALGORITHM_INEFFICIENT** - Inefficient algorithms

## Optimization Suggestions

The system automatically generates optimization suggestions based on detected bottlenecks:

### Data Loading Optimizations
- Increase `num_workers` for parallel loading
- Enable `pin_memory=True` for faster GPU transfer
- Use `persistent_workers=True` to avoid initialization overhead
- Implement data prefetching with `prefetch_factor`
- Use memory-mapped files for large datasets
- Optimize batch sizes for memory efficiency

### Preprocessing Optimizations
- Move preprocessing to GPU using `torch.cuda.amp`
- Cache preprocessed data for repeated operations
- Use `torch.jit.script` for preprocessing functions
- Implement batch preprocessing operations
- Use vectorized operations instead of loops
- Consider specialized preprocessing libraries

## Real-Time Monitoring

The advanced bottleneck profiler provides real-time monitoring capabilities:

```python
# Start real-time monitoring
profiler.start_profiling_session("real_time_monitoring")

# Monitor automatically detects bottlenecks
# - Memory usage above 80%
# - GPU memory usage above 90%
# - CPU usage patterns
# - I/O bottlenecks

# Get real-time status
summary = profiler.get_bottleneck_summary()
print(f"Active bottlenecks: {len(summary.get('bottlenecks', []))}")

# Stop monitoring
session = profiler.stop_profiling_session()
```

## Report Generation

All profilers generate comprehensive reports:

### Bottleneck Profiler Reports
- Session summaries with timing information
- Real-time bottleneck detection logs
- Performance metrics over time
- Optimization suggestions
- Visual charts and graphs

### Data Loading Reports
- Configuration comparison tables
- Throughput analysis
- Memory usage patterns
- Optimal configuration recommendations
- Performance visualization charts

### Preprocessing Reports
- Function performance analysis
- Batch size optimization results
- Caching effectiveness
- GPU utilization patterns
- Algorithm efficiency metrics

## Integration with Existing Code

### Minimal Code Changes

```python
# Before: Direct function call
result = my_preprocessing_function(data)

# After: Profiled function call
with profiler.profile_operation("my_preprocessing", BottleneckType.PREPROCESSING):
    result = my_preprocessing_function(data)
```

### Context Manager Usage

```python
# Profile specific operations
with profiler.profile_operation("data_loading", BottleneckType.DATA_LOADING):
    for batch in data_loader:
        # Your data loading code
        pass

with profiler.profile_operation("model_inference", BottleneckType.CPU_COMPUTATION):
    output = model(input_data)
```

### Automatic Profiling

```python
# Enable auto-optimize for automatic improvements
config = BottleneckProfilerConfig(auto_optimize=True)
profiler = BottleneckProfiler(config)

# The profiler will automatically suggest and apply optimizations
```

## Performance Metrics

### Memory Tracking
- **Process memory usage** (RSS)
- **Available system memory**
- **Memory pressure indicators**
- **Memory allocation patterns**

### GPU Tracking
- **GPU memory allocated**
- **GPU memory reserved**
- **GPU utilization patterns**
- **Memory transfer overhead**

### CPU Tracking
- **CPU usage percentage**
- **CPU frequency monitoring**
- **Core utilization patterns**
- **Process priority information**

### I/O Tracking
- **Read/write operations**
- **I/O wait times**
- **Disk usage patterns**
- **Network I/O metrics**

## Advanced Features

### Session Management
```python
# Start multiple profiling sessions
session1 = profiler.start_profiling_session("training_phase_1")
session2 = profiler.start_profiling_session("training_phase_2")

# Compare sessions
session1_summary = profiler.get_session_summary(session1)
session2_summary = profiler.get_session_summary(session2)
```

### Custom Bottleneck Detection
```python
# Define custom bottleneck detection rules
def custom_bottleneck_detector(metrics):
    if metrics['custom_metric'] > threshold:
        return CustomBottleneckType.CUSTOM_ISSUE
    return None

profiler.add_custom_bottleneck_detector(custom_bottleneck_detector)
```

### Export and Import
```python
# Save profiling results
profiler.save_profiling_results("my_profile.pt")

# Load previous results
profiler.load_profiling_results("my_profile.pt")
```

## Best Practices

### 1. **Start with Basic Profiling**
```python
config = BottleneckProfilerConfig(profiling_level=ProfilingLevel.BASIC)
profiler = BottleneckProfiler(config)
```

### 2. **Use Appropriate Sampling Intervals**
```python
# For real-time monitoring
config.sampling_interval = 0.1  # 100ms

# For production profiling
config.sampling_interval = 1.0  # 1 second
```

### 3. **Profile Representative Workloads**
```python
# Profile with realistic data sizes
sample_data = torch.randn(1000, 784)  # Representative batch size
```

### 4. **Monitor Multiple Metrics**
```python
# Enable all tracking for comprehensive analysis
config.enable_memory_tracking = True
config.enable_gpu_tracking = True
config.enable_cpu_tracking = True
config.enable_io_tracking = True
```

### 5. **Regular Report Generation**
```python
# Generate reports after each profiling session
if config.save_profiles:
    profiler.generate_optimization_report()
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce batch sizes
   - Enable gradient checkpointing
   - Use mixed precision training

2. **GPU Underutilization**
   - Move preprocessing to GPU
   - Increase batch sizes
   - Use GPU-optimized operations

3. **Slow Data Loading**
   - Increase `num_workers`
   - Enable `pin_memory`
   - Use `persistent_workers`

4. **High CPU Usage**
   - Vectorize operations
   - Use GPU acceleration
   - Implement caching

### Performance Tuning

```python
# Iterative optimization
for iteration in range(5):
    # Profile current configuration
    profile = profiler.profile_current_configuration()
    
    # Apply optimizations based on bottlenecks
    optimizations = profiler.get_optimization_suggestions()
    
    # Implement suggested changes
    profiler.apply_optimizations(optimizations)
    
    # Measure improvement
    improvement = profiler.measure_improvement()
    print(f"Iteration {iteration}: {improvement:.2f}% improvement")
```

## Examples

### Complete Training Loop Profiling

```python
# Setup profiler
config = BottleneckProfilerConfig(
    profiling_level=ProfilingLevel.COMPREHENSIVE,
    enable_real_time_monitoring=True
)
profiler = BottleneckProfiler(config)

# Start profiling session
session_id = profiler.start_profiling_session("complete_training")

try:
    for epoch in range(num_epochs):
        # Profile data loading
        with profiler.profile_operation(f"epoch_{epoch}_data_loading", 
                                      BottleneckType.DATA_LOADING):
            for batch_idx, (data, target) in enumerate(data_loader):
                # Profile preprocessing
                with profiler.profile_operation("preprocessing", 
                                              BottleneckType.PREPROCESSING):
                    data = preprocess_data(data)
                
                # Profile model inference
                with profiler.profile_operation("model_inference", 
                                              BottleneckType.CPU_COMPUTATION):
                    output = model(data)
                
                # Profile loss computation
                with profiler.profile_operation("loss_computation", 
                                              BottleneckType.CPU_COMPUTATION):
                    loss = criterion(output, target)
                
                # Profile backward pass
                with profiler.profile_operation("backward_pass", 
                                              BottleneckType.CPU_COMPUTATION):
                    loss.backward()
                    optimizer.step()

finally:
    # Stop profiling and generate report
    session = profiler.stop_profiling_session()
    summary = profiler.get_bottleneck_summary()
    
    print(f"Training completed with {len(summary.get('bottlenecks', []))} bottlenecks detected")
    
    # Generate comprehensive report
    profiler.generate_optimization_report()
```

## Conclusion

This comprehensive profiling system provides deep insights into performance bottlenecks in data loading and preprocessing operations. By using these tools effectively, you can:

- **Identify performance bottlenecks** before they become critical issues
- **Optimize data loading** for maximum throughput
- **Improve preprocessing efficiency** with GPU acceleration and caching
- **Monitor performance** in real-time during training
- **Generate actionable reports** for continuous improvement

Start with basic profiling and gradually increase the level of detail as you become familiar with the system. The automatic optimization suggestions and real-time monitoring will help you achieve optimal performance for your deep learning workflows.






