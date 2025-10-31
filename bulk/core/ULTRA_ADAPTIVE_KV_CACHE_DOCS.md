# Ultra-Adaptive K/V Cache Engine Documentation

## Overview

The Ultra-Adaptive K/V Cache Engine is a high-performance, production-ready caching and optimization system designed for TruthGPT bulk processing. It provides intelligent cache management, multi-GPU support, persistence, and advanced performance monitoring.

## Key Features

### 🚀 Performance
- **Multi-GPU Support**: Automatic detection and load balancing across multiple GPUs
- **Adaptive Caching**: Intelligent cache strategies based on workload patterns
- **Dynamic Batching**: Automatic batch size optimization
- **Parallel Processing**: Configurable worker pool for concurrent operations

### 💾 Persistence & Reliability
- **Disk Cache Persistence**: Cache survives restarts
- **Checkpointing**: Periodic state snapshots for recovery
- **Session Management**: Efficient session tracking and cleanup
- **Error Recovery**: Comprehensive error tracking and handling

### 📊 Observability
- **Detailed Metrics**: P50, P95, P99 latencies, throughput, error rates
- **Memory Tracking**: Real-time memory usage monitoring
- **GPU Utilization**: Per-GPU workload and utilization tracking
- **Performance Stats**: Comprehensive engine statistics

## Installation

```python
from bulk.core.ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    AdaptiveConfig,
    AdaptiveMode,
    TruthGPTIntegration
)
```

## Quick Start

### Basic Usage

```python
from ultra_adaptive_kv_cache_engine import (
    UltraAdaptiveKVCacheEngine,
    AdaptiveConfig,
    TruthGPTIntegration
)

# Create engine with default config
engine = TruthGPTIntegration.create_engine_for_truthgpt()

# Process a request
request = {
    'text': 'Your input text here',
    'max_length': 100,
    'temperature': 0.7,
    'session_id': 'user_123'
}

result = await engine.process_request(request)
print(result['response']['text'])
```

### Configuration Options

```python
config = AdaptiveConfig(
    # Core settings
    model_name="truthgpt-base",
    model_size="medium",
    max_sequence_length=4096,
    batch_size=1,
    
    # Adaptive settings
    adaptive_mode=AdaptiveMode.AUTO,  # AUTO, BULK, STREAMING, INTERACTIVE, BATCH
    auto_scale=True,
    dynamic_batching=True,
    load_balancing=True,
    
    # K/V Cache settings
    use_kv_cache=True,
    cache_size=8192,
    cache_strategy=CacheStrategy.ADAPTIVE,
    compression_ratio=0.3,
    quantization_bits=8,
    
    # Performance settings
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_parallel_processing=True,
    num_workers=4,
    
    # Memory settings
    memory_strategy=MemoryStrategy.BALANCED,
    max_memory_usage=0.8,
    memory_cleanup_interval=100,
    
    # Monitoring
    enable_metrics=True,
    enable_profiling=True,
    log_level="INFO",
    
    # Persistence
    enable_cache_persistence=True,
    cache_persistence_path="./cache_data",
    checkpoint_interval=300,  # seconds
    
    # Multi-GPU
    use_multi_gpu=True,
    gpu_load_balancing=True,
    
    # Advanced optimizations
    enable_checkpointing=True,
    enable_prefetching=True,
    enable_speculative_execution=False
)

engine = UltraAdaptiveKVCacheEngine(config)
```

## Advanced Usage

### Pre-configured Engines

```python
# For bulk processing
bulk_engine = TruthGPTIntegration.create_bulk_engine()

# For streaming
streaming_engine = TruthGPTIntegration.create_streaming_engine()

# For general TruthGPT use
truthgpt_engine = TruthGPTIntegration.create_engine_for_truthgpt()
```

### Batch Processing

```python
requests = [
    {'text': 'Request 1', 'max_length': 50, 'temperature': 0.7, 'session_id': 's1'},
    {'text': 'Request 2', 'max_length': 50, 'temperature': 0.7, 'session_id': 's2'},
    # ... more requests
]

results = await engine.process_batch(requests)
for result in results:
    if result['success']:
        print(result['response']['text'])
```

### Workload Adaptation

```python
workload_info = {
    'batch_size': 10,
    'sequence_length': 2048,
    'request_rate': 15.0,
    'memory_usage': 0.85
}

adaptation_result = engine.adapt_to_workload(workload_info)
print(f"Adapted: {adaptation_result['adapted']}")
print(f"New config: {adaptation_result['new_config']}")
print(f"Performance impact: {adaptation_result['performance_impact']}")
```

### Performance Monitoring

```python
# Get comprehensive stats
stats = engine.get_performance_stats()

print(f"Total requests: {stats['engine_stats']['total_requests']}")
print(f"Average latency: {stats['engine_stats']['avg_response_time']*1000:.2f}ms")
print(f"P95 latency: {stats['engine_stats']['p95_response_time']*1000:.2f}ms")
print(f"Throughput: {stats['engine_stats']['throughput_tokens_per_sec']:.2f} tokens/s")
print(f"Memory usage: {stats['memory_usage']*100:.2f}%")
print(f"Active sessions: {stats['active_sessions']}")
print(f"GPU utilization: {stats['gpu_utilization']}")
```

### Cache Management

```python
# Clear all caches
engine.clear_cache()

# Cleanup old sessions (older than 1 hour)
engine.cleanup_sessions(max_age=3600)

# Get cache statistics
stats = engine.get_performance_stats()
cache_stats = stats.get('cache_persistence', {})
print(f"Cache size: {cache_stats.get('cache_size_bytes', 0) / (1024**2):.2f} MB")
print(f"Cache files: {cache_stats.get('cache_files', 0)}")
```

### Checkpointing

```python
# Manual checkpoint save
engine._save_checkpoint()

# Load latest checkpoint
success = engine.load_checkpoint()
if success:
    print("Checkpoint loaded successfully")

# Load specific checkpoint
success = engine.load_checkpoint("/path/to/checkpoint.json")
```

## Multi-GPU Configuration

The engine automatically detects and uses available GPUs. To enable multi-GPU:

```python
config = AdaptiveConfig(
    use_multi_gpu=True,
    gpu_load_balancing=True,  # Automatic load balancing
    # ... other settings
)
```

The engine will:
- Detect all available GPUs
- Balance workload across GPUs
- Track utilization per GPU
- Select optimal GPU for each request

## Cache Persistence

Enable disk persistence for cache to survive restarts:

```python
config = AdaptiveConfig(
    enable_cache_persistence=True,
    cache_persistence_path="./cache_data",  # Directory for cache files
    # ... other settings
)
```

Cache files are stored in:
- `cache_data/sessions/` - Session cache files
- `cache_data/checkpoints/` - Engine checkpoints
- `cache_data/metadata/` - Cache metadata

## Error Handling

The engine tracks errors and provides detailed error information:

```python
result = await engine.process_request(request)

if not result['success']:
    print(f"Error: {result['error']}")
    print(f"Processing time: {result['processing_time']}s")
    
# Get error history
stats = engine.get_performance_stats()
recent_errors = stats.get('recent_errors', [])
error_count = stats.get('error_count', 0)
```

## Performance Tuning

### For High Throughput

```python
config = AdaptiveConfig(
    num_workers=8,  # More workers
    dynamic_batching=True,
    cache_size=16384,  # Larger cache
    compression_ratio=0.5,  # More compression
    quantization_bits=4,  # More aggressive quantization
    use_parallel_processing=True
)
```

### For Low Latency

```python
config = AdaptiveConfig(
    num_workers=2,  # Fewer workers, less overhead
    cache_strategy=CacheStrategy.SPEED,
    memory_strategy=MemoryStrategy.SPEED,
    enable_prefetching=True,
    use_mixed_precision=True
)
```

### For Memory Efficiency

```python
config = AdaptiveConfig(
    compression_ratio=0.5,  # Higher compression
    quantization_bits=4,  # Lower precision
    memory_strategy=MemoryStrategy.AGGRESSIVE,
    max_memory_usage=0.6,  # Lower memory limit
    memory_cleanup_interval=50  # More frequent cleanup
)
```

## Testing

Run comprehensive tests:

```bash
pytest agents/backend/onyx/server/features/bulk/core/test_ultra_adaptive_kv_cache.py -v
```

## Benchmarking

Run performance benchmarks:

```bash
# Full benchmark suite
python agents/backend/onyx/server/features/bulk/core/ultra_adaptive_kv_cache_benchmark.py --mode full

# Single request benchmark
python agents/backend/onyx/server/features/bulk/core/ultra_adaptive_kv_cache_benchmark.py --mode single --iterations 1000

# Batch processing benchmark
python agents/backend/onyx/server/features/bulk/core/ultra_adaptive_kv_cache_benchmark.py --mode batch --iterations 50

# Concurrent load benchmark
python agents/backend/onyx/server/features/bulk/core/ultra_adaptive_kv_cache_benchmark.py --mode concurrent --iterations 500

# Cache performance benchmark
python agents/backend/onyx/server/features/bulk/core/ultra_adaptive_kv_cache_benchmark.py --mode cache --iterations 200
```

## Architecture

### Components

1. **Cache Engine**: Manages K/V cache with adaptive strategies
2. **Decoder**: Handles text generation with optimizations
3. **Optimizer**: Optimizes cache usage and performance
4. **Session Manager**: Tracks and manages active sessions
5. **GPU Manager**: Handles multi-GPU load balancing
6. **Persistence Layer**: Manages disk cache and checkpoints
7. **Metrics Collector**: Gathers performance statistics

### Data Flow

```
Request → Session Check → Cache Lookup → GPU Selection → Processing → Cache Update → Response
```

## Best Practices

1. **Session Management**: Use meaningful session IDs that group related requests
2. **Batch Processing**: Process multiple requests in batches for better throughput
3. **Monitoring**: Regularly check performance stats and adapt configuration
4. **Cache Tuning**: Adjust cache size based on available memory
5. **Cleanup**: Periodically cleanup old sessions to free memory
6. **Checkpointing**: Enable checkpointing for long-running processes
7. **Error Handling**: Always check `success` flag in results

## Troubleshooting

### High Memory Usage
- Reduce `cache_size`
- Increase `compression_ratio`
- Lower `quantization_bits`
- Enable `memory_strategy=MemoryStrategy.AGGRESSIVE`

### Low Throughput
- Increase `num_workers`
- Enable `dynamic_batching`
- Use larger `cache_size`
- Check GPU utilization

### High Latency
- Reduce `num_workers` (less overhead)
- Enable `prefetching`
- Use `cache_strategy=CacheStrategy.SPEED`
- Check network/disk I/O

### Cache Misses
- Increase `cache_size`
- Improve session ID strategy (reuse sessions)
- Enable persistence for warm cache

## API Reference

### UltraAdaptiveKVCacheEngine

#### Methods

- `process_request(request: Dict) -> Dict`: Process single request
- `process_batch(requests: List[Dict]) -> List[Dict]`: Process batch of requests
- `adapt_to_workload(workload_info: Dict) -> Dict`: Adapt to workload
- `get_performance_stats() -> Dict`: Get performance statistics
- `clear_cache()`: Clear all caches
- `cleanup_sessions(max_age: int)`: Cleanup old sessions
- `shutdown()`: Graceful shutdown
- `load_checkpoint(path: Optional[str]) -> bool`: Load checkpoint

### AdaptiveConfig

See configuration options above for all available settings.

## License

Part of the TruthGPT project.

