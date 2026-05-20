# Performance Guide

## Overview

This guide covers performance optimization, streaming processing, and intelligent caching.

## Performance Optimizer

### Automatic Optimization

The performance optimizer learns from operations and adjusts parameters automatically.

```python
from image_upscaling_ai.models import PerformanceOptimizer

optimizer = PerformanceOptimizer(
    target_throughput=2.0,  # Target: 2 images/second
    max_memory_mb=2048
)

# Record operations
optimizer.record_operation(
    operation_time=0.5,
    memory_usage_mb=512,
    gpu_utilization=0.8,
    cache_hit=True
)

# Get optimal settings
batch_size = optimizer.get_optimal_batch_size()
tile_size = optimizer.get_optimal_tile_size()
concurrency = optimizer.get_optimal_concurrency()

# Get statistics
stats = optimizer.get_statistics()
print(f"Average throughput: {stats['avg_throughput']:.2f} img/s")
print(f"Optimal batch size: {stats['optimal_batch_size']}")
```

### Features

- **Automatic Parameter Tuning**: Adjusts batch size, tile size, concurrency
- **Performance Monitoring**: Tracks operation times, memory, GPU/CPU usage
- **Throughput Optimization**: Maximizes images per second
- **Memory Management**: Optimizes for available memory

## Streaming Processor

### Real-Time Processing

Process images as they arrive in real-time.

```python
from image_upscaling_ai.models import StreamingProcessor
from image_upscaling_ai.models import RealESRGANModelManager

manager = RealESRGANModelManager()

# Create streaming processor
processor = StreamingProcessor(
    upscale_func=lambda img, scale: manager.upscale_async(img, scale),
    max_queue_size=10,
    timeout=30.0
)

# Process stream
async def image_stream():
    for image_path in image_paths:
        image = Image.open(image_path)
        yield image

async def process_stream():
    async for upscaled in processor.process_stream(
        image_stream(),
        scale_factor=4.0,
        progress_callback=lambda current, total: print(f"{current}/{total}")
    ):
        # Process upscaled image
        upscaled.save(f"output_{time.time()}.png")
```

### Features

- **Async Streaming**: Process images as they arrive
- **Backpressure Management**: Queue size limits
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Continues on errors
- **Timeout Protection**: Prevents hanging operations

## Intelligent Cache

### Advanced Caching

Intelligent cache with learning and optimization.

```python
from image_upscaling_ai.models import IntelligentCache

cache = IntelligentCache(
    cache_dir="./cache",
    max_size_mb=1000,
    max_entries=100,
    ttl=86400  # 24 hours
)

# Get from cache
cached = cache.get(image_path, 4.0, "realesrgan")
if cached:
    print("Cache hit!")
    return cached

# Upscale
upscaled = upscaler.upscale(image, 4.0)

# Cache result
cache.set(
    image_path,
    4.0,
    "realesrgan",
    upscaled,
    quality_score=0.85
)

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
```

### Features

- **LRU Eviction**: Least recently used entries removed first
- **Quality-Based Prioritization**: High-quality results prioritized
- **Access Pattern Learning**: Learns from access patterns
- **Automatic Cleanup**: Removes expired entries
- **Size Management**: Respects memory limits

### Cache Management

```python
# Cleanup expired entries
expired_count = cache.cleanup_expired()
print(f"Removed {expired_count} expired entries")

# Clear all cache
cache.clear()

# Get detailed statistics
stats = cache.get_stats()
print(f"Entries: {stats['entries']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Complete Performance Pipeline

### Optimized Processing

```python
from image_upscaling_ai.models import (
    PerformanceOptimizer,
    IntelligentCache,
    RealESRGANModelManager,
    AdaptivePreprocessor,
    AdaptivePostprocessor
)

# Initialize components
optimizer = PerformanceOptimizer(target_throughput=2.0)
cache = IntelligentCache(cache_dir="./cache", max_size_mb=2000)
manager = RealESRGANModelManager()
preprocessor = AdaptivePreprocessor()
postprocessor = AdaptivePostprocessor()

async def optimized_upscale(image_path: str, scale_factor: float):
    # Check cache
    cached = cache.get(image_path, scale_factor, "realesrgan")
    if cached:
        optimizer.record_operation(0.01, cache_hit=True)
        return cached
    
    start_time = time.time()
    
    # Preprocess
    image = Image.open(image_path)
    preprocessed = preprocessor.preprocess(image, mode="auto")
    
    # Upscale with optimal settings
    batch_size = optimizer.get_optimal_batch_size()
    upscaled = await manager.upscale_async(
        preprocessed,
        scale_factor,
        max_concurrent=optimizer.get_optimal_concurrency()
    )
    
    # Postprocess
    final = postprocessor.postprocess(upscaled, original=image, mode="auto")
    
    # Record operation
    operation_time = time.time() - start_time
    optimizer.record_operation(
        operation_time,
        memory_usage_mb=512,
        cache_hit=False
    )
    
    # Cache result
    cache.set(image_path, scale_factor, "realesrgan", final, quality_score=0.85)
    
    return final
```

## Best Practices

### 1. Performance Optimization

```python
# Set realistic targets
optimizer = PerformanceOptimizer(
    target_throughput=1.0,  # Start conservative
    max_memory_mb=2048
)

# Record all operations
for operation in operations:
    optimizer.record_operation(
        operation_time,
        memory_usage_mb,
        cache_hit=cache_hit
    )

# Use optimal settings
batch_size = optimizer.get_optimal_batch_size()
```

### 2. Caching Strategy

```python
# Use intelligent cache
cache = IntelligentCache(
    max_size_mb=2000,  # Enough for ~100 images
    max_entries=100,
    ttl=86400  # 24 hours
)

# Always check cache first
cached = cache.get(path, scale, method)
if cached:
    return cached

# Cache high-quality results
if quality_score > 0.8:
    cache.set(path, scale, method, result, quality_score)
```

### 3. Streaming Processing

```python
# Use streaming for large batches
processor = StreamingProcessor(
    upscale_func=upscale_func,
    max_queue_size=10,  # Prevent memory issues
    timeout=30.0
)

# Process with progress tracking
async for result in processor.process_stream(
    image_stream,
    scale_factor,
    progress_callback=update_progress
):
    process_result(result)
```

### 4. Memory Management

```python
# Monitor memory usage
import psutil

process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024

# Adjust based on available memory
if memory_mb > 1500:
    # High memory, can use larger batches
    batch_size = 4
else:
    # Low memory, use smaller batches
    batch_size = 1
```

## Monitoring

### Performance Metrics

```python
# Get optimizer statistics
stats = optimizer.get_statistics()
print(f"Throughput: {stats['avg_throughput']:.2f} img/s")
print(f"Avg time: {stats['avg_operation_time']:.2f}s")
print(f"Optimal batch: {stats['optimal_batch_size']}")

# Get cache statistics
cache_stats = cache.get_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
print(f"Cache size: {cache_stats['total_size_mb']:.2f} MB")
```

### Real-Time Monitoring

```python
import time

start = time.time()
result = await upscale_image(image, 4.0)
elapsed = time.time() - start

# Record for optimization
optimizer.record_operation(
    elapsed,
    memory_usage_mb=get_memory_usage(),
    cache_hit=was_cached
)

# Log performance
logger.info(f"Upscaled in {elapsed:.2f}s, throughput: {1/elapsed:.2f} img/s")
```

## Troubleshooting

### Low Throughput

```python
# Check optimizer recommendations
stats = optimizer.get_statistics()
if stats['avg_throughput'] < target:
    # Increase concurrency
    concurrency = optimizer.get_optimal_concurrency()
    # Or reduce batch size
    batch_size = max(1, optimizer.get_optimal_batch_size() - 1)
```

### High Memory Usage

```python
# Reduce cache size
cache = IntelligentCache(max_size_mb=1000)  # Reduce from 2000

# Cleanup expired
cache.cleanup_expired()

# Use smaller tiles
tile_size = optimizer.get_optimal_tile_size()
```

### Cache Misses

```python
# Check cache statistics
stats = cache.get_stats()
if stats['hit_rate'] < 0.5:
    # Increase cache size
    cache = IntelligentCache(max_size_mb=3000)
    # Or increase TTL
    cache = IntelligentCache(ttl=172800)  # 48 hours
```

## Summary

The performance system provides:
1. **Automatic Optimization**: Learns and adapts
2. **Streaming Processing**: Real-time processing
3. **Intelligent Caching**: Smart cache management
4. **Performance Monitoring**: Track and optimize

Use these features for maximum performance and efficiency.


