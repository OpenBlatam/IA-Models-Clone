# Parallel Processing Guide

## Overview

The video processing system implements an advanced parallel processing architecture designed for high-performance, enterprise-grade video content processing. This guide covers the complete system architecture, usage patterns, and optimization strategies.

## Architecture

### Core Components

```
video/
├── models/           # Data models with batch processing support
├── processors/       # Specialized processing engines
├── utils/           # Parallel processing utilities
├── examples/        # Usage examples
├── benchmarks/      # Performance benchmarks
└── docs/           # Documentation
```

### Parallel Processing Backends

The system supports multiple parallel processing backends:

1. **Threading** (`BackendType.THREAD`)
   - Best for I/O-bound operations
   - Low memory overhead
   - Good for network requests

2. **Multiprocessing** (`BackendType.PROCESS`)
   - Best for CPU-intensive operations
   - True parallelism
   - Higher memory usage

3. **Joblib** (`BackendType.JOBLIB`)
   - Optimized for scientific computing
   - Automatic backend selection
   - Good for ML operations

4. **Dask** (`BackendType.DASK`)
   - Distributed computing support
   - Advanced scheduling
   - Good for large datasets

5. **Async** (`BackendType.ASYNC`)
   - Non-blocking I/O operations
   - High concurrency
   - Best for web scraping

## Quick Start

### Basic Usage

```python
from onyx.server.features.video.processors.video_processor import create_high_performance_processor
from onyx.server.features.video.models.video_models import VideoClipRequest

# Create processor
processor = create_high_performance_processor()

# Generate requests
requests = [
    VideoClipRequest(
        youtube_url="https://youtube.com/watch?v=example",
        language="en",
        max_clip_length=60
    )
    for _ in range(100)
]

# Process in parallel
results = processor.process_batch_parallel(requests)
```

### Viral Content Processing

```python
from onyx.server.features.video.processors.viral_processor import create_high_performance_viral_processor

# Create viral processor
viral_processor = create_high_performance_viral_processor()

# Process with viral variants
viral_results = viral_processor.process_batch_parallel(
    requests,
    n_variants=5,
    audience_profile={'age': '18-35', 'interests': ['tech']}
)
```

## Advanced Configuration

### Custom Parallel Configuration

```python
from onyx.server.features.video.utils.parallel_utils import ParallelConfig, BackendType

# Custom configuration
config = ParallelConfig(
    max_workers=16,
    chunk_size=1000,
    timeout=60.0,
    backend=BackendType.PROCESS,
    use_uvloop=True,
    use_numba=True
)

# Create processor with custom config
processor = VideoClipProcessor(config)
```

### Backend Selection Strategy

The system automatically selects the best backend based on:

1. **Data size**: Small datasets → Threading, Large datasets → Multiprocessing
2. **Operation type**: I/O-bound → Async, CPU-bound → Multiprocessing
3. **Available resources**: CPU cores, memory, network bandwidth

```python
# Auto-backend selection
results = processor.process_batch_parallel(requests)

# Manual backend selection
results = processor.process_batch_parallel(requests, backend=BackendType.PROCESS)
```

## Performance Optimization

### Batch Processing Patterns

```python
# Optimal batch sizes
small_batch = 10-50 items      # Threading/Async
medium_batch = 50-500 items    # Joblib/Multiprocessing
large_batch = 500+ items       # Dask/Distributed

# Chunk processing for large datasets
for chunk in processor.chunk_data(large_dataset, chunk_size=1000):
    results = processor.process_batch_parallel(chunk)
```

### Memory Management

```python
# Memory-efficient processing
processor = create_high_performance_processor(
    memory_limit_gb=8,
    enable_garbage_collection=True
)

# Streaming processing for very large datasets
for batch in processor.stream_process(large_dataset):
    yield batch
```

### Caching Strategies

```python
# Enable result caching
processor = create_high_performance_processor(
    enable_caching=True,
    cache_ttl_hours=24
)

# Custom cache key generation
def custom_cache_key(request: VideoClipRequest) -> str:
    return f"{request.youtube_url}_{request.language}_{request.max_clip_length}"
```

## Specialized Processors

### Video Encoding Processor

```python
from onyx.server.features.video.processors.video_processor import VideoEncodingProcessor

encoder = VideoEncodingProcessor()

# Batch encoding
encoded_data = encoder.batch_encode_videos(video_results)

# Parallel encoding with custom settings
encoded_data = encoder.batch_encode_videos(
    video_results,
    format="mp4",
    quality="high",
    parallel_workers=8
)
```

### Viral Analytics Processor

```python
from onyx.server.features.video.processors.viral_processor import ViralAnalyticsProcessor

analytics = ViralAnalyticsProcessor()

# Batch performance analysis
performance_data = analytics.batch_analyze_performance(viral_results)

# Custom metrics
custom_metrics = analytics.batch_analyze_performance(
    viral_results,
    metrics=['engagement_rate', 'viral_coefficient', 'audience_reach']
)
```

## Error Handling & Resilience

### Robust Error Handling

```python
# Automatic retry with exponential backoff
processor = create_high_performance_processor(
    max_retries=3,
    retry_delay_seconds=1,
    exponential_backoff=True
)

# Custom error handling
def custom_error_handler(error: Exception, request: VideoClipRequest):
    logger.error(f"Failed to process {request.youtube_url}: {error}")
    return VideoClipResponse(error=str(error))

processor.set_error_handler(custom_error_handler)
```

### Circuit Breaker Pattern

```python
# Circuit breaker for external services
processor = create_high_performance_processor(
    circuit_breaker_enabled=True,
    failure_threshold=5,
    recovery_timeout_seconds=60
)
```

## Monitoring & Observability

### Performance Metrics

```python
# Get processing statistics
stats = processor.get_processing_stats()

# Monitor real-time performance
for metric in processor.get_performance_metrics():
    print(f"{metric.name}: {metric.value}")

# Export metrics to monitoring systems
processor.export_metrics_to_prometheus()
processor.export_metrics_to_datadog()
```

### Logging & Tracing

```python
# Structured logging
import structlog
logger = structlog.get_logger()

# Enable request tracing
processor = create_high_performance_processor(
    enable_tracing=True,
    trace_sampling_rate=0.1
)

# Custom logging
def custom_logger(level: str, message: str, **kwargs):
    logger.log(level, message, **kwargs)

processor.set_logger(custom_logger)
```

## Testing & Validation

### Unit Testing

```python
import pytest
from onyx.server.features.video.processors.video_processor import VideoClipProcessor

def test_parallel_processing():
    processor = VideoClipProcessor()
    requests = generate_test_requests(10)
    
    results = processor.process_batch_parallel(requests)
    
    assert len(results) == 10
    assert all(result.success for result in results)
```

### Performance Testing

```python
from onyx.server.features.video.benchmarks.performance_benchmark import run_comprehensive_benchmark

# Run performance benchmarks
benchmark_results = run_comprehensive_benchmark()

# Analyze results
for test_name, results in benchmark_results.items():
    print(f"{test_name}: {results}")
```

## Best Practices

### 1. Backend Selection

- **I/O Operations**: Use Async or Threading
- **CPU Operations**: Use Multiprocessing or Joblib
- **Large Datasets**: Use Dask or Distributed
- **Unknown Workload**: Use Auto-selection

### 2. Batch Sizing

- Start with small batches (10-50 items)
- Increase batch size based on performance
- Monitor memory usage
- Use chunking for very large datasets

### 3. Error Handling

- Always implement error handlers
- Use circuit breakers for external services
- Implement retry logic with backoff
- Log errors with context

### 4. Monitoring

- Monitor processing time per item
- Track success/failure rates
- Monitor memory usage
- Set up alerts for performance degradation

### 5. Resource Management

- Set appropriate worker limits
- Monitor CPU and memory usage
- Use connection pooling for external services
- Implement graceful shutdown

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```python
   # Reduce batch size
   processor = create_high_performance_processor(chunk_size=100)
   
   # Enable garbage collection
   processor = create_high_performance_processor(enable_gc=True)
   ```

2. **Timeout Issues**
   ```python
   # Increase timeout
   processor = create_high_performance_processor(timeout=120.0)
   
   # Use async for I/O operations
   results = await processor.process_batch_async(requests)
   ```

3. **Performance Issues**
   ```python
   # Profile performance
   stats = processor.get_processing_stats()
   
   # Try different backends
   results = processor.process_batch_parallel(requests, backend=BackendType.PROCESS)
   ```

### Debug Mode

```python
# Enable debug mode
processor = create_high_performance_processor(debug=True)

# Get detailed logs
logs = processor.get_debug_logs()
for log in logs:
    print(f"{log.timestamp}: {log.message}")
```

## API Reference

### VideoClipProcessor

```python
class VideoClipProcessor:
    def process_batch_parallel(
        self,
        requests: List[VideoClipRequest],
        backend: BackendType = BackendType.AUTO
    ) -> List[VideoClipResponse]
    
    async def process_batch_async(
        self,
        requests: List[VideoClipRequest]
    ) -> List[VideoClipResponse]
    
    def validate_batch(
        self,
        requests: List[VideoClipRequest]
    ) -> List[bool]
```

### ViralVideoProcessor

```python
class ViralVideoProcessor:
    def process_batch_parallel(
        self,
        requests: List[VideoClipRequest],
        n_variants: int = 5,
        audience_profile: Dict = None
    ) -> List[ViralVideoBatchResponse]
    
    def batch_generate_variants(
        self,
        results: List[ViralVideoBatchResponse]
    ) -> List[List[ViralVideoVariant]]
```

### ParallelConfig

```python
@dataclass
class ParallelConfig:
    max_workers: int = 8
    chunk_size: int = 1000
    timeout: float = 30.0
    backend: BackendType = BackendType.AUTO
    use_uvloop: bool = True
    use_numba: bool = False
    memory_limit_gb: int = 4
    enable_caching: bool = False
    enable_tracing: bool = False
    debug: bool = False
```

## Examples

See the `examples/` directory for complete working examples:

- `parallel_processing_examples.py`: Basic usage examples
- `performance_benchmark.py`: Performance testing
- `custom_processors.py`: Custom processor implementations

## Contributing

When contributing to the parallel processing system:

1. Follow the established patterns
2. Add comprehensive tests
3. Update documentation
4. Run performance benchmarks
5. Ensure backward compatibility 