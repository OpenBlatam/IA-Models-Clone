# Performance Guide - Universal Model Benchmark AI

## 🚀 Performance Optimization

### Rate Limiting

```python
from core.rate_limiter import RateLimiter, RateLimit

limiter = RateLimiter()

# Check rate limit
allowed, remaining = limiter.check_rate_limit(
    key="user_123",
    limit=RateLimit(requests=100, window_seconds=60),
    algorithm="sliding_window"
)

if allowed:
    # Process request
    pass
else:
    # Rate limit exceeded
    print(f"Rate limit exceeded. Remaining: {remaining}")
```

### Metrics Collection

```python
from core.metrics import metrics_collector

# Record benchmark metrics
metrics_collector.record_benchmark(
    model="llama2-7b",
    benchmark="mmlu",
    accuracy=0.85,
    throughput=100.0,
    duration=120.0,
)

# Record errors
metrics_collector.record_error("TimeoutError", "inference")

# Update system metrics
metrics_collector.update_system_metrics(
    active_experiments=5,
    total_models=10,
    total_cost=500.0,
)

# Export metrics (Prometheus format)
metrics = metrics_collector.export_metrics()
```

### Performance Profiling

```python
from core.performance import performance_optimizer

# Profile a function
@performance_optimizer.profiler.profile
def my_function():
    # Your code here
    pass

# Get performance report
report = performance_optimizer.get_performance_report()
print(report)

# Get recommendations
recommendations = performance_optimizer.get_recommendations()
for rec in recommendations:
    print(rec)
```

### Memory Optimization

```python
from core.performance import MemoryOptimizer

optimizer = MemoryOptimizer()

# Get memory usage
usage = optimizer.get_memory_usage()
print(f"Memory: {usage['rss_mb']:.2f} MB")

# Optimize memory
result = optimizer.optimize_memory()
print(f"Freed: {result['freed_mb']:.2f} MB")

# System memory
system_mem = optimizer.get_system_memory()
print(f"System memory: {system_mem['percent']:.1f}% used")
```

### CPU Optimization

```python
from core.performance import CPUOptimizer

optimizer = CPUOptimizer()

# Get CPU usage
cpu = optimizer.get_cpu_usage()
print(f"CPU: {cpu['percent']:.1f}%")

# CPU frequency
freq = optimizer.get_cpu_frequency()
print(f"CPU Frequency: {freq.get('current_mhz', 0)} MHz")
```

## 📊 Prometheus Metrics

### Available Metrics

- `benchmark_requests_total` - Total benchmark requests
- `benchmark_duration_seconds` - Benchmark execution duration
- `benchmark_accuracy` - Benchmark accuracy
- `benchmark_throughput_tokens_per_second` - Throughput
- `active_experiments` - Active experiments count
- `total_models` - Total registered models
- `total_cost_usd` - Total cost
- `errors_total` - Error counts
- `api_requests_total` - API request counts
- `api_duration_seconds` - API request duration

### Accessing Metrics

```bash
# Get metrics endpoint
curl http://localhost:8000/metrics

# Or use Prometheus to scrape
# Add to prometheus.yml:
scrape_configs:
  - job_name: 'benchmark-api'
    static_configs:
      - targets: ['localhost:8000']
```

## ⚡ Performance Best Practices

### 1. Use Caching

```python
from core.cache import LRUCache

cache = LRUCache(max_size=1000)

# Cache expensive operations
@cache.cached
def expensive_operation(key):
    # Expensive computation
    return result
```

### 2. Batch Processing

```python
from core.batching import DynamicBatcher

batcher = DynamicBatcher(max_batch_size=32)

# Batch requests
for item in items:
    batcher.add(item)
    if batcher.is_ready():
        batch = batcher.get_batch()
        process_batch(batch)
```

### 3. Async Operations

```python
import asyncio
from core.async_helpers import batch_process_async

async def process_items(items):
    results = await batch_process_async(items, process_item, batch_size=10)
    return results
```

### 4. Memory Management

```python
# Regular memory cleanup
from core.performance import MemoryOptimizer

optimizer = MemoryOptimizer()
optimizer.optimize_memory()
```

### 5. Rate Limiting

```python
# Protect API endpoints
from core.rate_limiter import RateLimiter

limiter = RateLimiter()

@app.get("/api/v1/benchmark")
async def benchmark_endpoint(user_id: str):
    allowed, remaining = limiter.check_rate_limit(user_id)
    if not allowed:
        raise HTTPException(429, "Rate limit exceeded")
    # Process request
```

## 🔍 Monitoring

### Performance Dashboard

Access performance metrics at:
- `/metrics` - Prometheus metrics
- `/api/v1/statistics` - System statistics
- `/health` - Health check

### Logging

```python
from core.logging_config import setup_logging, log_performance

setup_logging(level="INFO")

@log_performance
def my_function():
    # Function execution is automatically logged
    pass
```

## 📈 Optimization Tips

1. **Use Rust for heavy computations** - Leverage Rust modules for performance-critical operations
2. **Enable caching** - Cache tokenization and results
3. **Batch requests** - Process multiple items together
4. **Use async** - For I/O-bound operations
5. **Monitor metrics** - Track performance continuously
6. **Optimize memory** - Regular garbage collection
7. **Rate limiting** - Protect against abuse
8. **Profile code** - Identify bottlenecks

## 🎯 Performance Targets

- API response time: < 100ms (p95)
- Benchmark execution: < 5 minutes (typical)
- Memory usage: < 8GB per worker
- CPU usage: < 80% average
- Throughput: > 100 tokens/second

## 🐛 Troubleshooting

### High Memory Usage

```python
# Check memory
from core.performance import MemoryOptimizer

optimizer = MemoryOptimizer()
usage = optimizer.get_memory_usage()
if usage['percent'] > 80:
    # Optimize
    optimizer.optimize_memory()
```

### Slow Performance

```python
# Profile functions
from core.performance import performance_optimizer

# Get slowest functions
slowest = performance_optimizer.profiler.get_slowest_functions(top_n=5)
for profile in slowest:
    print(f"{profile.function_name}: {profile.avg_time:.2f}s")
```

### High CPU Usage

```python
# Check CPU
from core.performance import CPUOptimizer

optimizer = CPUOptimizer()
cpu = optimizer.get_cpu_usage()
if cpu['percent'] > 80:
    # Consider scaling or optimization
    pass
```












