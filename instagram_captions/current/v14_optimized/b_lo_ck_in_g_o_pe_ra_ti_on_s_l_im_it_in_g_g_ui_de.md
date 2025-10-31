# Blocking Operations Limiting Guide for Instagram Captions API v14.0

## Overview

This guide covers the comprehensive blocking operations limiting system implemented for the Instagram Captions API v14.0. The system provides rate limiting, concurrency control, circuit breaker patterns, and resource management to prevent blocking operations from overwhelming the system.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Rate Limiting](#rate-limiting)
3. [Concurrency Control](#concurrency-control)
4. [Circuit Breaker Pattern](#circuit-breaker-pattern)
5. [Operation Types](#operation-types)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [API Endpoints](#api-endpoints)
9. [Best Practices](#best-practices)
10. [Monitoring and Metrics](#monitoring-and-metrics)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)

## Core Concepts

### What are Blocking Operations?

Blocking operations are tasks that can potentially block the event loop or consume excessive resources:

- **I/O Operations**: Database queries, file operations, network requests
- **CPU-Intensive Tasks**: AI model inference, data processing, encryption
- **Resource-Heavy Operations**: Large file uploads, batch processing
- **External API Calls**: Third-party service integrations

### Why Limit Blocking Operations?

- **Prevent Resource Exhaustion**: Avoid overwhelming system resources
- **Maintain Responsiveness**: Keep the API responsive under load
- **Fair Resource Distribution**: Ensure fair access for all users
- **System Stability**: Prevent cascading failures
- **Cost Control**: Manage resource usage and costs

## Rate Limiting

### Token Bucket Algorithm

The system uses a token bucket algorithm for rate limiting:

```python
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket"""
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
```

### Sliding Window Counter

Additional sliding window counter for precise rate limiting:

```python
class SlidingWindowCounter:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def add_request(self, timestamp: float = None) -> int:
        """Add a request and return current count"""
        if timestamp is None:
            timestamp = time.time()
        
        async with self._lock:
            # Remove old requests outside the window
            cutoff = timestamp - self.window_size
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Add new request
            self.requests.append(timestamp)
            return len(self.requests)
```

### Rate Limit Configuration

```python
@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_size: int = 60  # seconds
    cooldown_period: int = 300  # seconds
```

## Concurrency Control

### Semaphore-Based Limiting

The system uses semaphores to control concurrent operations:

```python
class ConcurrencyLimiter:
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.user_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.user_locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
    
    @asynccontextmanager
    async def limit_concurrency(self, user_id: str = None):
        """Context manager for concurrency limiting"""
        # Global semaphore
        async with self.semaphore:
            # User-specific semaphore
            if user_id:
                user_sem = await self.get_user_semaphore(user_id)
                async with user_sem:
                    yield
            else:
                yield
```

### Concurrency Configuration

```python
@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency control"""
    max_concurrent_requests: int = 50
    max_concurrent_per_user: int = 5
    queue_size: int = 100
    timeout_seconds: int = 30
```

## Circuit Breaker Pattern

### Circuit Breaker Implementation

The circuit breaker pattern prevents cascading failures:

```python
class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = LimiterState.ALLOW
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == LimiterState.CIRCUIT_OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = LimiterState.CIRCUIT_HALF_OPEN
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise e
```

### Circuit Breaker States

```python
class LimiterState(Enum):
    """States of the rate limiter"""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_HALF_OPEN = "circuit_half_open"
```

### Circuit Breaker Configuration

```python
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    monitor_interval: int = 10
```

## Operation Types

### Supported Operation Types

```python
class OperationType(Enum):
    """Types of operations that can be limited"""
    CAPTION_GENERATION = "caption_generation"
    BATCH_PROCESSING = "batch_processing"
    AI_MODEL_LOADING = "ai_model_loading"
    DATABASE_QUERY = "database_query"
    EXTERNAL_API_CALL = "external_api_call"
    FILE_OPERATION = "file_operation"
    CACHE_OPERATION = "cache_operation"
    HEAVY_COMPUTATION = "heavy_computation"
```

### Operation-Specific Configurations

```python
operation_configs = {
    OperationType.CAPTION_GENERATION: {
        "rate_limit": RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
        "concurrency": ConcurrencyConfig(max_concurrent_requests=20, max_concurrent_per_user=3),
        "circuit_breaker": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
    },
    OperationType.BATCH_PROCESSING: {
        "rate_limit": RateLimitConfig(requests_per_minute=10, requests_per_hour=100),
        "concurrency": ConcurrencyConfig(max_concurrent_requests=10, max_concurrent_per_user=2),
        "circuit_breaker": CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
    },
    OperationType.AI_MODEL_LOADING: {
        "rate_limit": RateLimitConfig(requests_per_minute=5, requests_per_hour=50),
        "concurrency": ConcurrencyConfig(max_concurrent_requests=5, max_concurrent_per_user=1),
        "circuit_breaker": CircuitBreakerConfig(failure_threshold=2, recovery_timeout=120)
    }
}
```

## Configuration

### Global Configuration

```python
# Default configurations
default_rate_config = RateLimitConfig()
default_concurrency_config = ConcurrencyConfig()
default_circuit_config = CircuitBreakerConfig()

# Operation type configurations
operation_configs = {
    OperationType.CAPTION_GENERATION: {
        "rate_limit": RateLimitConfig(requests_per_minute=30, requests_per_hour=500),
        "concurrency": ConcurrencyConfig(max_concurrent_requests=20, max_concurrent_per_user=3),
        "circuit_breaker": CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
    }
}
```

### Environment Variables

```bash
# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000
RATE_LIMIT_BURST_LIMIT=10

# Concurrency Control
MAX_CONCURRENT_REQUESTS=50
MAX_CONCURRENT_PER_USER=5
CONCURRENCY_TIMEOUT=30

# Circuit Breaker
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
```

## Usage Examples

### Decorator Usage

```python
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="single_generation",
    user_id_param="user_id"
)
async def generate_caption(
    request: OptimizedRequest,
    user_id: str
) -> OptimizedResponse:
    """Generate caption with rate limiting and concurrency control"""
    return await optimized_engine.generate_caption(request)

@limit_blocking_thread_operations(
    operation_type=OperationType.HEAVY_COMPUTATION,
    identifier="data_processing",
    user_id_param="user_id"
)
async def process_large_dataset(
    data: List[Any],
    user_id: str
) -> Dict[str, Any]:
    """Process large dataset in thread pool with limiting"""
    return await heavy_computation_function(data)
```

### Context Manager Usage

```python
async def batch_process_with_limits(batch_data: List[Any], user_id: str):
    """Process batch with manual limiting"""
    
    # Rate limiting
    async with rate_limit_context(OperationType.BATCH_PROCESSING, f"batch_{user_id}"):
        # Concurrency limiting
        async with concurrency_limit_context(OperationType.BATCH_PROCESSING, user_id):
            results = []
            for item in batch_data:
                result = await process_item(item)
                results.append(result)
            return results
```

### Direct API Usage

```python
async def custom_operation_with_limits(operation_type: OperationType, func: Callable, user_id: str):
    """Execute function with all limits applied"""
    
    return await blocking_limiter.execute_with_limits(
        operation_type=operation_type,
        func=func,
        identifier="custom_operation",
        user_id=user_id,
        *args,
        **kwargs
    )
```

## API Endpoints

### Caption Generation with Limiting

```http
POST /api/v14/generate
Content-Type: application/json
Authorization: Bearer <api_key>

{
    "content_description": "A beautiful sunset over the ocean",
    "style": "casual",
    "hashtag_count": 15
}
```

### Batch Processing with Enhanced Limiting

```http
POST /api/v14/batch
Content-Type: application/json
Authorization: Bearer <api_key>

{
    "requests": [
        {
            "content_description": "Sunset photo",
            "style": "casual"
        },
        {
            "content_description": "Mountain landscape",
            "style": "professional"
        }
    ]
}
```

### Priority Generation

```http
POST /api/v14/generate/priority
Content-Type: application/json
Authorization: Bearer <api_key>

{
    "content_description": "Urgent caption needed",
    "style": "casual",
    "priority_level": 5
}
```

### Limits Status

```http
GET /api/v14/limits/status
Authorization: Bearer <api_key>
```

### Reset Limits

```http
POST /api/v14/limits/reset
Authorization: Bearer <api_key>
```

## Best Practices

### 1. Choose Appropriate Operation Types

```python
# ✅ Good - Specific operation type
@limit_blocking_operations(OperationType.CAPTION_GENERATION)
async def generate_caption(request):
    pass

# ❌ Bad - Generic operation type
@limit_blocking_operations(OperationType.HEAVY_COMPUTATION)
async def generate_caption(request):
    pass
```

### 2. Use Meaningful Identifiers

```python
# ✅ Good - Descriptive identifier
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="priority_generation"
)

# ❌ Bad - Generic identifier
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="default"
)
```

### 3. Handle Rate Limit Errors Gracefully

```python
async def generate_caption_with_fallback(request):
    try:
        return await generate_caption(request)
    except HTTPException as e:
        if e.status_code == 429:  # Rate limit exceeded
            # Return cached result or fallback
            return get_cached_caption(request)
        raise
```

### 4. Use Thread Pool for Blocking Operations

```python
@limit_blocking_thread_operations(OperationType.HEAVY_COMPUTATION)
async def process_large_file(file_path: str):
    """Process large file in thread pool"""
    return await blocking_limiter.execute_blocking_in_thread(
        OperationType.HEAVY_COMPUTATION,
        process_file_sync,
        "file_processing",
        user_id,
        file_path
    )
```

### 5. Monitor and Adjust Limits

```python
async def monitor_and_adjust_limits():
    """Monitor performance and adjust limits dynamically"""
    metrics = await blocking_limiter.get_metrics()
    
    for operation_type, operation_metrics in metrics.items():
        error_rate = operation_metrics["error_rate"]
        
        if error_rate > 0.1:  # 10% error rate
            # Reduce limits
            await adjust_limits(operation_type, "reduce")
        elif error_rate < 0.01:  # 1% error rate
            # Increase limits
            await adjust_limits(operation_type, "increase")
```

## Monitoring and Metrics

### Available Metrics

```python
@dataclass
class OperationMetrics:
    """Metrics for operation tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    throttled_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: float = 0.0
    error_rate: float = 0.0
```

### Getting Metrics

```python
# Get all metrics
all_metrics = await blocking_limiter.get_metrics()

# Get specific operation metrics
caption_metrics = await blocking_limiter.get_metrics(OperationType.CAPTION_GENERATION)

# Reset metrics
await blocking_limiter.reset_metrics(OperationType.CAPTION_GENERATION)
```

### Metrics Dashboard

```python
async def get_metrics_dashboard():
    """Get comprehensive metrics dashboard"""
    metrics = await blocking_limiter.get_metrics()
    
    dashboard = {
        "timestamp": time.time(),
        "overall": {
            "total_requests": sum(m["total_requests"] for m in metrics.values()),
            "success_rate": sum(m["successful_requests"] for m in metrics.values()) / 
                          max(1, sum(m["total_requests"] for m in metrics.values())),
            "average_response_time": sum(m["average_response_time"] for m in metrics.values()) / 
                                   max(1, len(metrics))
        },
        "by_operation": metrics,
        "alerts": []
    }
    
    # Check for alerts
    for operation_type, operation_metrics in metrics.items():
        if operation_metrics["error_rate"] > 0.1:
            dashboard["alerts"].append({
                "operation": operation_type,
                "type": "high_error_rate",
                "value": operation_metrics["error_rate"]
            })
    
    return dashboard
```

## Troubleshooting

### Common Issues

#### 1. Rate Limit Exceeded

**Symptoms**: 429 status code, "Rate limit exceeded" error

**Solutions**:
- Check current rate limits
- Implement exponential backoff
- Use batch processing for multiple requests

```python
async def handle_rate_limit_exceeded():
    try:
        return await generate_caption(request)
    except HTTPException as e:
        if e.status_code == 429:
            # Wait and retry with exponential backoff
            await asyncio.sleep(2 ** retry_count)
            return await generate_caption(request)
```

#### 2. Circuit Breaker Open

**Symptoms**: 503 status code, "Circuit breaker is open" error

**Solutions**:
- Wait for recovery timeout
- Implement fallback mechanisms
- Check underlying service health

```python
async def handle_circuit_breaker_open():
    try:
        return await generate_caption(request)
    except HTTPException as e:
        if e.status_code == 503:
            # Use fallback service or cached result
            return get_fallback_caption(request)
```

#### 3. Concurrency Limit Exceeded

**Symptoms**: Request queuing, slow response times

**Solutions**:
- Reduce concurrent requests
- Implement request queuing
- Scale horizontally

```python
async def handle_concurrency_limit():
    # Use priority queuing
    if is_priority_request(request):
        return await generate_caption_priority(request)
    else:
        return await generate_caption(request)
```

### Debugging Tools

#### 1. Limits Status Endpoint

```python
# Check current limits status
status = await get_limits_status()
print(f"Rate limit tokens: {status['caption_generation']['rate_limit']['tokens_available']}")
print(f"Concurrent requests: {status['caption_generation']['metrics']['total_requests']}")
```

#### 2. Metrics Monitoring

```python
# Monitor real-time metrics
async def monitor_metrics():
    while True:
        metrics = await blocking_limiter.get_metrics()
        for operation, data in metrics.items():
            print(f"{operation}: {data['error_rate']:.2%} error rate")
        await asyncio.sleep(60)
```

#### 3. Performance Profiling

```python
# Profile operation performance
async def profile_operation():
    start_time = time.time()
    result = await generate_caption(request)
    processing_time = time.time() - start_time
    
    print(f"Operation took {processing_time:.2f} seconds")
    return result
```

## Performance Optimization

### 1. Optimize Rate Limiting

```python
# Use efficient rate limiting algorithms
class OptimizedTokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
```

### 2. Implement Caching

```python
# Cache rate limit results
@with_cache(ttl=60)
async def check_rate_limit_cached(operation_type: OperationType, identifier: str):
    return await blocking_limiter.check_rate_limit(operation_type, identifier)
```

### 3. Use Connection Pooling

```python
# Reuse connections for external API calls
async def external_api_call_with_pooling():
    async with http_client() as client:
        async with client.get(url) as response:
            return await response.json()
```

### 4. Implement Request Batching

```python
# Batch multiple requests
async def batch_requests(requests: List[OptimizedRequest]):
    return await blocking_limiter.execute_with_limits(
        OperationType.BATCH_PROCESSING,
        optimized_engine.batch_generate,
        "batch_processing",
        user_id,
        requests
    )
```

### 5. Use Background Processing

```python
# Process heavy operations in background
async def background_processing(request: OptimizedRequest):
    # Start background task
    background_tasks.add_task(process_heavy_operation, request)
    
    # Return immediate response
    return OptimizedResponse(
        request_id=generate_request_id(),
        status="processing",
        message="Request queued for processing"
    )
```

## Conclusion

The blocking operations limiting system provides comprehensive protection against resource exhaustion and ensures fair resource distribution. Key benefits include:

- **Resource Protection**: Prevent system overload
- **Fair Access**: Ensure equal access for all users
- **System Stability**: Prevent cascading failures
- **Performance Monitoring**: Track and optimize performance
- **Flexible Configuration**: Adapt to different use cases

By following the patterns and best practices outlined in this guide, you can build robust, scalable APIs that handle blocking operations efficiently while maintaining system stability and performance. 