# API Performance Metrics System

## Overview

This document provides a comprehensive overview of the API performance metrics system that prioritizes response time, latency, and throughput monitoring. The system provides real-time performance tracking, analytics, alerting, and optimization recommendations.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Metrics](#core-metrics)
3. [Performance Tracking](#performance-tracking)
4. [Real-time Monitoring](#real-time-monitoring)
5. [Alerting System](#alerting-system)
6. [Analytics & Insights](#analytics--insights)
7. [Integration Guide](#integration-guide)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)

## Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │ Performance     │    │ Analytics &     │
│                 │    │ Metrics         │    │ Alerting        │
│  - Middleware   │───▶│  - Tracking     │───▶│  - Real-time    │
│  - Decorators   │    │  - Monitoring   │    │  - Alerts       │
│  - Services     │    │  - Storage      │    │  - Insights     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Relationships

- **APIPerformanceMetrics**: Core metrics collection and storage
- **PerformanceMonitor**: Analytics and insights generation
- **PerformanceThreshold**: Alert configuration and management
- **PerformanceAlert**: Alert generation and tracking
- **Middleware**: Request/response performance tracking
- **Decorators**: Function-level performance tracking

## Core Metrics

### 1. Response Time

**Definition**: Time taken to process and respond to a request

**Key Indicators**:
- **Mean Response Time**: Average response time across all requests
- **P95 Response Time**: 95th percentile response time
- **P99 Response Time**: 99th percentile response time
- **Min/Max Response Time**: Best and worst case performance

**Implementation**:
```python
def get_response_time_stats(self, endpoint: str = "*") -> MetricSummary:
    """Get response time statistics for an endpoint."""
    if endpoint == "*":
        metrics = list(self.metrics["response_time"])
    else:
        metrics = list(self.endpoint_metrics.get(endpoint, []))
    
    if not metrics:
        return MetricSummary()
    
    values = [m.value for m in metrics]
    values.sort()
    
    return MetricSummary(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=values[int(len(values) * 0.95)] if len(values) > 0 else 0.0,
        p99=values[int(len(values) * 0.99)] if len(values) > 0 else 0.0,
        min=min(values),
        max=max(values),
        sum=sum(values),
        std_dev=statistics.stdev(values) if len(values) > 1 else 0.0
    )
```

### 2. Latency

**Definition**: Time delay between request and response

**Key Indicators**:
- **Network Latency**: Time for data to travel over network
- **Processing Latency**: Time for server to process request
- **Database Latency**: Time for database operations
- **External API Latency**: Time for external service calls

**Implementation**:
```python
@track_database_performance
async def get_products_batch(self, product_ids: List[str]) -> List[Dict[str, Any]]:
    """Get multiple products with database performance tracking."""
    # Simulate database batch query
    await asyncio.sleep(0.2)
    
    products = []
    for product_id in product_ids:
        products.append({"id": product_id, "name": f"Product {product_id}"})
    
    return products

@track_external_api_performance
async def get_product_reviews(self, product_id: str) -> List[Dict[str, Any]]:
    """Get product reviews with external API performance tracking."""
    # Simulate external API call
    await asyncio.sleep(0.15)
    
    return [
        {"id": 1, "rating": 5, "comment": "Great product!"},
        {"id": 2, "rating": 4, "comment": "Good quality"}
    ]
```

### 3. Throughput

**Definition**: Number of requests processed per unit time

**Key Indicators**:
- **Requests Per Second (RPS)**: Current throughput
- **Peak Throughput**: Maximum observed throughput
- **Throughput Trend**: Historical throughput patterns
- **Concurrent Requests**: Number of simultaneous requests

**Implementation**:
```python
def get_throughput(self) -> float:
    """Get current throughput (requests per second)."""
    if not self.throughput_history:
        return 0.0
    return statistics.mean(self.throughput_history)

def _update_throughput(self):
    """Update throughput calculation."""
    current_time = time.time()
    time_diff = current_time - self.last_throughput_calc
    
    if time_diff >= 1.0:  # Calculate every second
        throughput = self.request_count / (current_time - self.start_time)
        self.throughput_history.append(throughput)
        self.last_throughput_calc = current_time
```

## Performance Tracking

### 1. Request-Level Tracking

**Middleware Integration**:
```python
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Track performance for all requests."""
    async with performance_tracking(request, Response()):
        response = await call_next(request)
        return response

@asynccontextmanager
async def performance_tracking(request: Request, response: Response):
    """Context manager for tracking request performance."""
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    try:
        yield
    finally:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record metrics
        metrics = get_performance_metrics()
        metrics.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time=response_time,
            request_id=request_id,
            user_id=getattr(request.state, "user_id", None),
            metadata={
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
        )
        metrics.record_request_end()
```

### 2. Function-Level Tracking

**Decorator Integration**:
```python
@track_performance("get_product")
async def get_product(self, product_id: str) -> Dict[str, Any]:
    """Get product with performance tracking."""
    # Simulate database query
    await asyncio.sleep(0.1)
    
    # Simulate cache hit/miss
    if product_id in self.cache:
        performance_metrics.record_cache_hit()
    else:
        performance_metrics.record_cache_miss()
        self.cache[product_id] = {"id": product_id, "name": f"Product {product_id}"}
    
    return self.cache[product_id]

def track_performance(endpoint_name: Optional[str] = None):
    """Decorator to track function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                metrics = get_performance_metrics()
                metrics.record_request(
                    endpoint=endpoint_name or func.__name__,
                    method="FUNCTION",
                    status_code=200,
                    response_time=response_time,
                    request_id=str(uuid.uuid4())
                )
        
        return wrapper
    return decorator
```

### 3. Component-Level Tracking

**Cache Performance**:
```python
def record_cache_hit(self):
    """Record a cache hit."""
    self.cache_hits += 1

def record_cache_miss(self):
    """Record a cache miss."""
    self.cache_misses += 1

def get_cache_hit_rate(self) -> float:
    """Get cache hit rate."""
    total_requests = self.cache_hits + self.cache_misses
    if total_requests == 0:
        return 0.0
    return self.cache_hits / total_requests
```

**Database Performance**:
```python
def record_database_query(self, query_time: float):
    """Record a database query."""
    self.db_query_count += 1
    self.db_query_time += query_time

def get_database_stats(self) -> Dict[str, Any]:
    """Get database performance statistics."""
    return {
        "query_count": self.db_query_count,
        "total_time": self.db_query_time,
        "avg_query_time": self.db_query_time / max(1, self.db_query_count)
    }
```

## Real-time Monitoring

### 1. System Metrics

**Memory Usage**:
```python
async def _collect_system_metrics(self):
    """Collect system metrics periodically."""
    while True:
        try:
            # Memory usage
            memory_percent = psutil.virtual_memory().percent / 100.0
            self.memory_usage.append(memory_percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            self.cpu_usage.append(cpu_percent)
            
            # Check thresholds
            if self.enable_alerts:
                await self._check_thresholds()
            
            await asyncio.sleep(10)  # Collect every 10 seconds
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            await asyncio.sleep(10)
```

**Concurrent Requests**:
```python
def get_concurrent_requests(self) -> Dict[str, int]:
    """Get concurrent request statistics."""
    return {
        "current": self.concurrent_requests,
        "max": self.max_concurrent_requests
    }
```

### 2. Performance Trends

**Throughput Trends**:
```python
def get_throughput_trend(self, window_minutes: int = 5) -> List[float]:
    """Get throughput trend over time window."""
    # Simplified implementation
    return list(self.metrics.throughput_history)
```

**Response Time Trends**:
```python
def get_response_time_percentiles(self, endpoint: str = "*") -> Dict[str, float]:
    """Get response time percentiles."""
    stats = self.metrics.get_response_time_stats(endpoint)
    return {
        "p50": stats.median,
        "p95": stats.p95,
        "p99": stats.p99,
        "p99.9": stats.p99  # Simplified
    }
```

## Alerting System

### 1. Threshold Configuration

**Performance Thresholds**:
```python
class PerformanceThreshold(BaseModel):
    """Performance threshold configuration."""
    metric_type: MetricType
    endpoint: str
    threshold_value: float
    comparison: str = "gt"  # gt, lt, gte, lte, eq
    alert_message: str = ""
    severity: str = "warning"  # warning, error, critical

# Default thresholds
default_thresholds = [
    PerformanceThreshold(
        metric_type=MetricType.RESPONSE_TIME,
        endpoint="*",
        threshold_value=1000.0,  # 1 second
        comparison="gt",
        alert_message="Response time exceeded 1 second",
        severity="warning"
    ),
    PerformanceThreshold(
        metric_type=MetricType.RESPONSE_TIME,
        endpoint="*",
        threshold_value=5000.0,  # 5 seconds
        comparison="gt",
        alert_message="Response time exceeded 5 seconds",
        severity="error"
    ),
    PerformanceThreshold(
        metric_type=MetricType.ERROR_RATE,
        endpoint="*",
        threshold_value=0.05,  # 5% error rate
        comparison="gt",
        alert_message="Error rate exceeded 5%",
        severity="warning"
    )
]
```

### 2. Alert Generation

**Threshold Checking**:
```python
async def _check_thresholds(self):
    """Check performance thresholds and generate alerts."""
    current_time = time.time()
    
    for threshold in self.thresholds:
        current_value = await self._get_current_value(threshold.metric_type, threshold.endpoint)
        
        if current_value is None:
            continue
        
        # Check threshold
        should_alert = False
        if threshold.comparison == "gt" and current_value > threshold.threshold_value:
            should_alert = True
        elif threshold.comparison == "lt" and current_value < threshold.threshold_value:
            should_alert = True
        elif threshold.comparison == "gte" and current_value >= threshold.threshold_value:
            should_alert = True
        elif threshold.comparison == "lte" and current_value <= threshold.threshold_value:
            should_alert = True
        elif threshold.comparison == "eq" and current_value == threshold.threshold_value:
            should_alert = True
        
        if should_alert:
            alert = PerformanceAlert(
                threshold=threshold,
                current_value=current_value,
                timestamp=current_time,
                message=threshold.alert_message,
                severity=threshold.severity
            )
            self.alerts.append(alert)
```

## Analytics & Insights

### 1. System Health Score

**Health Calculation**:
```python
def get_system_health_score(self) -> float:
    """Calculate system health score (0-100)."""
    score = 100.0
    
    # Response time penalty
    response_time = self.metrics.get_response_time_stats().mean
    if response_time > 1000:  # > 1 second
        score -= 20
    elif response_time > 500:  # > 500ms
        score -= 10
    
    # Error rate penalty
    error_rate = self.metrics.get_error_rate()
    if error_rate > 0.05:  # > 5%
        score -= 30
    elif error_rate > 0.01:  # > 1%
        score -= 15
    
    # Memory usage penalty
    memory_usage = self.metrics.get_memory_usage()
    if memory_usage > 0.9:  # > 90%
        score -= 20
    elif memory_usage > 0.8:  # > 80%
        score -= 10
    
    # CPU usage penalty
    cpu_usage = self.metrics.get_cpu_usage()
    if cpu_usage > 0.9:  # > 90%
        score -= 20
    elif cpu_usage > 0.8:  # > 80%
        score -= 10
    
    return max(0.0, score)
```

### 2. Performance Recommendations

**Optimization Insights**:
```python
def get_performance_recommendations(self) -> List[str]:
    """Get performance improvement recommendations."""
    recommendations = []
    
    # Response time recommendations
    response_time = self.metrics.get_response_time_stats().mean
    if response_time > 1000:
        recommendations.append("Consider implementing caching for slow endpoints")
        recommendations.append("Optimize database queries")
    elif response_time > 500:
        recommendations.append("Monitor response times closely")
    
    # Error rate recommendations
    error_rate = self.metrics.get_error_rate()
    if error_rate > 0.05:
        recommendations.append("Investigate and fix error sources")
        recommendations.append("Implement better error handling")
    elif error_rate > 0.01:
        recommendations.append("Monitor error patterns")
    
    # Memory usage recommendations
    memory_usage = self.metrics.get_memory_usage()
    if memory_usage > 0.9:
        recommendations.append("Consider scaling up memory resources")
        recommendations.append("Implement memory cleanup")
    elif memory_usage > 0.8:
        recommendations.append("Monitor memory usage trends")
    
    # Cache recommendations
    cache_hit_rate = self.metrics.get_cache_hit_rate()
    if cache_hit_rate < 0.5:
        recommendations.append("Consider implementing or improving caching strategy")
    
    return recommendations
```

## Integration Guide

### 1. FastAPI Integration

**Basic Setup**:
```python
from fastapi import FastAPI
from performance_metrics import get_performance_metrics, performance_tracking

app = FastAPI()

# Add performance middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    async with performance_tracking(request, Response()):
        response = await call_next(request)
        return response

# Add metrics endpoints
@app.get("/metrics/performance")
async def get_performance_metrics():
    metrics = get_performance_metrics()
    return metrics.get_comprehensive_stats()
```

### 2. Service Layer Integration

**Performance Tracking**:
```python
from performance_metrics import track_performance, track_database_performance

class ProductService:
    @track_performance("get_product")
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        # Implementation
        pass
    
    @track_database_performance
    async def get_products_batch(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        # Implementation
        pass
```

### 3. Custom Metrics

**Custom Metric Tracking**:
```python
def record_custom_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
    """Record a custom metric."""
    metric = PerformanceMetric(
        name=name,
        value=value,
        timestamp=time.time(),
        endpoint="custom",
        method="CUSTOM",
        status_code=200,
        request_id=str(uuid.uuid4()),
        metadata=metadata or {}
    )
    
    self.metrics[name].append(metric)
```

## Best Practices

### 1. Performance Thresholds

**Reasonable Thresholds**:
```python
# Response time thresholds
response_time_thresholds = [
    PerformanceThreshold(
        metric_type=MetricType.RESPONSE_TIME,
        endpoint="*",
        threshold_value=500.0,   # 500ms - warning
        comparison="gt",
        severity="warning"
    ),
    PerformanceThreshold(
        metric_type=MetricType.RESPONSE_TIME,
        endpoint="*",
        threshold_value=1000.0,  # 1s - error
        comparison="gt",
        severity="error"
    ),
    PerformanceThreshold(
        metric_type=MetricType.RESPONSE_TIME,
        endpoint="*",
        threshold_value=5000.0,  # 5s - critical
        comparison="gt",
        severity="critical"
    )
]

# Error rate thresholds
error_rate_thresholds = [
    PerformanceThreshold(
        metric_type=MetricType.ERROR_RATE,
        endpoint="*",
        threshold_value=0.01,    # 1% - warning
        comparison="gt",
        severity="warning"
    ),
    PerformanceThreshold(
        metric_type=MetricType.ERROR_RATE,
        endpoint="*",
        threshold_value=0.05,    # 5% - error
        comparison="gt",
        severity="error"
    )
]
```

### 2. Monitoring Strategy

**Key Metrics to Monitor**:
1. **Response Time**: P95 and P99 percentiles
2. **Throughput**: Requests per second
3. **Error Rate**: Percentage of failed requests
4. **System Resources**: CPU and memory usage
5. **Cache Performance**: Hit rates and miss rates
6. **Database Performance**: Query times and connection pools
7. **External Dependencies**: API call latencies

### 3. Alert Management

**Alert Best Practices**:
1. **Avoid Alert Fatigue**: Set meaningful thresholds
2. **Escalation Paths**: Define severity levels and escalation procedures
3. **Alert Grouping**: Group related alerts to reduce noise
4. **Alert Documentation**: Document alert meanings and resolution steps
5. **Alert Testing**: Regularly test alert mechanisms

## Performance Optimization

### 1. Response Time Optimization

**Caching Strategy**:
```python
@track_cache_performance
async def get_product(self, product_id: str) -> Dict[str, Any]:
    # Check cache first
    cache_key = f"product:{product_id}"
    cached_product = await cache.get(cache_key)
    
    if cached_product:
        performance_metrics.record_cache_hit()
        return cached_product
    
    # Fetch from database
    product = await database.get_product(product_id)
    
    # Cache the result
    await cache.set(cache_key, product, ttl=3600)
    performance_metrics.record_cache_miss()
    
    return product
```

**Database Optimization**:
```python
@track_database_performance
async def get_products_batch(self, product_ids: List[str]) -> List[Dict[str, Any]]:
    # Use batch queries instead of individual queries
    query = "SELECT * FROM products WHERE id = ANY($1)"
    products = await database.fetch_all(query, product_ids)
    return products
```

### 2. Throughput Optimization

**Concurrency Management**:
```python
async def process_requests_concurrently(self, requests: List[Dict]) -> List[Dict]:
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(100)  # Max 100 concurrent requests
    
    async def process_request(request):
        async with semaphore:
            return await self.process_single_request(request)
    
    # Process requests concurrently
    results = await asyncio.gather(*[
        process_request(req) for req in requests
    ])
    
    return results
```

**Connection Pooling**:
```python
# Configure database connection pool
database_config = {
    "min_size": 10,
    "max_size": 50,
    "max_queries": 50000,
    "max_inactive_connection_lifetime": 300.0
}
```

### 3. System Resource Optimization

**Memory Management**:
```python
# Implement memory cleanup
async def cleanup_old_data(self):
    """Cleanup old data to free memory."""
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
    
    # Cleanup old metrics
    for metric_type in self.metrics:
        while (self.metrics[metric_type] and 
               self.metrics[metric_type][0].timestamp < cutoff_time):
            self.metrics[metric_type].popleft()
```

**CPU Optimization**:
```python
# Use async operations for I/O-bound tasks
async def process_data_async(self, data: List[Any]) -> List[Any]:
    """Process data asynchronously to optimize CPU usage."""
    tasks = [self.process_item_async(item) for item in data]
    return await asyncio.gather(*tasks)
```

## Benefits

### 1. **Real-time Monitoring**
- Immediate visibility into API performance
- Proactive issue detection
- Performance trend analysis

### 2. **Performance Optimization**
- Data-driven optimization decisions
- Bottleneck identification
- Resource utilization insights

### 3. **Alert Management**
- Automated performance alerts
- Configurable thresholds
- Severity-based escalation

### 4. **Analytics & Insights**
- Performance recommendations
- System health scoring
- Optimization guidance

### 5. **Scalability**
- Efficient metrics storage
- Background processing
- Resource cleanup

## Conclusion

The API performance metrics system provides:

1. **Comprehensive Monitoring**: Response time, latency, throughput, and system metrics
2. **Real-time Tracking**: Immediate performance visibility
3. **Intelligent Alerting**: Configurable thresholds with severity levels
4. **Performance Analytics**: Health scoring and optimization recommendations
5. **Easy Integration**: FastAPI middleware and decorators
6. **Scalable Architecture**: Efficient storage and processing

This system ensures optimal API performance through continuous monitoring, proactive alerting, and data-driven optimization strategies. 