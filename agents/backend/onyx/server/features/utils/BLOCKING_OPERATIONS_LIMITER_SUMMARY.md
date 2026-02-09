# 🚀 Blocking Operations Limiter System

## Overview

The Blocking Operations Limiter System is a comprehensive solution designed to identify, monitor, and limit blocking operations in FastAPI routes. The system provides automatic detection of blocking patterns, async conversion, performance monitoring, and optimization recommendations to ensure optimal API performance.

## Architecture

### Core Components

1. **Blocking Operations Limiter** (`blocking_operations_limiter.py`)
   - Circuit breaker patterns for failure handling
   - Rate limiting for operation control
   - Resource pool management
   - Timeout handling strategies
   - Performance metrics collection

2. **Route Async Optimizer** (`route_async_optimizer.py`)
   - Automatic route optimization
   - Async conversion patterns
   - Background task management
   - Performance monitoring integration
   - Optimization level configuration

3. **Route Performance Monitor** (`route_performance_monitor.py`)
   - Real-time blocking operation detection
   - Performance analytics and alerting
   - Historical performance tracking
   - Optimization recommendations
   - Performance regression detection

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Blocking Operations Limiter                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Operations    │  │   Route Async   │  │   Route      │ │
│  │    Limiter      │  │   Optimizer     │  │ Performance  │ │
│  │                 │  │                 │  │   Monitor    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│           │                     │                    │       │
│           ▼                     ▼                    ▼       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Performance Management                     │ │
│  │  • Circuit Breakers                                     │ │
│  │  • Rate Limiters                                        │ │
│  │  • Resource Pools                                       │ │
│  │  • Timeout Strategies                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│           │                     │                    │       │
│           ▼                     ▼                    ▼       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              FastAPI Integration                        │ │
│  │  • Middleware Integration                              │ │
│  │  • Route Optimization                                  │ │
│  │  • Performance Monitoring                              │ │
│  │  • Alert Management                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Blocking Operation Detection

#### Automatic Detection
The system automatically detects common blocking patterns:

```python
BLOCKING_PATTERNS = {
    "time.sleep": BlockingOperationType.SLEEP_OPERATION,
    "requests.get": BlockingOperationType.NETWORK_IO,
    "requests.post": BlockingOperationType.NETWORK_IO,
    "urllib.request": BlockingOperationType.NETWORK_IO,
    "subprocess.run": BlockingOperationType.SUBPROCESS,
    "subprocess.call": BlockingOperationType.SUBPROCESS,
    "os.system": BlockingOperationType.SUBPROCESS,
    "sqlite3.connect": BlockingOperationType.DATABASE_QUERY,
    "open(": BlockingOperationType.FILE_IO,
    "file(": BlockingOperationType.FILE_IO,
    "read(": BlockingOperationType.FILE_IO,
    "write(": BlockingOperationType.FILE_IO,
    "seek(": BlockingOperationType.FILE_IO,
    "aiohttp.ClientSession": BlockingOperationType.EXTERNAL_API,
    "httpx.Client": BlockingOperationType.EXTERNAL_API
}
```

#### Operation Types
- **DATABASE_QUERY**: Database operations that could block
- **FILE_IO**: File system operations
- **NETWORK_IO**: Network requests and responses
- **CPU_INTENSIVE**: Computationally expensive operations
- **MEMORY_OPERATION**: Memory-intensive operations
- **EXTERNAL_API**: Third-party API calls
- **SLEEP_OPERATION**: Deliberate delays
- **SUBPROCESS**: System process execution

### 2. Circuit Breaker Pattern

#### Implementation
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

#### States
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit is open, requests fail fast
- **HALF_OPEN**: Testing if service has recovered

### 3. Rate Limiting

#### Implementation
```python
class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit token"""
        with self.lock:
            now = time.time()
            
            # Remove expired requests
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
```

### 4. Resource Pool Management

#### Thread Pool
```python
class ResourcePool:
    def __init__(self, max_workers: int = 10, pool_type: str = "thread"):
        self.max_workers = max_workers
        self.pool_type = pool_type
        
        if pool_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif pool_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        
        self.active_operations = 0
        self.lock = threading.Lock()
    
    async def submit(self, func: Callable, *args, **kwargs):
        """Submit a function to the resource pool"""
        with self.lock:
            if self.active_operations >= self.max_workers:
                raise Exception("Resource pool is full")
            
            self.active_operations += 1
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            return result
        finally:
            with self.lock:
                self.active_operations -= 1
```

### 5. Timeout Strategies

#### Strategy Types
```python
class TimeoutStrategy(Enum):
    CANCEL = "cancel"           # Cancel operation on timeout
    BACKGROUND = "background"   # Move to background on timeout
    CACHE_FALLBACK = "cache_fallback"  # Use cached result on timeout
    ERROR_RESPONSE = "error_response"  # Return error on timeout
```

#### Implementation
```python
async def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs):
    """Execute function with timeout"""
    if asyncio.iscoroutinefunction(func):
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
    else:
        # For sync functions, run in executor
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, func, *args, **kwargs),
            timeout=timeout
        )
```

## Usage Patterns

### 1. Basic Operation Limiting

```python
from agents.backend.onyx.server.features.utils.blocking_operations_limiter import (
    limit_blocking_operations, OperationConfig, OperationType, BlockingLevel
)

# Configure operation limits
config = OperationConfig(
    operation_type=OperationType.DATABASE_QUERY,
    blocking_level=BlockingLevel.HIGH,
    timeout_seconds=10.0,
    timeout_strategy=TimeoutStrategy.BACKGROUND,
    max_retries=3,
    circuit_breaker_enabled=True,
    rate_limit_enabled=True
)

@limit_blocking_operations("database_query", config)
async def database_operation(user_id: str):
    # This operation will be limited and monitored
    return await perform_database_query(user_id)
```

### 2. Route-Level Optimization

```python
from agents.backend.onyx.server.features.utils.route_async_optimizer import (
    optimize_route, RouteOptimizationLevel
)

@app.get("/users/{user_id}")
@optimize_route(RouteOptimizationLevel.ADVANCED)
async def get_user(user_id: str):
    """Route with automatic optimization"""
    return await user_service.get_user(user_id)
```

### 3. Performance Monitoring

```python
from agents.backend.onyx.server.features.utils.route_performance_monitor import (
    setup_route_monitoring
)

# Setup monitoring for the entire app
app = FastAPI()
monitor = setup_route_monitoring(app)

# Routes are automatically monitored
@app.get("/api/data")
async def get_data():
    return {"data": "example"}
```

### 4. Automatic Async Conversion

```python
from agents.backend.onyx.server.features.utils.blocking_operations_limiter import (
    async_operation, timeout_operation
)

@async_operation(OperationType.FILE_IO, timeout_seconds=5.0)
async def file_operation(file_path: str):
    """Automatically converted to async if needed"""
    with open(file_path, 'r') as f:
        return f.read()

@timeout_operation(timeout_seconds=30.0, strategy=TimeoutStrategy.BACKGROUND)
async def long_running_operation():
    """Operation with timeout and background fallback"""
    await asyncio.sleep(25.0)  # This would timeout
    return "Operation completed"
```

### 5. Background Task Management

```python
from agents.backend.onyx.server.features.utils.blocking_operations_limiter import (
    background_task
)

@app.post("/process-data")
@background_task("data_processing")
async def process_data(background_tasks: BackgroundTasks, data: Dict[str, Any]):
    """Route that moves processing to background"""
    async def background_processing():
        # Long-running processing
        await asyncio.sleep(60.0)
        logger.info("Background processing completed")
    
    background_tasks.add_task(background_processing)
    return {"message": "Processing started in background"}
```

## FastAPI Integration

### 1. Middleware Integration

```python
from fastapi import FastAPI
from agents.backend.onyx.server.features.utils.blocking_operations_limiter import (
    get_blocking_limiter
)

app = FastAPI()

@app.middleware("http")
async def blocking_operations_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Get limiter
    limiter = await get_blocking_limiter()
    
    try:
        # Process request with limits
        response = await call_next(request)
        
        # Record metrics
        response_time = time.time() - start_time
        limiter.record_request(
            endpoint=request.url.path,
            method=request.method,
            response_time=response_time,
            status_code=response.status_code
        )
        
        return response
        
    except Exception as e:
        # Record error
        response_time = time.time() - start_time
        limiter.record_request(
            endpoint=request.url.path,
            method=request.method,
            response_time=response_time,
            status_code=500
        )
        raise
```

### 2. Route Optimization

```python
from agents.backend.onyx.server.features.utils.route_async_optimizer import (
    create_optimized_app, RouteOptimizationLevel
)

# Create app with automatic optimization
app = create_optimized_app(
    "My Optimized API",
    RouteOptimizationLevel.ADVANCED
)

# Routes are automatically optimized
@app.get("/api/users")
async def get_users():
    return await user_service.get_all_users()

@app.post("/api/users")
async def create_user(user_data: UserCreate):
    return await user_service.create_user(user_data)
```

### 3. Performance Monitoring

```python
from agents.backend.onyx.server.features.utils.route_performance_monitor import (
    setup_route_monitoring
)

app = FastAPI()

# Setup comprehensive monitoring
monitor = setup_route_monitoring(app)

# Monitoring endpoints are automatically added:
# - GET /monitoring/routes/performance
# - GET /monitoring/alerts
# - POST /monitoring/alerts/{alert_id}/acknowledge
# - POST /monitoring/alerts/{alert_id}/resolve
# - GET /monitoring/summary
```

## Configuration

### 1. Operation Configuration

```python
# Database operations
DB_CONFIG = OperationConfig(
    operation_type=OperationType.DATABASE_QUERY,
    blocking_level=BlockingLevel.HIGH,
    timeout_seconds=10.0,
    timeout_strategy=TimeoutStrategy.BACKGROUND,
    max_retries=3,
    circuit_breaker_enabled=True,
    rate_limit_enabled=True,
    max_concurrent=20
)

# File operations
FILE_CONFIG = OperationConfig(
    operation_type=OperationType.FILE_IO,
    blocking_level=BlockingLevel.MEDIUM,
    timeout_seconds=5.0,
    timeout_strategy=TimeoutStrategy.CANCEL,
    max_retries=2,
    circuit_breaker_enabled=False,
    rate_limit_enabled=False,
    max_concurrent=10
)

# External API calls
API_CONFIG = OperationConfig(
    operation_type=OperationType.EXTERNAL_API,
    blocking_level=BlockingLevel.CRITICAL,
    timeout_seconds=15.0,
    timeout_strategy=TimeoutStrategy.CACHE_FALLBACK,
    max_retries=5,
    circuit_breaker_enabled=True,
    rate_limit_enabled=True,
    max_concurrent=5
)
```

### 2. Performance Thresholds

```python
PERFORMANCE_THRESHOLDS = {
    "response_time_warning": 1.0,    # 1 second
    "response_time_error": 5.0,      # 5 seconds
    "response_time_critical": 10.0,  # 10 seconds
    "blocking_operations_warning": 3,
    "blocking_operations_error": 10,
    "error_rate_warning": 0.05,      # 5%
    "error_rate_error": 0.10,        # 10%
    "error_rate_critical": 0.20      # 20%
}
```

### 3. Optimization Levels

```python
# Basic optimization
BASIC_CONFIG = {
    "enable_auto_optimization": True,
    "enable_performance_monitoring": True,
    "enable_background_tasks": False,
    "enable_circuit_breakers": False,
    "enable_rate_limiting": False
}

# Advanced optimization
ADVANCED_CONFIG = {
    "enable_auto_optimization": True,
    "enable_performance_monitoring": True,
    "enable_background_tasks": True,
    "enable_circuit_breakers": True,
    "enable_rate_limiting": True
}

# Ultra optimization
ULTRA_CONFIG = {
    "enable_auto_optimization": True,
    "enable_performance_monitoring": True,
    "enable_background_tasks": True,
    "enable_circuit_breakers": True,
    "enable_rate_limiting": True,
    "enable_caching": True,
    "enable_load_balancing": True,
    "enable_auto_scaling": True
}
```

## Performance Optimization Strategies

### 1. Database Query Optimization

```python
# Before: Blocking database query
def get_user_data(user_id: str):
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    connection.close()
    return result

# After: Async database query with limits
@limit_blocking_operations("database_query", DB_CONFIG)
async def get_user_data_async(user_id: str):
    async with async_database_connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            result = await cursor.fetchone()
            return result
```

### 2. File I/O Optimization

```python
# Before: Blocking file operation
def read_file_sync(file_path: str):
    with open(file_path, 'r') as f:
        return f.read()

# After: Async file operation with limits
@limit_blocking_operations("file_io", FILE_CONFIG)
async def read_file_async(file_path: str):
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()
```

### 3. External API Call Optimization

```python
# Before: Blocking API call
def call_external_api_sync(url: str):
    response = requests.get(url, timeout=30)
    return response.json()

# After: Async API call with circuit breaker
@limit_blocking_operations("external_api", API_CONFIG)
async def call_external_api_async(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 4. CPU-Intensive Operation Optimization

```python
# Before: Blocking CPU-intensive operation
def compute_heavy_sync(data: List[int]):
    result = 0
    for i in data:
        result += i ** 2
    return result

# After: Async CPU-intensive operation with resource pool
@limit_blocking_operations("computation", COMPUTATION_CONFIG)
async def compute_heavy_async(data: List[int]):
    # Use process pool for CPU-intensive tasks
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, compute_heavy_sync, data)
```

## Monitoring and Alerting

### 1. Performance Metrics

```python
# Get route performance
performance_data = monitor.get_route_performance("/api/users", "GET")
print(f"Average response time: {performance_data['average_response_time']:.2f}s")
print(f"Success rate: {performance_data['success_rate']:.2%}")
print(f"Blocking operations: {performance_data['total_blocking_operations']}")

# Get performance summary
summary = monitor.get_performance_summary()
print(f"Overall success rate: {summary['overall']['overall_success_rate']:.2%}")
print(f"Blocking to async ratio: {summary['overall']['blocking_to_async_ratio']:.2f}")
```

### 2. Performance Alerts

```python
# Get active alerts
alerts = monitor.get_performance_alerts(PerformanceAlertLevel.WARNING)
for alert in alerts:
    print(f"Alert: {alert['message']}")
    print(f"Route: {alert['route_path']}")
    print(f"Level: {alert['level']}")

# Acknowledge alerts
monitor.acknowledge_alert(alert_id)

# Resolve alerts
monitor.resolve_alert(alert_id)
```

### 3. Optimization Recommendations

```python
# Get optimization recommendations
recommendations = monitor.get_performance_summary()["optimization_recommendations"]
for recommendation in recommendations:
    print(f"Recommendation: {recommendation}")
```

## Best Practices

### 1. Route Design

- **Use async functions** for all route handlers
- **Avoid blocking operations** in route handlers
- **Move long operations** to background tasks
- **Implement proper error handling** for all operations
- **Use appropriate timeouts** for external calls

### 2. Database Operations

- **Use async database drivers** (asyncpg, aiosqlite)
- **Implement connection pooling** for database connections
- **Use prepared statements** to avoid SQL injection
- **Implement proper transaction handling**
- **Use database indexes** for better performance

### 3. File Operations

- **Use async file libraries** (aiofiles)
- **Implement proper file locking** for concurrent access
- **Use streaming** for large files
- **Implement proper error handling** for file operations
- **Use appropriate file permissions**

### 4. External API Calls

- **Use async HTTP clients** (aiohttp, httpx)
- **Implement circuit breakers** for external services
- **Use connection pooling** for HTTP clients
- **Implement proper retry logic** with exponential backoff
- **Cache responses** when appropriate

### 5. Performance Monitoring

- **Monitor response times** for all routes
- **Track blocking operations** and convert to async
- **Set up alerts** for performance issues
- **Monitor resource usage** (CPU, memory, disk I/O)
- **Track error rates** and investigate issues

## Migration Guide

### 1. From Sync to Async Routes

```python
# Before: Sync route
@app.get("/users/{user_id}")
def get_user(user_id: str):
    user = database.get_user(user_id)  # Blocking
    return user

# After: Async route with limits
@app.get("/users/{user_id}")
@limit_blocking_operations("get_user", DB_CONFIG)
async def get_user(user_id: str):
    user = await database.get_user_async(user_id)  # Async
    return user
```

### 2. From Blocking to Non-blocking Operations

```python
# Before: Blocking file read
def read_config():
    with open('config.json', 'r') as f:
        return json.load(f)

# After: Async file read with limits
@limit_blocking_operations("read_config", FILE_CONFIG)
async def read_config():
    async with aiofiles.open('config.json', 'r') as f:
        content = await f.read()
        return json.loads(content)
```

### 3. From Direct API Calls to Optimized Calls

```python
# Before: Direct API call
def fetch_data(url: str):
    response = requests.get(url)
    return response.json()

# After: Optimized API call
@limit_blocking_operations("fetch_data", API_CONFIG)
async def fetch_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## Troubleshooting

### 1. Common Issues

**High Response Times**
- Check for blocking operations in route handlers
- Implement async patterns for database queries
- Use background tasks for long-running operations
- Implement caching for frequently accessed data

**High Error Rates**
- Check circuit breaker states
- Implement proper error handling
- Add retry logic with exponential backoff
- Monitor external service health

**Resource Exhaustion**
- Increase resource pool sizes
- Implement rate limiting
- Use connection pooling
- Monitor resource usage

### 2. Performance Tuning

```python
# Optimize resource pools
RESOURCE_CONFIG = {
    "thread_pool_size": 20,
    "process_pool_size": 4,
    "max_concurrent_operations": 100,
    "connection_pool_size": 20
}

# Optimize timeouts
TIMEOUT_CONFIG = {
    "database_timeout": 10.0,
    "file_timeout": 5.0,
    "api_timeout": 15.0,
    "computation_timeout": 30.0
}

# Optimize rate limits
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 1000,
    "requests_per_hour": 10000,
    "burst_limit": 100
}
```

### 3. Monitoring and Debugging

```python
# Enable debug logging
import logging
logging.getLogger("blocking_operations").setLevel(logging.DEBUG)

# Get detailed metrics
limiter = await get_blocking_limiter()
detailed_metrics = limiter.get_operation_metrics("database_query")
print(f"Database operation metrics: {detailed_metrics}")

# Check circuit breaker states
circuit_breakers = limiter.circuit_breakers
for name, cb in circuit_breakers.items():
    print(f"Circuit breaker {name}: {cb.state}")
```

## Conclusion

The Blocking Operations Limiter System provides a comprehensive solution for identifying, monitoring, and limiting blocking operations in FastAPI routes. By implementing async patterns, circuit breakers, rate limiting, and performance monitoring, the system ensures optimal API performance and reliability.

Key benefits include:
- **Automatic detection** of blocking operations
- **Async conversion** of sync operations
- **Performance monitoring** and alerting
- **Circuit breaker patterns** for failure handling
- **Rate limiting** for operation control
- **Resource pool management** for efficient resource usage
- **Background task processing** for long-running operations
- **Comprehensive optimization recommendations**

The system is designed to be production-ready with minimal overhead and provides comprehensive monitoring, alerting, and optimization capabilities for modern API development. 