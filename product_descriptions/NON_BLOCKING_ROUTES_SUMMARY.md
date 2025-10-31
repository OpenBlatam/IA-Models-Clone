# Non-Blocking Routes System

## Overview

This document provides a comprehensive overview of the non-blocking routes system that eliminates blocking operations in FastAPI routes through async-first patterns, connection pooling, background task processing, and circuit breaker patterns.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Non-Blocking Patterns](#non-blocking-patterns)
4. [Connection Pooling](#connection-pooling)
5. [Background Task Processing](#background-task-processing)
6. [Circuit Breaker Pattern](#circuit-breaker-pattern)
7. [Performance Benefits](#performance-benefits)
8. [Best Practices](#best-practices)
9. [Integration Guide](#integration-guide)
10. [Error Handling](#error-handling)

## Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │ Non-Blocking    │    │ Connection      │
│                 │    │ Route Manager   │    │ Pools           │
│  - Async Routes │───▶│  - Decorators   │───▶│  - Database     │
│  - Background   │    │  - Task Manager │    │  - Redis        │
│  - Circuit      │    │  - Circuit      │    │  - HTTP         │
│  - Breakers     │    │  - Breakers     │    │  - Pools        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Relationships

- **NonBlockingRouteManager**: Central coordinator for all non-blocking operations
- **ConnectionPool**: Generic connection pooling for various services
- **DatabaseConnectionPool**: Async database connection management
- **RedisConnectionPool**: Redis connection pooling
- **HTTPConnectionPool**: HTTP client connection pooling
- **BackgroundTaskManager**: Background task processing
- **CircuitBreaker**: Failure protection and recovery

## Core Components

### 1. NonBlockingRouteManager

**Purpose**: Central manager for all non-blocking operations

**Key Features**:
- Connection pool management
- Background task coordination
- Circuit breaker management
- Timeout and retry logic
- Resource cleanup

**Implementation**:
```python
class NonBlockingRouteManager:
    def __init__(self):
        self.db_pool: Optional[DatabaseConnectionPool] = None
        self.redis_pool: Optional[RedisConnectionPool] = None
        self.http_pool: Optional[HTTPConnectionPool] = None
        self.task_manager = BackgroundTaskManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    async def initialize_pools(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        """Initialize connection pools."""
        if database_url:
            self.db_pool = DatabaseConnectionPool(database_url)
            await self.db_pool.initialize()
        
        if redis_url:
            self.redis_pool = RedisConnectionPool(redis_url)
            await self.redis_pool.initialize()
        
        self.http_pool = HTTPConnectionPool()
        await self.http_pool.initialize()
```

### 2. Connection Pooling

**Purpose**: Efficient resource management for external services

**Types of Pools**:
- **DatabaseConnectionPool**: PostgreSQL/MySQL connection pooling
- **RedisConnectionPool**: Redis connection management
- **HTTPConnectionPool**: HTTP client session pooling

**Benefits**:
- Reduced connection overhead
- Better resource utilization
- Connection reuse
- Automatic cleanup

**Implementation**:
```python
class DatabaseConnectionPool(ConnectionPool):
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a database query asynchronously."""
        connection = await self.get_connection()
        try:
            result = await connection.fetch(query, *args)
            return [dict(row) for row in result]
        finally:
            await self.return_connection("database", connection)
    
    async def execute_transaction(self, queries: List[tuple]) -> bool:
        """Execute multiple queries in a transaction."""
        connection = await self.get_connection()
        try:
            async with connection.transaction():
                for query, args in queries:
                    await connection.execute(query, *args)
            return True
        except Exception as e:
            print(f"Transaction failed: {e}")
            return False
        finally:
            await self.return_connection("database", connection)
```

### 3. Background Task Manager

**Purpose**: Process heavy operations without blocking the main request

**Key Features**:
- Thread pool execution
- Process pool execution
- Task tracking and management
- Timeout handling
- Resource cleanup

**Implementation**:
```python
class BackgroundTaskManager:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def run_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a function in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    async def run_in_process(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Run a function in a process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
```

### 4. Circuit Breaker Pattern

**Purpose**: Prevent cascading failures and provide graceful degradation

**States**:
- **CLOSED**: Normal operation, requests are allowed
- **OPEN**: Circuit is open, requests are blocked
- **HALF_OPEN**: Limited requests allowed for testing

**Implementation**:
```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise BlockingOperationError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
```

## Non-Blocking Patterns

### 1. Route Decorators

**Purpose**: Ensure routes are non-blocking with automatic timeout and error handling

**Available Decorators**:
- `@non_blocking_route()`: General non-blocking route protection
- `@async_database_operation()`: Database operation protection
- `@async_external_api()`: External API call protection
- `@background_task()`: Background task processing

**Implementation**:
```python
def non_blocking_route(timeout: float = 30.0):
    """Decorator to ensure routes are non-blocking."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await route_manager.execute_with_timeout(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            except BlockingOperationError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return wrapper
    return decorator

def async_database_operation(timeout: float = 10.0):
    """Decorator for async database operations."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not route_manager.db_pool:
                raise HTTPException(status_code=503, detail="Database pool not initialized")
            
            try:
                return await route_manager.execute_with_timeout(
                    func(*args, **kwargs),
                    timeout=timeout,
                    operation_type=OperationType.DATABASE
                )
            except BlockingOperationError as e:
                raise HTTPException(status_code=503, detail=str(e))
        return wrapper
    return decorator
```

### 2. Service Layer Patterns

**Purpose**: Implement non-blocking operations in service classes

**Database Operations**:
```python
class ProductService:
    @async_database_operation()
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get product from database asynchronously."""
        query = "SELECT * FROM products WHERE id = $1"
        result = await route_manager.db_pool.execute_query(query, product_id)
        return result[0] if result else None
    
    @async_database_operation()
    async def get_products_batch(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple products asynchronously."""
        placeholders = ",".join(f"${i+1}" for i in range(len(product_ids)))
        query = f"SELECT * FROM products WHERE id IN ({placeholders})"
        return await route_manager.db_pool.execute_query(query, *product_ids)
```

**External API Operations**:
```python
class ProductService:
    @async_external_api()
    async def get_product_reviews(self, product_id: str) -> Dict[str, Any]:
        """Get product reviews from external API."""
        url = f"https://api.reviews.com/products/{product_id}/reviews"
        return await route_manager.http_pool.get(url)
```

**Background Tasks**:
```python
class ProductService:
    @background_task()
    async def update_product_analytics(self, product_id: str) -> Dict[str, Any]:
        """Update product analytics in background."""
        # Simulate heavy computation
        await asyncio.sleep(2.0)
        return {
            "product_id": product_id,
            "analytics_updated": True,
            "timestamp": time.time()
        }
```

### 3. Concurrent Operations

**Purpose**: Execute multiple operations concurrently for better performance

**Implementation**:
```python
@app.post("/products/batch")
@non_blocking_route()
async def get_products_batch(product_ids: List[str]):
    """Get multiple products with concurrent operations."""
    # Get products from database
    products = await product_service.get_products_batch(product_ids)
    
    # Get reviews for all products concurrently
    review_tasks = [
        product_service.get_product_reviews(product["id"])
        for product in products
    ]
    
    reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
    
    # Combine results
    for product, review in zip(products, reviews):
        if isinstance(review, dict):
            product["reviews"] = review.get("reviews", [])
        else:
            product["reviews"] = []
    
    return {"products": products, "count": len(products)}
```

## Connection Pooling

### 1. Database Connection Pool

**Features**:
- Connection reuse
- Automatic cleanup
- Transaction support
- Query execution
- Connection limits

**Configuration**:
```python
db_pool = DatabaseConnectionPool(
    database_url="postgresql://user:password@localhost/db",
    pool_size=10,
    max_overflow=20,
    timeout=30.0,
    retry_attempts=3
)
```

### 2. Redis Connection Pool

**Features**:
- Connection pooling
- Automatic reconnection
- Command execution
- Key management

**Usage**:
```python
redis_pool = RedisConnectionPool(
    redis_url="redis://localhost:6379",
    pool_size=10,
    max_overflow=20,
    timeout=30.0
)

# Get value
value = await redis_pool.get("key")

# Set value
success = await redis_pool.set("key", "value", expire=3600)
```

### 3. HTTP Connection Pool

**Features**:
- Session reuse
- Connection limits
- Timeout management
- Retry logic

**Usage**:
```python
http_pool = HTTPConnectionPool(
    pool_size=10,
    max_overflow=20,
    timeout=30.0
)

# GET request
response = await http_pool.get("https://api.example.com/data")

# POST request
response = await http_pool.post(
    "https://api.example.com/data",
    {"key": "value"}
)
```

## Background Task Processing

### 1. Task Management

**Features**:
- Task submission
- Status tracking
- Result retrieval
- Error handling
- Resource cleanup

**Implementation**:
```python
# Submit background task
task_id = await route_manager.execute_in_background(
    heavy_computation_function,
    data_parameter
)

# Check task status
try:
    result = await route_manager.task_manager.wait_for_task(task_id, timeout=10.0)
    print(f"Task completed: {result}")
except asyncio.TimeoutError:
    print("Task still running")
```

### 2. Thread and Process Pools

**Thread Pool**: For I/O-bound operations
```python
async def io_intensive_task(file_path: str) -> str:
    """Run I/O-intensive task in thread pool."""
    def read_file(path: str) -> str:
        with open(path, 'r') as f:
            return f.read()
    
    return await route_manager.task_manager.run_in_thread(read_file, file_path)
```

**Process Pool**: For CPU-intensive operations
```python
async def cpu_intensive_task(data: List[int]) -> List[int]:
    """Run CPU-intensive task in process pool."""
    def process_data(numbers: List[int]) -> List[int]:
        return [n * n for n in numbers]
    
    return await route_manager.task_manager.run_in_process(process_data, data)
```

## Circuit Breaker Pattern

### 1. Configuration

**Parameters**:
- `failure_threshold`: Number of failures before opening circuit
- `recovery_timeout`: Time to wait before attempting recovery
- `expected_exception`: Exception types to count as failures

**Implementation**:
```python
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception
)
```

### 2. Usage

**Service Integration**:
```python
async def call_external_service():
    """Call external service with circuit breaker protection."""
    circuit_breaker = route_manager.get_circuit_breaker("external_api")
    
    try:
        result = await circuit_breaker.call(external_api_function)
        return result
    except Exception as e:
        # Handle circuit breaker failures
        return {"fallback": "data"}
```

## Performance Benefits

### 1. Response Time Improvement

**Before (Blocking)**:
```python
def get_product_blocking(product_id: str):
    time.sleep(0.1)  # Blocking database call
    return product_data
```

**After (Non-blocking)**:
```python
@async_database_operation()
async def get_product_async(product_id: str):
    await asyncio.sleep(0.1)  # Non-blocking database call
    return product_data
```

**Performance Comparison**:
- **Blocking**: 10 requests × 0.1s = 1.0s total
- **Non-blocking**: 10 requests × 0.1s = 0.1s total (concurrent)
- **Improvement**: 90% faster response time

### 2. Throughput Improvement

**Concurrent Operations**:
```python
# Sequential (blocking)
for product_id in product_ids:
    product = get_product_blocking(product_id)
    reviews = get_reviews_blocking(product_id)

# Concurrent (non-blocking)
product_tasks = [get_product_async(pid) for pid in product_ids]
review_tasks = [get_reviews_async(pid) for pid in product_ids]

products, reviews = await asyncio.gather(
    asyncio.gather(*product_tasks),
    asyncio.gather(*review_tasks)
)
```

### 3. Resource Utilization

**Connection Pooling Benefits**:
- Reduced connection overhead
- Better resource utilization
- Connection reuse
- Automatic cleanup

**Background Processing Benefits**:
- Non-blocking heavy operations
- Better user experience
- Resource isolation
- Scalability improvement

## Best Practices

### 1. Route Design

**Do's**:
- Use async/await for all I/O operations
- Implement connection pooling
- Use background tasks for heavy operations
- Implement circuit breakers for external services
- Set appropriate timeouts

**Don'ts**:
- Use blocking operations in routes
- Create new connections for each request
- Block the event loop with heavy computation
- Ignore error handling and recovery

### 2. Service Layer Design

**Async-First Approach**:
```python
class ProductService:
    def __init__(self):
        self.db_pool = route_manager.db_pool
        self.cache_pool = route_manager.redis_pool
    
    @async_database_operation()
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        # Check cache first
        cached = await self.cache_pool.get(f"product:{product_id}")
        if cached:
            return cached
        
        # Get from database
        product = await self.db_pool.execute_query(
            "SELECT * FROM products WHERE id = $1",
            product_id
        )
        
        # Cache result
        await self.cache_pool.set(f"product:{product_id}", product)
        return product
```

### 3. Error Handling

**Graceful Degradation**:
```python
@app.get("/products/{product_id}")
@non_blocking_route()
async def get_product(product_id: str):
    try:
        # Try primary data source
        product = await product_service.get_product(product_id)
        return {"product": product, "source": "primary"}
    except Exception as e:
        # Fallback to cache
        cached_product = await cache_service.get_cached_data(f"product:{product_id}")
        if cached_product:
            return {"product": cached_product, "source": "cache"}
        
        # Return error
        raise HTTPException(status_code=404, detail="Product not found")
```

### 4. Monitoring and Observability

**Performance Metrics**:
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pools": {
            "database": route_manager.db_pool is not None,
            "redis": route_manager.redis_pool is not None,
            "http": route_manager.http_pool is not None
        },
        "active_tasks": len(route_manager.task_manager.tasks),
        "circuit_breakers": {
            name: breaker.state
            for name, breaker in route_manager.circuit_breakers.items()
        }
    }
```

## Integration Guide

### 1. FastAPI Integration

**Basic Setup**:
```python
from fastapi import FastAPI
from non_blocking_routes import NonBlockingRouteManager

app = FastAPI()

# Initialize route manager
route_manager = NonBlockingRouteManager()

@app.on_event("startup")
async def startup_event():
    await route_manager.initialize_pools(
        database_url="postgresql://user:password@localhost/db",
        redis_url="redis://localhost:6379"
    )

@app.on_event("shutdown")
async def shutdown_event():
    await route_manager.shutdown()
```

### 2. Service Integration

**Service Classes**:
```python
from non_blocking_routes import (
    async_database_operation, async_external_api, background_task
)

class ProductService:
    @async_database_operation()
    async def get_product(self, product_id: str):
        # Implementation
        pass
    
    @async_external_api()
    async def get_reviews(self, product_id: str):
        # Implementation
        pass
    
    @background_task()
    async def update_analytics(self, product_id: str):
        # Implementation
        pass
```

### 3. Route Integration

**Route Decorators**:
```python
from non_blocking_routes import non_blocking_route

@app.get("/products/{product_id}")
@non_blocking_route()
async def get_product(product_id: str):
    product = await product_service.get_product(product_id)
    return {"product": product}
```

## Error Handling

### 1. Exception Types

**BlockingOperationError**: Raised when blocking operations are detected
**TimeoutError**: Raised when operations exceed timeout
**CircuitBreakerError**: Raised when circuit breaker is open

### 2. Error Handlers

**Global Error Handling**:
```python
@app.exception_handler(BlockingOperationError)
async def blocking_operation_handler(request: Request, exc: BlockingOperationError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service temporarily unavailable",
            "message": str(exc),
            "type": "blocking_operation_error"
        }
    )

@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(request: Request, exc: asyncio.TimeoutError):
    return JSONResponse(
        status_code=408,
        content={
            "error": "Request timeout",
            "message": "Operation timed out",
            "type": "timeout_error"
        }
    )
```

### 3. Recovery Strategies

**Circuit Breaker Recovery**:
```python
async def call_with_fallback(service_func, fallback_func):
    """Call service with fallback strategy."""
    try:
        return await service_func()
    except Exception as e:
        # Try fallback
        return await fallback_func()
```

## Benefits

### 1. **Performance Improvement**
- Reduced response times through concurrency
- Better resource utilization
- Improved throughput

### 2. **Scalability**
- Better handling of concurrent requests
- Resource pooling and reuse
- Background processing

### 3. **Reliability**
- Circuit breaker protection
- Graceful error handling
- Automatic recovery

### 4. **Maintainability**
- Clean separation of concerns
- Reusable patterns
- Easy testing and debugging

### 5. **User Experience**
- Faster response times
- Non-blocking operations
- Better error handling

## Conclusion

The non-blocking routes system provides:

1. **Comprehensive Non-Blocking Patterns**: Async-first approach with decorators and service patterns
2. **Efficient Resource Management**: Connection pooling for databases, Redis, and HTTP clients
3. **Background Processing**: Heavy operations moved to background tasks
4. **Fault Tolerance**: Circuit breaker patterns for external services
5. **Performance Optimization**: Concurrent operations and resource reuse
6. **Easy Integration**: Simple decorators and service patterns

This system ensures optimal API performance by eliminating blocking operations and providing robust, scalable, and maintainable code patterns. 