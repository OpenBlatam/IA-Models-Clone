# Limiting Blocking Operations in FastAPI Routes

## Overview

Blocking operations in FastAPI routes can severely impact performance and user experience. This guide covers best practices for identifying, avoiding, and properly handling blocking operations in production systems.

## Key Principles

### 1. **Async-First Approach**
- Use `async def` for all route handlers
- Avoid synchronous I/O operations in routes
- Leverage FastAPI's async capabilities

### 2. **Background Task Processing**
- Move heavy operations to background tasks
- Use `BackgroundTasks` for non-critical operations
- Implement task queues for complex processing

### 3. **Thread Pool Management**
- Use `ThreadPoolExecutor` for CPU-bound operations
- Limit thread pool size to prevent resource exhaustion
- Implement proper error handling for threaded operations

### 4. **Database and External Service Optimization**
- Use async database drivers
- Implement connection pooling
- Use async HTTP clients for external APIs

## Common Blocking Operations to Avoid

### ❌ **Synchronous Database Operations**
```python
# BAD: Blocking database operation
@app.post("/posts")
def create_post(post_data: dict):
    # This blocks the event loop
    result = db.execute("INSERT INTO posts VALUES (...)")
    return result
```

### ❌ **Synchronous File Operations**
```python
# BAD: Blocking file operation
@app.post("/upload")
def upload_file(file: UploadFile):
    # This blocks the event loop
    with open("file.txt", "w") as f:
        f.write(file.file.read())
    return {"status": "uploaded"}
```

### ❌ **CPU-Intensive Operations**
```python
# BAD: CPU-intensive operation in route
@app.post("/analyze")
def analyze_text(text: str):
    # This blocks the event loop
    result = heavy_nlp_processing(text)
    return result
```

### ❌ **Synchronous External API Calls**
```python
# BAD: Blocking HTTP request
@app.get("/external-data")
def get_external_data():
    # This blocks the event loop
    response = requests.get("https://api.example.com/data")
    return response.json()
```

## Best Practices Implementation

### ✅ **Async Route Handlers**
```python
@app.post("/posts")
async def create_post(post_data: PostRequest):
    # All operations are async
    post_id = await db.create_post(post_data)
    await cache.set(f"post:{post_id}", post_data)
    return {"id": post_id}
```

### ✅ **Background Tasks**
```python
@app.post("/posts")
async def create_post(
    post_data: PostRequest,
    background_tasks: BackgroundTasks
):
    # Immediate response
    post_id = await db.create_post(post_data)
    
    # Heavy operations in background
    background_tasks.add_task(process_analytics, post_id)
    background_tasks.add_task(send_notifications, post_id)
    background_tasks.add_task(generate_image, post_data.content)
    
    return {"id": post_id, "status": "processing"}
```

### ✅ **Thread Pool for CPU-Bound Operations**
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Global thread pool
thread_pool = ThreadPoolExecutor(max_workers=4)

@app.post("/analyze")
async def analyze_text(text: str):
    # CPU-intensive operation in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        thread_pool, 
        heavy_nlp_processing, 
        text
    )
    return result
```

### ✅ **Async External API Calls**
```python
import httpx

@app.get("/external-data")
async def get_external_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

## Performance Optimization Patterns

### 1. **Connection Pooling**
```python
# Database connection pool
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# HTTP client with connection pooling
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    )
)
```

### 2. **Caching Strategy**
```python
from aiocache import cached

@app.get("/posts/{post_id}")
@cached(ttl=300)  # Cache for 5 minutes
async def get_post(post_id: str):
    return await db.get_post(post_id)
```

### 3. **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/posts")
@limiter.limit("10/minute")
async def create_post(request: Request, post_data: PostRequest):
    return await process_post(post_data)
```

### 4. **Circuit Breaker Pattern**
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def external_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()
```

## Monitoring and Observability

### 1. **Performance Metrics**
```python
from prometheus_client import Histogram, Counter

REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
BLOCKING_OPERATIONS = Counter('blocking_operations_total', 'Blocking operations detected')

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Check for blocking operations
    if not asyncio.iscoroutinefunction(call_next):
        BLOCKING_OPERATIONS.inc()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    
    return response
```

### 2. **Async Profiling**
```python
import asyncio
import time

async def profile_operation(operation_name: str, coro):
    start_time = time.time()
    try:
        result = await coro
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.3f}s")
        return result
    except Exception as e:
        logger.error(f"{operation_name} failed: {e}")
        raise
```

## Error Handling for Async Operations

### 1. **Timeout Management**
```python
import asyncio

@app.post("/posts")
async def create_post(post_data: PostRequest):
    try:
        # Set timeout for database operation
        result = await asyncio.wait_for(
            db.create_post(post_data),
            timeout=5.0
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Operation timeout")
```

### 2. **Graceful Degradation**
```python
@app.get("/posts/{post_id}")
async def get_post(post_id: str):
    try:
        # Try cache first
        post = await cache.get(f"post:{post_id}")
        if post:
            return post
        
        # Fallback to database
        post = await db.get_post(post_id)
        if post:
            await cache.set(f"post:{post_id}", post, ttl=300)
            return post
        
        raise HTTPException(status_code=404, detail="Post not found")
        
    except Exception as e:
        logger.error(f"Error fetching post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Testing Async Routes

### 1. **Async Test Patterns**
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_post():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/posts", json={
            "content": "Test post",
            "post_type": "educational"
        })
        assert response.status_code == 200
        assert "id" in response.json()
```

### 2. **Performance Testing**
```python
import asyncio
import time

async def benchmark_route():
    start_time = time.time()
    
    # Simulate concurrent requests
    tasks = []
    for i in range(100):
        task = asyncio.create_task(client.post("/posts", json=post_data))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    print(f"Processed 100 requests in {duration:.3f}s")
    return results
```

## Production Deployment Considerations

### 1. **Worker Configuration**
```python
# uvicorn configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for CPU-bound tasks
        loop="uvloop",  # Faster event loop
        http="httptools",  # Faster HTTP parser
        access_log=False,  # Disable access logs for performance
    )
```

### 2. **Resource Limits**
```python
# Docker resource limits
# docker-compose.yml
services:
  api:
    image: linkedin-posts-api
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### 3. **Health Checks**
```python
@app.get("/health")
async def health_check():
    try:
        # Check database connectivity
        await db.ping()
        
        # Check cache connectivity
        await cache.ping()
        
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
```

## Summary

1. **Always use `async def` for route handlers**
2. **Move heavy operations to background tasks**
3. **Use thread pools for CPU-bound operations**
4. **Implement proper connection pooling**
5. **Add comprehensive monitoring and metrics**
6. **Handle timeouts and errors gracefully**
7. **Test async operations thoroughly**
8. **Configure production resources appropriately**

By following these patterns, you can build high-performance, non-blocking FastAPI applications that scale efficiently and provide excellent user experience. 