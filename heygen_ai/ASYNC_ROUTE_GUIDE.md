# Async Route Management Guide

A comprehensive guide to limiting blocking operations in routes for the HeyGen AI FastAPI application using async patterns, background processing, and non-blocking strategies.

## 🎯 Overview

This guide covers the complete async route management system designed to:
- **Prevent Blocking Operations**: Identify and prevent blocking operations in routes
- **Implement Async Patterns**: Use async/await patterns for non-blocking operations
- **Background Processing**: Move heavy operations to background tasks
- **Thread Pool Management**: Use thread pools for CPU-intensive operations
- **Process Pool Management**: Use process pools for heavy computations
- **Rate Limiting**: Implement rate limiting to prevent overload
- **Circuit Breaker**: Add circuit breaker patterns for fault tolerance
- **Caching**: Implement response caching for performance
- **Monitoring**: Monitor route performance and blocking operations

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Async Route Manager](#async-route-manager)
3. [Async Operation Patterns](#async-operation-patterns)
4. [Background Processing](#background-processing)
5. [Thread and Process Pools](#thread-and-process-pools)
6. [Middleware System](#middleware-system)
7. [Blocking Operation Detection](#blocking-operation-detection)
8. [Best Practices](#best-practices)
9. [Integration Examples](#integration-examples)
10. [Troubleshooting](#troubleshooting)

## 🏗️ System Architecture

### **Async Route Management Architecture**

```
FastAPI Application
├── Async Route Middleware (Request/Response handling)
├── Async Route Manager (Core async management)
├── Thread Pool Manager (CPU-intensive tasks)
├── Process Pool Manager (Heavy computations)
├── Background Task Queue (Non-blocking operations)
├── Async Operation Patterns (Common patterns)
└── Monitoring & Analytics (Performance tracking)
```

### **Core Components**

1. **Async Route Manager**: Main coordinator for async operations
2. **Thread Pool Manager**: Handles CPU-intensive operations
3. **Process Pool Manager**: Manages heavy computations
4. **Background Task Queue**: Processes non-blocking tasks
5. **Async Operation Patterns**: Common async patterns and utilities
6. **Middleware System**: Request/response processing middleware

## 🔧 Async Route Manager

### **1. Basic Setup**

```python
from api.async_operations.async_route_manager import AsyncRouteManager, OperationConfig, OperationType, ExecutionStrategy

# Configure async route manager
config = OperationConfig(
    operation_type=OperationType.IO_INTENSIVE,
    execution_strategy=ExecutionStrategy.ASYNC,
    timeout=30.0,
    max_retries=3,
    enable_caching=True,
    cache_ttl=300,
    max_concurrent=10
)

# Initialize async route manager
route_manager = AsyncRouteManager()
await route_manager.initialize()
```

### **2. FastAPI Integration**

```python
from fastapi import FastAPI
from api.async_operations.async_route_manager import AsyncRouteMiddleware

# Create FastAPI app
app = FastAPI(title="HeyGen AI API")

# Add async route middleware
app.add_middleware(AsyncRouteMiddleware, route_manager=route_manager)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await route_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await route_manager.cleanup()
```

### **3. Route Registration**

```python
# Register route handlers with async optimization
@route_manager.register_route_handler("/api/videos")
async def get_videos():
    return await video_service.get_videos()

@route_manager.register_route_handler("/api/process-video")
async def process_video():
    return await video_processor.process()
```

## 🔄 Async Operation Patterns

### **1. Database Operations**

```python
from api.async_operations.async_patterns import AsyncDatabasePatterns

# Initialize database patterns
db_patterns = AsyncDatabasePatterns(connection_pool)

# Async database query
@app.get("/users")
async def get_users():
    users = await db_patterns.execute_query(
        "SELECT * FROM users WHERE active = true",
        timeout=10.0
    )
    return users

# Batch database operations
@app.post("/users/batch")
async def create_users_batch(users: List[UserCreate]):
    queries = [
        {"query": "INSERT INTO users (name, email) VALUES ($1, $2)", "params": {"name": user.name, "email": user.email}}
        for user in users
    ]
    results = await db_patterns.execute_batch_queries(queries, batch_size=50)
    return results

# Stream database results
@app.get("/users/stream")
async def stream_users():
    return StreamingResponse(
        db_patterns.stream_query_results("SELECT * FROM users"),
        media_type="application/json"
    )
```

### **2. File I/O Operations**

```python
from api.async_operations.async_patterns import AsyncFileIOPatterns

# Initialize file I/O patterns
file_patterns = AsyncFileIOPatterns("./data")

# Async file read
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    content = await file_patterns.read_file_async(file_path)
    return {"content": content}

# Async file write
@app.post("/files/{file_path:path}")
async def write_file(file_path: str, content: str):
    success = await file_patterns.write_file_async(file_path, content)
    return {"success": success}

# Stream file content
@app.get("/files/{file_path:path}/stream")
async def stream_file(file_path: str):
    return StreamingResponse(
        file_patterns.read_file_chunks(file_path),
        media_type="application/octet-stream"
    )
```

### **3. Network Operations**

```python
from api.async_operations.async_patterns import AsyncNetworkPatterns

# Initialize network patterns
network_patterns = AsyncNetworkPatterns(timeout=30.0, max_retries=3)

# Async HTTP request
@app.get("/external-data")
async def get_external_data():
    response = await network_patterns.make_request(
        "https://api.external.com/data",
        method="GET",
        headers={"Authorization": "Bearer token"}
    )
    return response

# Batch network requests
@app.get("/external-data/batch")
async def get_external_data_batch(urls: List[str]):
    requests = [
        {"url": url, "method": "GET"}
        for url in urls
    ]
    results = await network_patterns.make_batch_requests(requests, max_concurrent=10)
    return results

# Stream network response
@app.get("/external-data/stream")
async def stream_external_data():
    return StreamingResponse(
        network_patterns.stream_request("https://api.external.com/stream"),
        media_type="application/octet-stream"
    )
```

### **4. Caching Operations**

```python
from api.async_operations.async_patterns import AsyncCachePatterns

# Initialize cache patterns
cache_patterns = AsyncCachePatterns("redis://localhost:6379")

# Cached operation
@app.get("/cached-data")
async def get_cached_data():
    return await cache_patterns.get_or_set_cached(
        "data_key",
        fetch_func=lambda: fetch_expensive_data(),
        ttl=300
    )

# Cache invalidation
@app.delete("/cached-data")
async def invalidate_cache():
    count = await cache_patterns.invalidate_pattern("data_*")
    return {"invalidated_keys": count}
```

## 🔄 Background Processing

### **1. Background Task Queue**

```python
from api.async_operations.async_route_manager import BackgroundTaskQueue

# Initialize background task queue
background_queue = BackgroundTaskQueue(max_size=1000)
await background_queue.start()

# Register task handlers
background_queue.register_handler("email_send", handle_email_send)
background_queue.register_handler("file_processing", handle_file_processing)
background_queue.register_handler("data_export", handle_data_export)

# Submit background tasks
@app.post("/send-email")
async def send_email(email_data: EmailData):
    task_id = f"email_{int(time.time() * 1000)}"
    await background_queue.submit_task("email_send", task_id, email_data.dict())
    return {"task_id": task_id, "status": "queued"}

# Background task handlers
async def handle_email_send(data: Dict[str, Any]):
    # Process email sending
    await email_service.send_email(data)
    logger.info(f"Email sent to {data.get('to')}")

async def handle_file_processing(data: Dict[str, Any]):
    # Process file
    await file_processor.process(data.get('file_path'))
    logger.info(f"File processed: {data.get('file_path')}")
```

### **2. Background Tasks with FastAPI**

```python
from fastapi import BackgroundTasks

# Background tasks
@app.post("/process-video")
async def process_video(video_data: VideoData, background_tasks: BackgroundTasks):
    # Add background task
    background_tasks.add_task(process_video_background, video_data)
    
    return {"message": "Video processing started"}

async def process_video_background(video_data: VideoData):
    # Heavy video processing
    await video_processor.process(video_data)
    logger.info(f"Video processed: {video_data.filename}")
```

## 🧵 Thread and Process Pools

### **1. Thread Pool Manager**

```python
from api.async_operations.async_route_manager import ThreadPoolManager

# Initialize thread pool
thread_pool = ThreadPoolManager(max_workers=20)

# CPU-intensive operations
@app.get("/heavy-computation")
async def heavy_computation(data: ComputationData):
    result = await thread_pool.submit_task(
        "computation_task",
        cpu_intensive_function,
        data.values
    )
    return {"result": result}

def cpu_intensive_function(values: List[float]) -> float:
    # CPU-intensive computation
    result = sum(x * x for x in values)
    return result

# File operations in thread pool
@app.post("/upload-file")
async def upload_file(file: UploadFile):
    content = await file.read()
    
    result = await thread_pool.submit_task(
        "file_upload",
        process_file_sync,
        content,
        file.filename
    )
    return {"success": result}

def process_file_sync(content: bytes, filename: str) -> bool:
    # Synchronous file processing
    with open(f"uploads/{filename}", "wb") as f:
        f.write(content)
    return True
```

### **2. Process Pool Manager**

```python
from api.async_operations.async_route_manager import ProcessPoolManager

# Initialize process pool
process_pool = ProcessPoolManager(max_workers=4)

# Heavy computations
@app.get("/machine-learning")
async def machine_learning_prediction(data: MLData):
    result = await process_pool.submit_task(
        "ml_prediction",
        ml_prediction_function,
        data.features
    )
    return {"prediction": result}

def ml_prediction_function(features: List[float]) -> float:
    # Heavy ML computation
    # This runs in a separate process
    result = complex_ml_algorithm(features)
    return result
```

## 🛡️ Middleware System

### **1. Async Route Middleware**

```python
from api.async_operations.async_middleware import AsyncRouteMiddleware, MiddlewareConfig

# Configure middleware
config = MiddlewareConfig(
    middleware_type=MiddlewareType.ASYNC_ROUTE,
    blocking_detection=BlockingDetectionLevel.ADVANCED,
    timeout=30.0,
    max_concurrent=100,
    enable_monitoring=True
)

# Add middleware
app.add_middleware(AsyncRouteMiddleware, config=config)
```

### **2. Non-Blocking Middleware**

```python
from api.async_operations.async_middleware import NonBlockingMiddleware

# Add non-blocking middleware
app.add_middleware(NonBlockingMiddleware, config=config)

# This middleware automatically detects and handles blocking operations
```

### **3. Rate Limiting Middleware**

```python
from api.async_operations.async_middleware import RateLimitingMiddleware

# Configure rate limiting
config = MiddlewareConfig(
    rate_limit=100,  # 100 requests per second
    enable_rate_limiting=True
)

# Add rate limiting middleware
app.add_middleware(RateLimitingMiddleware, config=config)
```

### **4. Circuit Breaker Middleware**

```python
from api.async_operations.async_middleware import CircuitBreakerMiddleware

# Add circuit breaker middleware
app.add_middleware(CircuitBreakerMiddleware, config=config)

# This prevents cascading failures
```

### **5. Caching Middleware**

```python
from api.async_operations.async_middleware import CachingMiddleware

# Configure caching
config = MiddlewareConfig(
    enable_caching=True,
    cache_ttl=300
)

# Add caching middleware
app.add_middleware(CachingMiddleware, config=config)
```

### **6. Monitoring Middleware**

```python
from api.async_operations.async_middleware import MonitoringMiddleware

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware, config=config)

# Get monitoring stats
@app.get("/monitoring/stats")
async def get_monitoring_stats():
    return monitoring_middleware.get_monitoring_stats()
```

## 🔍 Blocking Operation Detection

### **1. Automatic Detection**

```python
# The system automatically detects blocking operations
blocking_ops = route_manager.detect_blocking_operations(func)

# Common blocking operations detected:
# - time.sleep()
# - requests.get() / requests.post()
# - subprocess.run()
# - file operations (open, read, write)
# - database operations (synchronous)
# - threading.Thread()
# - multiprocessing.Process()
```

### **2. Manual Detection**

```python
# Check for blocking operations in route
@app.get("/check-blocking")
async def check_blocking_operations():
    route_handlers = route_manager.route_handlers
    
    blocking_report = {}
    for path, handler in route_handlers.items():
        blocking_ops = route_manager.detect_blocking_operations(handler)
        if blocking_ops:
            blocking_report[path] = blocking_ops
    
    return blocking_report
```

### **3. Blocking Operation Prevention**

```python
# Use async decorators to prevent blocking operations
@async_route(
    operation_type=OperationType.IO_INTENSIVE,
    execution_strategy=ExecutionStrategy.ASYNC
)
async def non_blocking_route():
    # This route is guaranteed to be non-blocking
    return await async_operation()

@non_blocking_route(
    operation_type=OperationType.NETWORK_INTENSIVE
)
async def network_route():
    # This route uses async network operations
    return await network_patterns.make_request("https://api.example.com")

@background_task("heavy_processing")
async def background_route():
    # This route moves heavy operations to background
    return {"status": "processing"}
```

## ✅ Best Practices

### **1. Async Route Design**

```python
# ✅ Good: Async route with proper error handling
@app.get("/users")
async def get_users():
    try:
        users = await user_service.get_users_async()
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ❌ Bad: Blocking route
@app.get("/users")
def get_users():
    # This blocks the event loop
    users = user_service.get_users_sync()
    return users
```

### **2. Database Operations**

```python
# ✅ Good: Async database operations
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = await db_patterns.execute_query(
        "SELECT * FROM users WHERE id = $1",
        {"user_id": user_id}
    )
    return user[0] if user else None

# ❌ Bad: Synchronous database operations
@app.get("/users/{user_id}")
def get_user(user_id: int):
    # This blocks the event loop
    user = db.execute_sync("SELECT * FROM users WHERE id = ?", (user_id,))
    return user
```

### **3. File Operations**

```python
# ✅ Good: Async file operations
@app.post("/upload")
async def upload_file(file: UploadFile):
    content = await file.read()
    success = await file_patterns.write_file_async(
        f"uploads/{file.filename}",
        content.decode()
    )
    return {"success": success}

# ❌ Bad: Synchronous file operations
@app.post("/upload")
def upload_file(file: UploadFile):
    # This blocks the event loop
    content = file.file.read()
    with open(f"uploads/{file.filename}", "w") as f:
        f.write(content)
    return {"success": True}
```

### **4. External API Calls**

```python
# ✅ Good: Async external API calls
@app.get("/external-data")
async def get_external_data():
    response = await network_patterns.make_request(
        "https://api.external.com/data",
        method="GET"
    )
    return response

# ❌ Bad: Synchronous external API calls
@app.get("/external-data")
def get_external_data():
    # This blocks the event loop
    response = requests.get("https://api.external.com/data")
    return response.json()
```

### **5. Heavy Computations**

```python
# ✅ Good: Background processing for heavy computations
@app.post("/process-data")
async def process_data(data: ProcessingData, background_tasks: BackgroundTasks):
    background_tasks.add_task(heavy_computation_background, data)
    return {"message": "Processing started"}

async def heavy_computation_background(data: ProcessingData):
    # Heavy computation in background
    result = await process_data_heavy(data)
    await save_result(result)

# ❌ Bad: Heavy computation in route
@app.post("/process-data")
def process_data(data: ProcessingData):
    # This blocks the event loop
    result = heavy_computation(data)
    return result
```

## 🔗 Integration Examples

### **1. Complete Async Route Setup**

```python
from fastapi import FastAPI, BackgroundTasks
from api.async_operations.async_route_manager import AsyncRouteManager, OperationConfig
from api.async_operations.async_patterns import AsyncDatabasePatterns, AsyncNetworkPatterns
from api.async_operations.async_middleware import MiddlewareManager, MiddlewareConfig

# Create FastAPI app
app = FastAPI(title="HeyGen AI API")

# Initialize components
route_manager = AsyncRouteManager()
db_patterns = AsyncDatabasePatterns(connection_pool)
network_patterns = AsyncNetworkPatterns()

# Setup middleware
middleware_manager = MiddlewareManager(MiddlewareConfig())
middleware_manager.setup_default_middlewares(app)

# Async routes
@app.get("/users")
async def get_users():
    return await db_patterns.execute_query("SELECT * FROM users")

@app.post("/send-email")
async def send_email(email_data: EmailData, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_email_background, email_data)
    return {"message": "Email queued for sending"}

@app.get("/external-data")
async def get_external_data():
    return await network_patterns.make_request("https://api.external.com/data")

@app.post("/process-video")
async def process_video(video_data: VideoData):
    # Use thread pool for CPU-intensive operations
    result = await route_manager.execute_operation(
        "video_processing",
        OperationType.CPU_INTENSIVE,
        ExecutionStrategy.THREAD_POOL,
        process_video_sync,
        video_data
    )
    return {"result": result}

# Background task
async def send_email_background(email_data: EmailData):
    await email_service.send_email(email_data)

# Sync function for thread pool
def process_video_sync(video_data: VideoData):
    # CPU-intensive video processing
    return video_processor.process(video_data)

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    await route_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await route_manager.cleanup()
```

### **2. Advanced Async Patterns**

```python
# Streaming responses
@app.get("/users/stream")
async def stream_users():
    return StreamingResponse(
        db_patterns.stream_query_results("SELECT * FROM users"),
        media_type="application/json"
    )

# Batch operations
@app.post("/users/batch")
async def create_users_batch(users: List[UserCreate]):
    queries = [
        {"query": "INSERT INTO users (name, email) VALUES ($1, $2)", "params": {"name": user.name, "email": user.email}}
        for user in users
    ]
    return await db_patterns.execute_batch_queries(queries)

# Cached operations
@app.get("/cached-data")
async def get_cached_data():
    return await cache_patterns.get_or_set_cached(
        "expensive_data",
        fetch_expensive_data,
        ttl=300
    )

# Rate limited operations
@app.get("/rate-limited")
async def rate_limited_operation():
    # This will be automatically rate limited by middleware
    return {"message": "Rate limited operation"}

# Circuit breaker protected operations
@app.get("/external-service")
async def external_service():
    # This will be protected by circuit breaker
    return await network_patterns.make_request("https://external-service.com")
```

## 🔧 Troubleshooting

### **1. Common Issues**

```python
# Issue: Blocking operations detected
# Solution: Use async patterns
@app.get("/users")
async def get_users():
    # Use async database operations
    return await db_patterns.execute_query("SELECT * FROM users")

# Issue: Timeout errors
# Solution: Increase timeout or use background processing
@app.post("/heavy-operation")
async def heavy_operation(data: HeavyData, background_tasks: BackgroundTasks):
    background_tasks.add_task(heavy_operation_background, data)
    return {"message": "Operation started"}

# Issue: Memory leaks
# Solution: Proper cleanup
@app.on_event("shutdown")
async def shutdown_event():
    await route_manager.cleanup()
    await network_patterns.close()
    await cache_patterns.close()
```

### **2. Performance Monitoring**

```python
# Monitor route performance
@app.get("/performance/stats")
async def get_performance_stats():
    return {
        "route_metrics": route_manager.get_route_metrics(),
        "thread_pool_stats": thread_pool.get_stats(),
        "background_queue_stats": background_queue.get_stats(),
        "middleware_stats": middleware_manager.get_middleware_stats()
    }

# Monitor blocking operations
@app.get("/blocking-operations")
async def get_blocking_operations():
    return {
        "detected_operations": route_manager.blocking_operations,
        "route_handlers": list(route_manager.route_handlers.keys())
    }
```

## 📊 Summary

### **Key Benefits**

1. **Non-Blocking Operations**: All routes use async patterns
2. **Background Processing**: Heavy operations moved to background
3. **Resource Management**: Efficient thread and process pool usage
4. **Fault Tolerance**: Circuit breaker and retry mechanisms
5. **Performance Monitoring**: Comprehensive metrics and monitoring
6. **Scalability**: Horizontal scaling support
7. **Error Handling**: Proper async error handling

### **Implementation Checklist**

- [ ] **Setup Async Route Manager**: Configure core async management
- [ ] **Implement Async Patterns**: Use async database, file, and network operations
- [ ] **Add Background Processing**: Move heavy operations to background
- [ ] **Configure Thread/Process Pools**: Handle CPU-intensive operations
- [ ] **Setup Middleware**: Add monitoring, caching, and rate limiting
- [ ] **Detect Blocking Operations**: Identify and prevent blocking code
- [ ] **Monitor Performance**: Track route performance and metrics
- [ ] **Test Async Operations**: Verify non-blocking behavior
- [ ] **Optimize Resource Usage**: Efficient resource management
- [ ] **Document Patterns**: Document async patterns and best practices

### **Next Steps**

1. **Integration**: Integrate with existing HeyGen AI services
2. **Customization**: Customize async patterns for specific needs
3. **Scaling**: Scale async operations for production workloads
4. **Advanced Patterns**: Implement advanced async patterns
5. **Automation**: Add automated blocking operation detection
6. **Reporting**: Create performance reports and dashboards

This comprehensive async route management system ensures your HeyGen AI API maintains optimal performance, prevents blocking operations, and scales efficiently to meet growing demands. 