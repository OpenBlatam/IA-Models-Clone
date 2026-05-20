# 🚀 PERFORMANCE OPTIMIZATION GUIDE

## Overview

This guide covers comprehensive performance optimization for AI Video systems using:
- **Async functions** for I/O-bound tasks
- **Intelligent caching strategies** for expensive operations
- **Lazy loading patterns** for resource management
- **Memory optimization** for large models
- **Background task processing** for non-blocking operations

## Table of Contents

1. [Async I/O Optimization](#async-io-optimization)
2. [Caching Strategies](#caching-strategies)
3. [Lazy Loading Patterns](#lazy-loading-patterns)
4. [Database Query Optimization](#database-query-optimization)
5. [Memory Management](#memory-management)
6. [Background Task Processing](#background-task-processing)
7. [Performance Monitoring](#performance-monitoring)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Async I/O Optimization

### Core Principles

```python
# ✅ GOOD: Use async for I/O operations
async def fetch_video_data(video_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/videos/{video_id}") as response:
            return await response.json()

# ❌ BAD: Blocking I/O in async context
async def fetch_video_data_bad(video_id: str):
    import requests
    response = requests.get(f"/api/videos/{video_id}")  # Blocking!
    return response.json()
```

### Batch Processing

```python
class AsyncIOOptimizer:
    async def batch_process_async(self, items: List[Any], processor: Callable):
        semaphore = asyncio.Semaphore(10)  # Limit concurrency
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)
        
        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### Timeout and Retry

```python
async def robust_operation():
    optimizer = AsyncIOOptimizer()
    
    # With timeout
    result = await optimizer.process_with_timeout(
        fetch_video_data("video_123"), 
        timeout=30.0
    )
    
    # With retry
    result = await optimizer.retry_with_backoff(
        lambda: fetch_video_data("video_123"),
        max_retries=3,
        base_delay=1.0
    )
```

## Caching Strategies

### Multi-Level Caching

```python
class AsyncCache:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_cache = {}  # Level 1: Memory
        self.config = CacheConfig(ttl=3600, max_size=1000)
    
    async def get(self, key: str) -> Optional[Any]:
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try Redis cache (persistent)
        if self.redis_client:
            value = await self.redis_client.get(key)
            if value:
                # Store in memory for next access
                await self._store_in_memory(key, json.loads(value))
                return json.loads(value)
        
        return None
```

### Model Caching

```python
class ModelCache:
    def __init__(self, cache: AsyncCache):
        self.cache = cache
        self.loaded_models = {}  # In-memory model storage
        self.model_loaders = {}  # Registered loaders
    
    async def get_model(self, model_name: str) -> Any:
        # Check memory first
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Try persistent cache
        cached_model = await self.cache.get(f"model:{model_name}")
        if cached_model:
            self.loaded_models[model_name] = cached_model
            return cached_model
        
        # Load using registered loader
        if model_name in self.model_loaders:
            model = await self.model_loaders[model_name]()
            self.loaded_models[model_name] = model
            await self.cache.set(f"model:{model_name}", model)
            return model
```

### Cache Configuration

```python
@dataclass
class CacheConfig:
    ttl: int = 3600                    # Time to live (seconds)
    max_size: int = 1000               # Maximum items in memory
    enable_compression: bool = True    # Enable compression
    enable_stats: bool = True          # Enable statistics
    eviction_policy: str = "lru"       # lru, lfu, ttl
```

## Lazy Loading Patterns

### Generic Lazy Loader

```python
class LazyLoader:
    def __init__(self, loader_func: Callable):
        self.loader_func = loader_func
        self._value = None
        self._loaded = False
        self._lock = asyncio.Lock()
    
    async def get(self) -> Any:
        if self._loaded:
            return self._value
        
        async with self._lock:
            if self._loaded:
                return self._value
            
            self._value = await self.loader_func()
            self._loaded = True
            return self._value
```

### Lazy Dictionary

```python
class LazyDict:
    def __init__(self):
        self._data = {}
        self._loaders = {}
        self._loaded = set()
        self._lock = asyncio.Lock()
    
    def register_loader(self, key: str, loader: Callable):
        self._loaders[key] = loader
    
    async def get(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        
        if key in self._loaders:
            async with self._lock:
                if key not in self._loaded:
                    self._data[key] = await self._loaders[key]()
                    self._loaded.add(key)
                return self._data[key]
```

### Usage Examples

```python
# Lazy model loading
async def load_stable_diffusion():
    # Expensive model loading
    return torch.load("stable_diffusion.pt")

lazy_model = LazyLoader(load_stable_diffusion)

# Model is only loaded when first accessed
model = await lazy_model.get()

# Lazy configuration loading
config_loader = LazyDict()
config_loader.register_loader("database", load_db_config)
config_loader.register_loader("redis", load_redis_config)

# Configs are loaded on-demand
db_config = await config_loader.get("database")
```

## Database Query Optimization

### Cached Queries

```python
class QueryOptimizer:
    def __init__(self, cache: AsyncCache):
        self.cache = cache
    
    async def cached_query(self, query_key: str, query_func: Callable, ttl: int = 300):
        # Try cache first
        cached_result = await self.cache.get(f"query:{query_key}")
        if cached_result:
            return cached_result
        
        # Execute query
        result = await query_func()
        
        # Cache result
        await self.cache.set(f"query:{query_key}", result, ttl)
        return result
```

### Batch Queries

```python
async def batch_get_videos(self, session: AsyncSession, video_ids: List[str]):
    # Group by cache hit/miss
    cache_hits = []
    cache_misses = []
    
    # Check cache for all videos
    for video_id in video_ids:
        cached_video = await self.cache.get(f"video:{video_id}")
        if cached_video:
            cache_hits.append(cached_video)
        else:
            cache_misses.append(video_id)
    
    # Query database for cache misses only
    db_videos = []
    if cache_misses:
        db_videos = await self._batch_query_videos_from_db(session, cache_misses)
        
        # Cache new results
        for video in db_videos:
            await self.cache.set(f"video:{video['id']}", video)
    
    # Combine and return in original order
    all_videos = cache_hits + db_videos
    video_map = {v['id']: v for v in all_videos}
    return [video_map.get(vid) for vid in video_ids if vid in video_map]
```

## Memory Management

### Model Memory Manager

```python
class ModelMemoryManager:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.loaded_models = {}
        self.model_sizes = {}
    
    async def load_model(self, model_name: str, model_data: Any, estimated_size_mb: float):
        model_size_bytes = estimated_size_mb * 1024 * 1024
        
        # Check if we have enough memory
        if not await self._can_load_model(model_size_bytes):
            # Free memory by unloading least recently used models
            await self._free_memory(model_size_bytes)
            
            if not await self._can_load_model(model_size_bytes):
                return False
        
        # Load model
        self.loaded_models[model_name] = model_data
        self.model_sizes[model_name] = model_size_bytes
        return True
```

### Memory Monitoring

```python
def get_memory_usage(self) -> Dict[str, float]:
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024
    }
```

## Background Task Processing

### Task Processor

```python
class BackgroundTaskProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
    
    async def start(self):
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
    
    async def add_task(self, task_func: Callable, *args, **kwargs):
        await self.task_queue.put((task_func, args, kwargs))
    
    async def _worker(self, worker_name: str):
        while self.running:
            try:
                task_func, args, kwargs = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                try:
                    result = await task_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
```

### Usage Examples

```python
# Initialize processor
processor = BackgroundTaskProcessor(max_workers=4)
await processor.start()

# Add background tasks
async def cleanup_video_files(video_id: str):
    # Cleanup temporary files
    pass

async def generate_thumbnails(video_path: str):
    # Generate video thumbnails
    pass

# Add tasks (non-blocking)
await processor.add_task(cleanup_video_files, "video_123")
await processor.add_task(generate_thumbnails, "/path/to/video.mp4")

# Wait for completion
await processor.task_queue.join()
```

## Performance Monitoring

### Metrics Collection

```python
@dataclass
class PerformanceMetrics:
    operation: str
    duration: float
    memory_delta: float
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: float = field(default_factory=time.time)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
        self.memory_optimizer = MemoryOptimizer()
    
    async def track_operation(self, operation: str, coro: Callable) -> Any:
        start_time = time.time()
        start_memory = self.memory_optimizer.get_memory_usage()["rss_mb"]
        
        try:
            result = await coro()
            
            duration = time.time() - start_time
            end_memory = self.memory_optimizer.get_memory_usage()["rss_mb"]
            memory_delta = end_memory - start_memory
            
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_delta=memory_delta
            )
            self.metrics.append(metric)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Operation {operation} failed after {duration:.3f}s: {e}")
            raise
```

### Performance Reports

```python
def get_performance_report(self) -> Dict[str, Any]:
    if not self.metrics:
        return {"message": "No metrics available"}
    
    durations = [m.duration for m in self.metrics]
    memory_deltas = [m.memory_delta for m in self.metrics]
    
    return {
        "total_operations": len(self.metrics),
        "avg_duration": sum(durations) / len(durations),
        "max_duration": max(durations),
        "min_duration": min(durations),
        "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
        "memory_usage": self.memory_optimizer.get_memory_usage(),
        "recent_operations": [
            {
                "operation": m.operation,
                "duration": m.duration,
                "memory_delta": m.memory_delta,
                "timestamp": m.timestamp
            }
            for m in self.metrics[-10:]
        ]
    }
```

## Best Practices

### 1. Async/Await Patterns

```python
# ✅ GOOD: Proper async patterns
async def process_video_pipeline(video_id: str):
    # I/O operations
    video_data = await fetch_video_data(video_id)
    model = await load_model("stable-diffusion")
    
    # CPU-bound operations in executor
    result = await asyncio.to_thread(process_video_sync, video_data, model)
    
    # More I/O operations
    await save_result(result)
    return result

# ❌ BAD: Mixing sync and async incorrectly
async def process_video_pipeline_bad(video_id: str):
    video_data = requests.get(f"/api/videos/{video_id}")  # Blocking!
    model = torch.load("model.pt")  # Blocking!
    return process_video(video_data, model)
```

### 2. Caching Strategy

```python
# ✅ GOOD: Multi-level caching
async def get_video_data(video_id: str):
    # Level 1: Memory cache (fastest)
    if video_id in memory_cache:
        return memory_cache[video_id]
    
    # Level 2: Redis cache (persistent)
    cached = await redis_client.get(f"video:{video_id}")
    if cached:
        memory_cache[video_id] = cached
        return cached
    
    # Level 3: Database (slowest)
    data = await db.get_video(video_id)
    await redis_client.set(f"video:{video_id}", data, ex=3600)
    memory_cache[video_id] = data
    return data
```

### 3. Memory Management

```python
# ✅ GOOD: Proper memory management
class ModelManager:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.loaded_models = {}
        self.model_sizes = {}
    
    async def load_model(self, model_name: str, model_data: Any, size_mb: float):
        # Check memory constraints
        if not self._can_load_model(size_mb):
            await self._free_memory(size_mb)
        
        # Load model
        self.loaded_models[model_name] = model_data
        self.model_sizes[model_name] = size_mb * 1024 * 1024
```

### 4. Error Handling

```python
# ✅ GOOD: Proper error handling with timeouts
async def robust_operation():
    try:
        result = await asyncio.wait_for(
            expensive_operation(),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
        raise
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

### 5. Resource Cleanup

```python
# ✅ GOOD: Proper resource cleanup
class VideoProcessor:
    def __init__(self):
        self.background_processor = BackgroundTaskProcessor()
    
    async def __aenter__(self):
        await self.background_processor.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.background_processor.stop()

# Usage
async with VideoProcessor() as processor:
    await processor.process_video("video_123")
```

## Examples

### Complete Video Processing Pipeline

```python
class OptimizedVideoPipeline:
    def __init__(self):
        self.cache = AsyncCache()
        self.model_cache = ModelCache(self.cache)
        self.memory_manager = ModelMemoryManager()
        self.background_processor = BackgroundTaskProcessor()
        self.performance_monitor = PerformanceMonitor()
    
    async def process_video(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return await self.performance_monitor.track_operation(
            "video_processing",
            lambda: self._process_video_internal(request)
        )
    
    async def _process_video_internal(self, request: Dict[str, Any]):
        # 1. Load model (lazy loading + caching)
        model = await self.model_cache.get_model(request["model_name"])
        
        # 2. Process video (async + executor for CPU-bound)
        video_data = await asyncio.to_thread(
            self._generate_video_sync, model, request
        )
        
        # 3. Save result (async I/O)
        file_path = await self._save_video_async(video_data, request["video_id"])
        
        # 4. Background cleanup
        await self.background_processor.add_task(
            self._cleanup_temp_files, request["video_id"]
        )
        
        return {
            "video_id": request["video_id"],
            "file_path": file_path,
            "status": "completed"
        }
```

### Memory-Optimized Model Loading

```python
async def load_models_with_memory_management():
    memory_manager = ModelMemoryManager(max_memory_gb=4.0)
    
    models_to_load = [
        ("stable-diffusion", load_stable_diffusion, 2048),  # 2GB
        ("text-to-video", load_text_to_video, 1536),        # 1.5GB
        ("upscaler", load_upscaler, 512),                   # 0.5GB
    ]
    
    for model_name, loader, size_mb in models_to_load:
        try:
            model_data = await loader()
            success = await memory_manager.load_model(model_name, model_data, size_mb)
            
            if success:
                logger.info(f"Loaded {model_name} successfully")
            else:
                logger.warning(f"Failed to load {model_name} - insufficient memory")
                
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
```

### Database Query Optimization

```python
async def optimized_database_queries():
    db_optimizer = QueryOptimizer(cache)
    
    # Single query with caching
    video = await db_optimizer.get_video_by_id(session, "video_123")
    
    # Batch queries
    video_ids = ["video_1", "video_2", "video_3", "video_4", "video_5"]
    videos = await db_optimizer.batch_get_videos(session, video_ids)
    
    # Cached query with custom TTL
    user_videos = await db_optimizer.cached_query(
        f"user_videos:{user_id}:{limit}:{offset}",
        lambda: query_user_videos(session, user_id, limit, offset),
        ttl=1800  # 30 minutes
    )
```

## Performance Tips

1. **Use async for all I/O operations** - Database queries, file operations, HTTP requests
2. **Run CPU-bound operations in executors** - Model inference, data processing
3. **Implement multi-level caching** - Memory → Redis → Database
4. **Use lazy loading for expensive resources** - Models, configurations, large files
5. **Monitor memory usage** - Implement automatic cleanup and eviction
6. **Process tasks in background** - Non-critical operations should be async
7. **Use timeouts and retries** - Prevent hanging operations
8. **Batch operations when possible** - Reduce overhead of multiple requests
9. **Monitor performance metrics** - Track operation times and resource usage
10. **Implement proper error handling** - Graceful degradation and recovery

## Integration with FastAPI

```python
from fastapi import FastAPI, BackgroundTasks
from performance_optimization import AIVideoPerformanceSystem

app = FastAPI()
performance_system = AIVideoPerformanceSystem()

@app.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    # Add background cleanup task
    background_tasks.add_task(performance_system.cleanup_temp_files, request.video_id)
    
    # Process video with optimization
    result = await performance_system.generate_video(request.dict())
    return result

@app.get("/system-stats")
async def get_system_stats():
    return await performance_system.get_system_stats()
```

This comprehensive performance optimization system provides the foundation for building high-performance AI Video applications with proper async patterns, intelligent caching, and efficient resource management. 