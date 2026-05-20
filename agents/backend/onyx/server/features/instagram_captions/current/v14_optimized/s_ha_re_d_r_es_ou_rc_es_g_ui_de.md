# Shared Resources Guide for Instagram Captions API v14.0

## Overview

This guide covers the comprehensive shared resources management system for the Instagram Captions API v14.0. The system provides efficient resource pooling, caching, and lifecycle management for optimal performance and scalability.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Resource Types](#resource-types)
3. [Connection Pools](#connection-pools)
4. [Caching System](#caching-system)
5. [AI Model Management](#ai-model-management)
6. [Resource Lifecycle](#resource-lifecycle)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [API Endpoints](#api-endpoints)
9. [Usage Examples](#usage-examples)
10. [Best Practices](#best-practices)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

## Core Concepts

### What are Shared Resources?

Shared resources are system components that are expensive to create and can be reused across multiple requests. They include:

- **Connection Pools**: Database, HTTP, and Redis connections
- **AI Models**: Pre-loaded machine learning models
- **Caches**: Memory and disk-based caching systems
- **Configuration**: Application settings and parameters

### Key Benefits

- **Performance**: Reuse expensive resources instead of recreating them
- **Scalability**: Handle high concurrency with limited resources
- **Reliability**: Centralized resource management and error handling
- **Monitoring**: Comprehensive metrics and health checks
- **Efficiency**: Optimal resource utilization and cleanup

## Resource Types

### ResourceType Enum

```python
class ResourceType(Enum):
    DATABASE = "database"
    HTTP_CLIENT = "http_client"
    REDIS = "redis"
    AI_MODEL = "ai_model"
    CACHE = "cache"
    FILE_STORAGE = "file_storage"
    CONFIG = "config"
    LOGGER = "logger"
    METRICS = "metrics"
```

### Resource States

```python
class ResourceState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    CLOSED = "closed"
```

## Connection Pools

### Database Pool

The database pool manages SQLAlchemy async sessions for database operations.

```python
class DatabasePool(ConnectionPool):
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        super().__init__(pool_size, max_overflow)
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize the database pool"""
        self.engine = create_async_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=30,
            echo=False
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
```

### HTTP Client Pool

The HTTP client pool manages aiohttp sessions for external API calls.

```python
class HTTPClientPool(ConnectionPool):
    def __init__(self, timeout: int = 30, max_connections: int = 100):
        super().__init__(max_connections, max_connections // 2)
        self.timeout = timeout
        self.session = None
    
    async def initialize(self):
        """Initialize the HTTP client pool"""
        connector = aiohttp.TCPConnector(
            limit=self.pool_size,
            limit_per_host=self.pool_size // 4,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
```

### Redis Pool

The Redis pool manages aioredis connections for caching and session storage.

```python
class RedisPool(ConnectionPool):
    def __init__(self, redis_url: str, pool_size: int = 20):
        super().__init__(pool_size, pool_size // 2)
        self.redis_url = redis_url
        self.redis_pool = None
    
    async def initialize(self):
        """Initialize the Redis pool"""
        self.redis_pool = aioredis.from_url(
            self.redis_url,
            max_connections=self.pool_size,
            encoding="utf-8",
            decode_responses=True
        )
```

## Caching System

### Shared Cache

The shared cache provides in-memory caching with automatic cleanup and TTL support.

```python
class SharedCache:
    def __init__(self, cache_size: int = 10000, ttl: int = 3600):
        self.cache_size = cache_size
        self.ttl = ttl
        self.memory_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        async with self._lock:
            if key in self.memory_cache:
                # Check if expired
                if time.time() - self.cache_timestamps[key] > self.ttl:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                    return None
                
                return self.memory_cache[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in cache"""
        async with self._lock:
            # Check cache size
            if len(self.memory_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.keys(), 
                               key=lambda k: self.cache_timestamps[k])
                del self.memory_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.memory_cache[key] = value
            self.cache_timestamps[key] = time.time()
```

### Cache Features

- **Automatic Expiration**: TTL-based cache invalidation
- **Size Management**: LRU eviction when cache is full
- **Thread Safety**: Async locks for concurrent access
- **Background Cleanup**: Automatic cleanup of expired entries
- **Statistics**: Detailed cache performance metrics

## AI Model Management

### AI Model Pool

The AI model pool manages machine learning model instances with memory management.

```python
class AIModelPool:
    def __init__(self, model_cache_size: int = 5, memory_limit_mb: int = 2048):
        self.model_cache_size = model_cache_size
        self.memory_limit_mb = memory_limit_mb
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ResourceInfo] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(model_cache_size)
    
    async def get_model(self, model_name: str, loader_func: Callable[[], Any]) -> Any:
        """Get an AI model instance"""
        async with self._semaphore:
            async with self._lock:
                if model_name in self.models:
                    # Update access info
                    self.model_info[model_name].access_count += 1
                    self.model_info[model_name].last_accessed = time.time()
                    return self.models[model_name]
                
                # Load new model
                try:
                    model = await self._load_model(model_name, loader_func)
                    self.models[model_name] = model
                    self.model_info[model_name] = ResourceInfo(
                        name=model_name,
                        resource_type=ResourceType.AI_MODEL,
                        state=ResourceState.READY
                    )
                    return model
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    raise
```

### Model Features

- **Lazy Loading**: Models are loaded only when needed
- **Memory Management**: Automatic cleanup of unused models
- **Concurrency Control**: Semaphore limits concurrent model loading
- **Access Tracking**: Monitor model usage patterns
- **Error Handling**: Graceful handling of model loading failures

## Resource Lifecycle

### Initialization

Resources are initialized during application startup:

```python
async def initialize_shared_resources(config: ResourceConfig) -> SharedResources:
    """Initialize global shared resources"""
    global shared_resources
    
    if shared_resources is None:
        shared_resources = SharedResources(config)
        await shared_resources.initialize()
    
    return shared_resources
```

### Shutdown

Resources are gracefully shut down during application shutdown:

```python
async def shutdown_shared_resources():
    """Shutdown global shared resources"""
    global shared_resources
    
    if shared_resources is not None:
        await shared_resources.shutdown()
        shared_resources = None
```

### Context Managers

Context managers provide safe resource usage:

```python
@asynccontextmanager
async def database_session():
    """Context manager for database sessions"""
    resources = await get_shared_resources()
    session = await resources.get_database_session()
    try:
        yield session
    finally:
        await resources.return_database_session(session)

@asynccontextmanager
async def http_client():
    """Context manager for HTTP clients"""
    resources = await get_shared_resources()
    client = await resources.get_http_client()
    try:
        yield client
    finally:
        await resources.return_http_client(client)

@asynccontextmanager
async def redis_client():
    """Context manager for Redis clients"""
    resources = await get_shared_resources()
    client = await resources.get_redis_client()
    try:
        yield client
    finally:
        await resources.return_redis_client(client)
```

## Monitoring and Metrics

### Resource Statistics

The system provides comprehensive statistics:

```python
async def get_stats(self) -> Dict[str, Any]:
    """Get comprehensive statistics"""
    cache_stats = await self.shared_cache.get_stats()
    model_info = await self.ai_model_pool.get_model_info()
    
    return {
        **self.stats,
        "cache_stats": cache_stats,
        "model_info": model_info,
        "resource_info": self.resource_info.copy(),
        "initialized": self._initialized,
        "shutdown": self._shutdown
    }
```

### Health Checks

Automatic health checks monitor resource status:

```python
async def _health_check(self):
    """Health check for resources"""
    while not self._shutdown:
        try:
            # Check database connectivity
            try:
                session = await self.get_database_session()
                await session.close()
                await self.return_database_session(session)
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
            
            # Check Redis connectivity
            try:
                redis = await self.get_redis_client()
                await redis.ping()
                await self.return_redis_client(redis)
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")
            
            await asyncio.sleep(self.config.health_check_interval)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(60)
```

## API Endpoints

### Resource Management

```http
POST /api/v14/resources/initialize
POST /api/v14/resources/shutdown
```

### Connection Pool Status

```http
GET /api/v14/resources/pools/database/status
GET /api/v14/resources/pools/http/status
GET /api/v14/resources/pools/redis/status
```

### Cache Operations

```http
GET /api/v14/resources/cache/status
POST /api/v14/resources/cache/clear
GET /api/v14/resources/cache/{key}
POST /api/v14/resources/cache/{key}
```

### AI Model Management

```http
GET /api/v14/resources/models/status
POST /api/v14/resources/models/{model_name}/load
DELETE /api/v14/resources/models/{model_name}
```

### Monitoring

```http
GET /api/v14/resources/monitoring/stats
GET /api/v14/resources/monitoring/health
POST /api/v14/resources/testing/performance
```

## Usage Examples

### Database Operations

```python
# Using context manager
async def get_user_data(user_id: str):
    async with database_session() as session:
        result = await session.execute(
            "SELECT * FROM users WHERE id = :user_id",
            {"user_id": user_id}
        )
        return result.fetchone()

# Using pool directly
async def get_user_data_direct(user_id: str):
    resources = await get_shared_resources()
    session = await resources.get_database_session()
    try:
        result = await session.execute(
            "SELECT * FROM users WHERE id = :user_id",
            {"user_id": user_id}
        )
        return result.fetchone()
    finally:
        await resources.return_database_session(session)
```

### HTTP Client Usage

```python
# Using context manager
async def fetch_external_data(url: str):
    async with http_client() as client:
        async with client.get(url) as response:
            return await response.json()

# Using pool directly
async def fetch_external_data_direct(url: str):
    resources = await get_shared_resources()
    client = await resources.get_http_client()
    try:
        async with client.get(url) as response:
            return await response.json()
    finally:
        await resources.return_http_client(client)
```

### Caching

```python
# Using cache decorator
@with_cache(ttl=3600)
async def get_expensive_data(key: str):
    # Expensive operation
    return {"data": f"expensive_data_for_{key}"}

# Using cache directly
async def get_cached_data(key: str):
    resources = await get_shared_resources()
    
    # Try to get from cache
    cached_value = await resources.get_cache(key)
    if cached_value is not None:
        return cached_value
    
    # Generate new value
    value = await generate_expensive_data(key)
    
    # Cache the result
    await resources.set_cache(key, value, ttl=3600)
    
    return value
```

### AI Model Usage

```python
# Using model decorator
@with_ai_model("caption_generator", load_caption_model)
async def generate_caption(text: str, model):
    return await model.generate(text)

# Using pool directly
async def generate_caption_direct(text: str):
    resources = await get_shared_resources()
    model = await resources.get_ai_model("caption_generator", load_caption_model)
    return await model.generate(text)
```

## Best Practices

### 1. Use Context Managers

Always use context managers for resource access to ensure proper cleanup:

```python
# ✅ Good
async with database_session() as session:
    result = await session.execute(query)
    return result.fetchone()

# ❌ Bad
session = await resources.get_database_session()
result = await session.execute(query)
# Missing cleanup!
```

### 2. Implement Proper Error Handling

Handle resource errors gracefully:

```python
async def safe_database_operation():
    try:
        async with database_session() as session:
            result = await session.execute(query)
            return result.fetchone()
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        # Return fallback data or raise appropriate error
        return None
```

### 3. Monitor Resource Usage

Regularly monitor resource usage and performance:

```python
async def monitor_resources():
    resources = await get_shared_resources()
    stats = await resources.get_stats()
    
    # Check cache hit rate
    cache_hit_rate = stats["cache_hits"] / max(1, stats["total_requests"])
    if cache_hit_rate < 0.8:
        logger.warning(f"Low cache hit rate: {cache_hit_rate:.2%}")
    
    # Check memory usage
    memory_usage = stats["system_metrics"]["memory_usage_percent"]
    if memory_usage > 80:
        logger.warning(f"High memory usage: {memory_usage:.1f}%")
```

### 4. Configure Appropriate Pool Sizes

Set pool sizes based on your application needs:

```python
config = ResourceConfig(
    database_pool_size=20,      # Adjust based on database capacity
    http_max_connections=100,   # Adjust based on external API limits
    redis_pool_size=20,         # Adjust based on Redis capacity
    model_cache_size=5,         # Adjust based on available memory
    cache_size=10000           # Adjust based on memory constraints
)
```

### 5. Use Caching Strategically

Cache expensive operations and frequently accessed data:

```python
@with_cache(ttl=1800)  # 30 minutes
async def get_user_profile(user_id: str):
    # Expensive database query
    return await fetch_user_from_database(user_id)

@with_cache(ttl=3600)  # 1 hour
async def get_ai_model_config(model_name: str):
    # Expensive model loading
    return await load_model_config(model_name)
```

## Configuration

### ResourceConfig Options

```python
@dataclass
class ResourceConfig:
    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost/db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    
    # HTTP Client
    http_timeout: int = 30
    http_max_connections: int = 100
    http_connection_limit: int = 50
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_pool_size: int = 20
    redis_max_connections: int = 50
    
    # AI Models
    model_cache_size: int = 5
    model_load_timeout: int = 60
    model_memory_limit_mb: int = 2048
    
    # Cache
    cache_size: int = 10000
    cache_ttl: int = 3600
    cache_cleanup_interval: int = 300
    
    # Performance
    enable_monitoring: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30
```

### Environment Variables

Configure resources using environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/instagram_captions
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://localhost:6379
REDIS_POOL_SIZE=20

# Cache
CACHE_SIZE=10000
CACHE_TTL=3600

# Monitoring
ENABLE_MONITORING=true
METRICS_INTERVAL=60
HEALTH_CHECK_INTERVAL=30
```

## Troubleshooting

### Common Issues

#### 1. Connection Pool Exhaustion

**Symptoms**: Database connection errors, timeouts

**Solutions**:
- Increase pool size
- Check for connection leaks
- Monitor connection usage

```python
# Increase pool size
config = ResourceConfig(
    database_pool_size=50,  # Increase from default 20
    database_max_overflow=50  # Increase from default 30
)
```

#### 2. Memory Issues

**Symptoms**: High memory usage, OOM errors

**Solutions**:
- Reduce cache size
- Unload unused AI models
- Monitor memory usage

```python
# Reduce cache size
config = ResourceConfig(
    cache_size=5000,  # Reduce from default 10000
    model_cache_size=3  # Reduce from default 5
)
```

#### 3. Cache Performance Issues

**Symptoms**: Low cache hit rate, slow responses

**Solutions**:
- Adjust TTL values
- Increase cache size
- Monitor cache patterns

```python
# Optimize cache settings
config = ResourceConfig(
    cache_size=20000,  # Increase cache size
    cache_ttl=7200     # Increase TTL to 2 hours
)
```

### Debugging Tools

#### 1. Resource Statistics

```python
# Get comprehensive stats
stats = await resources.get_stats()
print(f"Cache hit rate: {stats['cache_hits'] / max(1, stats['total_requests']):.2%}")
print(f"Memory usage: {stats['system_metrics']['memory_usage_percent']:.1f}%")
```

#### 2. Health Checks

```python
# Check resource health
health = await get_resource_health()
for resource, status in health["health_checks"].items():
    print(f"{resource}: {status}")
```

#### 3. Performance Testing

```python
# Test resource performance
performance = await test_resource_performance()
for resource, metrics in performance["performance_results"].items():
    print(f"{resource}: {metrics['ops_per_second']:.1f} ops/sec")
```

## Conclusion

The shared resources system provides a robust foundation for managing expensive resources efficiently. Key benefits include:

- **Performance**: Reuse expensive resources instead of recreating them
- **Scalability**: Handle high concurrency with limited resources
- **Reliability**: Centralized resource management and error handling
- **Monitoring**: Comprehensive metrics and health checks
- **Efficiency**: Optimal resource utilization and cleanup

By following the patterns and best practices outlined in this guide, you can build scalable, performant applications that efficiently manage shared resources. 