# 🚀 COMPREHENSIVE CACHING SYSTEM GUIDE

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Caching Strategies](#caching-strategies)
6. [Cache Decorators](#cache-decorators)
7. [Cache Invalidation](#cache-invalidation)
8. [Cache Warming](#cache-warming)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Performance Optimization](#performance-optimization)
11. [Best Practices](#best-practices)
12. [Integration Examples](#integration-examples)
13. [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Comprehensive Caching System provides production-ready caching for FastAPI applications with support for both Redis and in-memory caching, multiple eviction strategies, cache warming, monitoring, and seamless integration with your existing middleware system.

### Key Features

- **Multi-tier Caching** - L1 (Memory) + L2 (Redis) for optimal performance
- **Multiple Strategies** - TTL, LRU, LFU, FIFO eviction policies
- **Cache Decorators** - Easy-to-use decorators for function caching
- **Cache Invalidation** - Pattern-based cache invalidation
- **Cache Warming** - Pre-load frequently accessed data
- **Monitoring** - Real-time cache statistics and metrics
- **Compression** - Automatic data compression for large objects
- **Error Handling** - Graceful fallback when Redis is unavailable

### Benefits

- **High Performance** - Sub-millisecond cache access times
- **Scalability** - Distributed caching with Redis
- **Reliability** - Automatic fallback to memory cache
- **Ease of Use** - Simple decorators and configuration
- **Observability** - Comprehensive monitoring and metrics
- **Flexibility** - Multiple caching strategies and tiers

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                 Cache Manager                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Cache     │ │   Cache     │ │   Cache     │          │
│  │  Decorators │ │ Invalidation│ │   Warming   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                 Multi-Tier Cache                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    L1       │ │    L2       │ │   Cache     │          │
│  │  Memory     │ │   Redis     │ │  Monitor    │          │
│  │  Cache      │ │   Cache     │ │             │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                 Core Services                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Cache     │ │   Cache     │ │   Cache     │          │
│  │   Key       │ │ Serializer  │ │   Stats     │          │
│  │ Generator   │ │             │ │  Collector  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Request** → Cache Decorator
2. **Key Generation** → Cache Key Generator
3. **L1 Check** → In-Memory Cache
4. **L2 Check** → Redis Cache (if L1 miss)
5. **Data Population** → Populate L1 from L2
6. **Response** → Return cached data

## ⚙️ Configuration

### Basic Configuration

```python
from caching_system import CacheConfig, create_cache_manager

# Create basic configuration
config = CacheConfig(
    redis_url="redis://localhost:6379",
    memory_cache_size=1000,
    memory_cache_ttl=300,
    enable_multi_tier=True
)

# Create cache manager
cache_manager = create_cache_manager(config)
```

### Advanced Configuration

```python
# Production configuration
config = CacheConfig(
    # Redis configuration
    redis_url="redis://redis:6379",
    redis_ttl=3600,
    redis_max_connections=50,
    redis_retry_on_timeout=True,
    
    # In-memory cache configuration
    memory_cache_size=5000,
    memory_cache_ttl=600,
    memory_cache_strategy=CacheStrategy.LRU,
    
    # Multi-tier configuration
    enable_multi_tier=True,
    cache_tier=CacheTier.BOTH,
    
    # Cache key configuration
    key_prefix="myapp",
    key_separator=":",
    
    # Serialization configuration
    enable_compression=True,
    compression_threshold=1024,
    
    # Monitoring configuration
    enable_monitoring=True,
    monitor_interval=60,
    
    # Cache warming configuration
    enable_cache_warming=True,
    warming_batch_size=100
)
```

### Environment-based Configuration

```python
import os
from caching_system import CacheConfig

def create_cache_config_from_env() -> CacheConfig:
    """Create cache configuration from environment variables."""
    return CacheConfig(
        redis_url=os.getenv("REDIS_URL"),
        memory_cache_size=int(os.getenv("MEMORY_CACHE_SIZE", "1000")),
        memory_cache_ttl=int(os.getenv("MEMORY_CACHE_TTL", "300")),
        memory_cache_strategy=CacheStrategy(os.getenv("CACHE_STRATEGY", "TTL")),
        enable_multi_tier=os.getenv("ENABLE_MULTI_TIER", "true").lower() == "true",
        enable_compression=os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
        enable_monitoring=os.getenv("ENABLE_CACHE_MONITORING", "true").lower() == "true",
        enable_cache_warming=os.getenv("ENABLE_CACHE_WARMING", "false").lower() == "true"
    )
```

## 💡 Usage Examples

### Basic Setup

```python
from fastapi import FastAPI
from caching_system import CacheConfig, create_cache_manager

# Create FastAPI app
app = FastAPI(title="My API", version="1.0.0")

# Create cache configuration
config = CacheConfig(
    redis_url="redis://localhost:6379",
    memory_cache_size=1000,
    enable_multi_tier=True
)

# Create cache manager
cache_manager = create_cache_manager(config)

# Start cache manager
@app.on_event("startup")
async def startup_event():
    await cache_manager.start()

# Stop cache manager
@app.on_event("shutdown")
async def shutdown_event():
    await cache_manager.stop()

# Add your routes
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### Using Cache Decorators

```python
from fastapi import FastAPI, HTTPException
from caching_system import create_cache_manager, CacheConfig

app = FastAPI()
config = CacheConfig(redis_url="redis://localhost:6379")
cache_manager = create_cache_manager(config)

# Cache expensive operations
@cache_manager.cached(ttl=3600, key_prefix="user_profile")
async def get_user_profile(user_id: int):
    # Simulate expensive database call
    await asyncio.sleep(0.1)
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "profile": f"Profile data for user {user_id}"
    }

# Cache with different TTL for different data types
@cache_manager.cached(ttl=1800, key_prefix="user_settings")
async def get_user_settings(user_id: int):
    # Simulate database call
    await asyncio.sleep(0.05)
    return {
        "user_id": user_id,
        "theme": "dark",
        "language": "en",
        "notifications": True
    }

# Cache invalidation
@cache_manager.cache_invalidate("user_profile:*")
async def update_user_profile(user_id: int, profile_data: dict):
    # Update user profile in database
    await update_user_in_db(user_id, profile_data)
    return {"user_id": user_id, "updated": True}

@app.get("/users/{user_id}/profile")
async def get_user_profile_endpoint(user_id: int):
    try:
        profile = await get_user_profile(user_id)
        return profile
    except Exception as e:
        raise HTTPException(status_code=404, detail="User not found")

@app.put("/users/{user_id}/profile")
async def update_user_profile_endpoint(user_id: int, profile_data: dict):
    result = await update_user_profile(user_id, profile_data)
    return result
```

### Manual Cache Operations

```python
# Direct cache operations
async def manual_cache_example():
    # Set cache entry
    await cache_manager.set("custom_key", {"data": "value"}, ttl=1800)
    
    # Get cache entry
    data = await cache_manager.get("custom_key")
    
    # Delete cache entry
    await cache_manager.delete("custom_key")
    
    # Clear pattern
    await cache_manager.clear_pattern("user:*")
    
    # Get cache statistics
    stats = await cache_manager.get_stats()
    print(f"Cache hit rate: {stats['l1_cache']['hit_rate']}")
```

### Cache Warming

```python
# Cache warming for frequently accessed data
async def warm_user_cache():
    async def user_data_source():
        # Fetch frequently accessed users from database
        users = await get_frequently_accessed_users()
        return [(f"user:{user['id']}", user) for user in users]
    
    await cache_manager.warm_cache(user_data_source)

# Warm cache on startup
@app.on_event("startup")
async def startup_event():
    await cache_manager.start()
    if cache_manager.config.enable_cache_warming:
        await warm_user_cache()
```

## 🎯 Caching Strategies

### Cache Eviction Strategies

#### TTL (Time To Live)
```python
config = CacheConfig(
    memory_cache_strategy=CacheStrategy.TTL,
    memory_cache_ttl=300  # 5 minutes
)
```
- **Use case**: Data that expires naturally
- **Example**: API responses, session data
- **Pros**: Simple, predictable expiration
- **Cons**: May expire while still needed

#### LRU (Least Recently Used)
```python
config = CacheConfig(
    memory_cache_strategy=CacheStrategy.LRU,
    memory_cache_size=1000
)
```
- **Use case**: Frequently accessed data
- **Example**: User profiles, configuration
- **Pros**: Keeps most used data
- **Cons**: May evict data that will be needed soon

#### LFU (Least Frequently Used)
```python
config = CacheConfig(
    memory_cache_strategy=CacheStrategy.LFU,
    memory_cache_size=1000
)
```
- **Use case**: Data with varying access patterns
- **Example**: Product catalog, search results
- **Pros**: Keeps most frequently used data
- **Cons**: May keep stale data if access patterns change

### Cache Tiers

#### L1 Only (Memory Cache)
```python
config = CacheConfig(
    enable_multi_tier=False,
    memory_cache_size=1000
)
```
- **Use case**: Single-instance applications
- **Pros**: Fastest access, no network overhead
- **Cons**: Not shared between instances

#### L2 Only (Redis Cache)
```python
config = CacheConfig(
    enable_multi_tier=False,
    cache_tier=CacheTier.L2
)
```
- **Use case**: Distributed applications
- **Pros**: Shared between instances
- **Cons**: Network overhead, slower than memory

#### Multi-tier (L1 + L2)
```python
config = CacheConfig(
    enable_multi_tier=True,
    cache_tier=CacheTier.BOTH
)
```
- **Use case**: High-performance distributed applications
- **Pros**: Best of both worlds
- **Cons**: More complex, requires Redis

## 🔧 Cache Decorators

### Basic Caching

```python
@cache_manager.cached(ttl=3600)
async def expensive_function(param1: str, param2: int):
    # Expensive operation
    return {"result": f"{param1}_{param2}"}
```

### Custom Key Prefix

```python
@cache_manager.cached(ttl=1800, key_prefix="user_data")
async def get_user_data(user_id: int, include_profile: bool = True):
    # Function implementation
    return {"user_id": user_id, "data": "..."}
```

### Tier-specific Caching

```python
# Cache only in memory (L1)
@cache_manager.cached(ttl=300, tier=CacheTier.L1)
async def fast_access_data(key: str):
    return {"key": key, "data": "..."}

# Cache only in Redis (L2)
@cache_manager.cached(ttl=3600, tier=CacheTier.L2)
async def shared_data(key: str):
    return {"key": key, "data": "..."}
```

### Cache Invalidation

```python
# Invalidate specific patterns
@cache_manager.cache_invalidate("user:*")
async def update_user(user_id: int, data: dict):
    # Update user
    return {"user_id": user_id, "updated": True}

# Invalidate multiple patterns
@cache_manager.cache_invalidate("user:*", "profile:*")
async def delete_user(user_id: int):
    # Delete user
    return {"user_id": user_id, "deleted": True}
```

## 🗑️ Cache Invalidation

### Pattern-based Invalidation

```python
# Invalidate all user-related cache
await cache_manager.clear_pattern("user:*")

# Invalidate specific user cache
await cache_manager.clear_pattern(f"user:{user_id}:*")

# Invalidate all cache
await cache_manager.clear_pattern("*")
```

### Manual Invalidation

```python
# Invalidate specific keys
await cache_manager.delete("user:123")
await cache_manager.delete("profile:123")

# Invalidate in specific tier
await cache_manager.delete("user:123", tier=CacheTier.L1)
await cache_manager.delete("user:123", tier=CacheTier.L2)
```

### Automatic Invalidation

```python
# Invalidate on data updates
@cache_manager.cache_invalidate("user:*")
async def update_user_data(user_id: int, data: dict):
    # Update database
    await db.update_user(user_id, data)
    return {"updated": True}

# Invalidate on data deletion
@cache_manager.cache_invalidate("user:*", "profile:*")
async def delete_user_data(user_id: int):
    # Delete from database
    await db.delete_user(user_id)
    return {"deleted": True}
```

## 🔥 Cache Warming

### Pre-loading Frequently Accessed Data

```python
async def warm_frequently_accessed_data():
    async def data_source():
        # Fetch frequently accessed data
        popular_products = await get_popular_products()
        active_users = await get_active_users()
        system_config = await get_system_config()
        
        data = []
        
        # Add products
        for product in popular_products:
            data.append((f"product:{product['id']}", product))
        
        # Add users
        for user in active_users:
            data.append((f"user:{user['id']}", user))
        
        # Add config
        data.append(("config:system", system_config))
        
        return data
    
    await cache_manager.warm_cache(data_source)
```

### Scheduled Cache Warming

```python
import asyncio
from datetime import datetime

async def scheduled_cache_warming():
    """Warm cache on schedule."""
    while True:
        try:
            print(f"Warming cache at {datetime.now()}")
            await warm_frequently_accessed_data()
            
            # Wait for next warming cycle (e.g., every hour)
            await asyncio.sleep(3600)
            
        except Exception as e:
            print(f"Cache warming failed: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry

# Start scheduled warming
@app.on_event("startup")
async def startup_event():
    await cache_manager.start()
    asyncio.create_task(scheduled_cache_warming())
```

### Conditional Cache Warming

```python
async def conditional_cache_warming():
    """Warm cache based on conditions."""
    
    # Check if cache is cold (low hit rate)
    stats = await cache_manager.get_stats()
    l1_hit_rate = stats["l1_cache"]["hit_rate"]
    l2_hit_rate = stats["l2_cache"]["hit_rate"]
    
    if l1_hit_rate < 0.5 or l2_hit_rate < 0.3:
        print("Cache hit rate is low, warming cache...")
        await warm_frequently_accessed_data()
    else:
        print("Cache hit rate is good, skipping warming")
```

## 📊 Monitoring and Metrics

### Cache Statistics

```python
async def get_cache_statistics():
    """Get comprehensive cache statistics."""
    stats = await cache_manager.get_stats()
    
    return {
        "l1_cache": {
            "hits": stats["l1_cache"]["hits"],
            "misses": stats["l1_cache"]["misses"],
            "hit_rate": stats["l1_cache"]["hit_rate"],
            "size": stats["l1_cache"]["size"],
            "max_size": stats["l1_cache"]["max_size"]
        },
        "l2_cache": {
            "hits": stats["l2_cache"]["hits"],
            "misses": stats["l2_cache"]["misses"],
            "hit_rate": stats["l2_cache"]["hit_rate"],
            "connected": stats["l2_cache"]["connected"],
            "errors": stats["l2_cache"]["errors"]
        },
        "total_operations": {
            "hits": stats["total_hits"],
            "misses": stats["total_misses"],
            "sets": stats["total_sets"]
        }
    }
```

### Cache Health Check

```python
@app.get("/health/cache")
async def cache_health_check():
    """Cache health check endpoint."""
    stats = await cache_manager.get_stats()
    
    # Check L1 cache health
    l1_healthy = stats["l1_cache"]["hit_rate"] > 0.1
    
    # Check L2 cache health
    l2_healthy = stats["l2_cache"]["connected"] and stats["l2_cache"]["errors"] < 10
    
    overall_healthy = l1_healthy and l2_healthy
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "l1_cache": {
            "status": "healthy" if l1_healthy else "unhealthy",
            "hit_rate": stats["l1_cache"]["hit_rate"]
        },
        "l2_cache": {
            "status": "healthy" if l2_healthy else "unhealthy",
            "connected": stats["l2_cache"]["connected"],
            "errors": stats["l2_cache"]["errors"]
        },
        "timestamp": datetime.now().isoformat()
    }
```

### Cache Monitoring Dashboard

```python
@app.get("/metrics/cache")
async def cache_metrics():
    """Cache metrics for monitoring dashboard."""
    stats = await cache_manager.get_stats()
    
    return {
        "cache_hit_rate": {
            "l1": stats["l1_cache"]["hit_rate"],
            "l2": stats["l2_cache"]["hit_rate"],
            "overall": (stats["total_hits"] / (stats["total_hits"] + stats["total_misses"])) if (stats["total_hits"] + stats["total_misses"]) > 0 else 0
        },
        "cache_operations": {
            "hits": stats["total_hits"],
            "misses": stats["total_misses"],
            "sets": stats["total_sets"]
        },
        "cache_size": {
            "l1_current": stats["l1_cache"]["size"],
            "l1_max": stats["l1_cache"]["max_size"],
            "l1_utilization": stats["l1_cache"]["size"] / stats["l1_cache"]["max_size"]
        },
        "cache_errors": {
            "l2_errors": stats["l2_cache"]["errors"]
        }
    }
```

## ⚡ Performance Optimization

### Cache Key Optimization

```python
# Use consistent, short keys
@cache_manager.cached(key_prefix="u")  # Short prefix
async def get_user(user_id: int):
    return {"id": user_id, "name": f"User {user_id}"}

# Use hash keys for long parameters
@cache_manager.cached(key_prefix="search")
async def search_products(query: str, filters: dict):
    # Generate hash for long query + filters
    key_data = f"{query}_{json.dumps(filters, sort_keys=True)}"
    return {"query": query, "results": []}
```

### TTL Optimization

```python
# Different TTL for different data types
@cache_manager.cached(ttl=3600, key_prefix="user_profile")  # 1 hour
async def get_user_profile(user_id: int):
    return {"profile": "..."}

@cache_manager.cached(ttl=300, key_prefix="user_session")   # 5 minutes
async def get_user_session(user_id: int):
    return {"session": "..."}

@cache_manager.cached(ttl=86400, key_prefix="config")       # 24 hours
async def get_system_config():
    return {"config": "..."}
```

### Compression Optimization

```python
# Enable compression for large objects
config = CacheConfig(
    enable_compression=True,
    compression_threshold=512  # Compress objects > 512 bytes
)

# Disable compression for small objects
config = CacheConfig(
    enable_compression=False  # No compression overhead
)
```

### Memory Cache Size Optimization

```python
# Large memory cache for high-traffic applications
config = CacheConfig(
    memory_cache_size=10000,  # 10K entries
    memory_cache_strategy=CacheStrategy.LRU
)

# Small memory cache for memory-constrained environments
config = CacheConfig(
    memory_cache_size=100,    # 100 entries
    memory_cache_strategy=CacheStrategy.TTL
)
```

## 🎯 Best Practices

### 1. Cache Key Design

**Good Keys:**
```python
# Consistent, descriptive keys
"user:123:profile"
"product:456:details"
"config:system:settings"
```

**Bad Keys:**
```python
# Inconsistent, unclear keys
"u123p"
"prod456"
"cfg"
```

### 2. TTL Strategy

```python
# Use appropriate TTL based on data characteristics
@cache_manager.cached(ttl=3600)    # Static data: 1 hour
async def get_system_config():
    pass

@cache_manager.cached(ttl=300)     # Semi-dynamic: 5 minutes
async def get_user_preferences(user_id: int):
    pass

@cache_manager.cached(ttl=60)      # Dynamic data: 1 minute
async def get_user_status(user_id: int):
    pass
```

### 3. Cache Invalidation Strategy

```python
# Invalidate related data together
@cache_manager.cache_invalidate("user:*", "profile:*")
async def update_user(user_id: int, data: dict):
    # Update user and related data
    pass

# Use specific patterns for targeted invalidation
@cache_manager.cache_invalidate(f"user:{user_id}:*")
async def update_user_specific(user_id: int, data: dict):
    # Update specific user only
    pass
```

### 4. Error Handling

```python
# Graceful fallback when cache fails
async def get_user_with_fallback(user_id: int):
    try:
        # Try cache first
        cached_user = await cache_manager.get(f"user:{user_id}")
        if cached_user:
            return cached_user
        
        # Fallback to database
        user = await get_user_from_db(user_id)
        
        # Cache for next time
        await cache_manager.set(f"user:{user_id}", user, ttl=3600)
        return user
        
    except Exception as e:
        # Log error and fallback to database
        logger.error(f"Cache error: {e}")
        return await get_user_from_db(user_id)
```

### 5. Cache Warming Strategy

```python
# Warm cache with most important data first
async def prioritized_cache_warming():
    # High priority: Frequently accessed data
    await warm_user_profiles()
    
    # Medium priority: Configuration data
    await warm_system_config()
    
    # Low priority: Historical data
    await warm_analytics_data()
```

## 🔗 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Depends, HTTPException
from caching_system import create_cache_manager, CacheConfig

app = FastAPI()

# Create cache manager
config = CacheConfig(
    redis_url="redis://redis:6379",
    memory_cache_size=1000,
    enable_multi_tier=True
)
cache_manager = create_cache_manager(config)

# Dependency for cache manager
async def get_cache_manager():
    return cache_manager

# Cached endpoint
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    cache_manager = Depends(get_cache_manager)
):
    @cache_manager.cached(ttl=1800, key_prefix="user")
    async def get_user_data(user_id: int):
        # Simulate database call
        await asyncio.sleep(0.1)
        return {"id": user_id, "name": f"User {user_id}"}
    
    return await get_user_data(user_id)

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    await cache_manager.start()

@app.on_event("shutdown")
async def shutdown_event():
    await cache_manager.stop()
```

### Database Integration

```python
from sqlalchemy.ext.asyncio import AsyncSession
from caching_system import create_cache_manager

# Cache database queries
@cache_manager.cached(ttl=3600, key_prefix="db_user")
async def get_user_from_db(session: AsyncSession, user_id: int):
    result = await session.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    return user.to_dict() if user else None

# Cache invalidation on updates
@cache_manager.cache_invalidate("db_user:*")
async def update_user_in_db(session: AsyncSession, user_id: int, data: dict):
    user = await session.get(User, user_id)
    if user:
        for key, value in data.items():
            setattr(user, key, value)
        await session.commit()
        return user.to_dict()
    return None
```

### External API Integration

```python
import httpx
from caching_system import create_cache_manager

cache_manager = create_cache_manager()

# Cache external API calls
@cache_manager.cached(ttl=1800, key_prefix="api_weather")
async def get_weather_data(city: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        response.raise_for_status()
        return response.json()

# Cache with different TTL for different APIs
@cache_manager.cached(ttl=3600, key_prefix="api_exchange")
async def get_exchange_rates():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.exchange.com/rates")
        response.raise_for_status()
        return response.json()
```

### Background Task Integration

```python
from fastapi import BackgroundTasks
from caching_system import create_cache_manager

cache_manager = create_cache_manager()

# Background cache warming
async def warm_cache_background():
    async def data_source():
        # Fetch data for warming
        return [("key1", "value1"), ("key2", "value2")]
    
    await cache_manager.warm_cache(data_source)

@app.post("/trigger-cache-warming")
async def trigger_cache_warming(background_tasks: BackgroundTasks):
    background_tasks.add_task(warm_cache_background)
    return {"message": "Cache warming started"}
```

## 🐛 Troubleshooting

### Common Issues

#### Low Cache Hit Rate
```python
# Check cache statistics
stats = await cache_manager.get_stats()
print(f"L1 hit rate: {stats['l1_cache']['hit_rate']}")
print(f"L2 hit rate: {stats['l2_cache']['hit_rate']}")

# Solutions:
# 1. Increase cache size
config = CacheConfig(memory_cache_size=5000)

# 2. Adjust TTL
@cache_manager.cached(ttl=7200)  # Increase TTL

# 3. Implement cache warming
await warm_frequently_accessed_data()
```

#### Redis Connection Issues
```python
# Check Redis connection
stats = await cache_manager.get_stats()
if not stats["l2_cache"]["connected"]:
    print("Redis not connected, using memory cache only")

# Solutions:
# 1. Check Redis URL
config = CacheConfig(redis_url="redis://localhost:6379")

# 2. Check Redis server
# redis-cli ping

# 3. Use fallback configuration
config = CacheConfig(
    redis_url=None,  # Disable Redis
    enable_multi_tier=False
)
```

#### High Memory Usage
```python
# Check memory cache utilization
stats = await cache_manager.get_stats()
utilization = stats["l1_cache"]["size"] / stats["l1_cache"]["max_size"]
print(f"Memory cache utilization: {utilization}")

# Solutions:
# 1. Reduce cache size
config = CacheConfig(memory_cache_size=500)

# 2. Use TTL strategy
config = CacheConfig(memory_cache_strategy=CacheStrategy.TTL)

# 3. Clear cache periodically
await cache_manager.clear_pattern("*")
```

#### Slow Cache Operations
```python
# Profile cache operations
import time

start_time = time.time()
result = await cache_manager.get("test_key")
end_time = time.time()
print(f"Cache get time: {(end_time - start_time) * 1000}ms")

# Solutions:
# 1. Use L1 cache only for fast access
@cache_manager.cached(tier=CacheTier.L1)

# 2. Optimize serialization
config = CacheConfig(enable_compression=False)

# 3. Use smaller cache keys
@cache_manager.cached(key_prefix="u")  # Short prefix
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("caching_system").setLevel(logging.DEBUG)

# Check cache operations
@cache_manager.cached(ttl=3600, key_prefix="debug")
async def debug_function(param: str):
    print(f"Function called with: {param}")
    return {"param": param, "timestamp": datetime.now().isoformat()}
```

### Performance Monitoring

```python
# Monitor cache performance
async def monitor_cache_performance():
    while True:
        stats = await cache_manager.get_stats()
        
        # Log performance metrics
        logger.info(
            "Cache performance",
            l1_hit_rate=stats["l1_cache"]["hit_rate"],
            l2_hit_rate=stats["l2_cache"]["hit_rate"],
            l1_size=stats["l1_cache"]["size"],
            l2_errors=stats["l2_cache"]["errors"]
        )
        
        # Alert on low performance
        if stats["l1_cache"]["hit_rate"] < 0.3:
            logger.warning("Low L1 cache hit rate detected")
        
        await asyncio.sleep(60)  # Check every minute
```

## 📚 Summary

The Comprehensive Caching System provides:

### ✅ **Complete Solution**
- Multi-tier caching (L1 + L2)
- Multiple eviction strategies
- Cache decorators and invalidation
- Cache warming and monitoring
- Compression and error handling

### ✅ **High Performance**
- Sub-millisecond access times
- Automatic L1 population from L2
- Compression for large objects
- Optimized key generation

### ✅ **Production Ready**
- Redis integration for distributed caching
- Graceful fallback when Redis is unavailable
- Comprehensive monitoring and metrics
- Error handling and recovery

### ✅ **Easy Integration**
- Simple decorators for caching
- Automatic cache key generation
- Pattern-based invalidation
- Background cache warming

### ✅ **Flexible Configuration**
- Environment-based configuration
- Multiple caching strategies
- Configurable TTL and sizes
- Monitoring and warming options

This caching system provides everything needed to implement high-performance, scalable caching in your FastAPI applications, with comprehensive monitoring and easy integration with your existing codebase. 