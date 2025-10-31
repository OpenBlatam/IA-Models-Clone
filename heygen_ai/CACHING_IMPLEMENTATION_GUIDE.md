# Caching Implementation Guide

A comprehensive guide for implementing caching for static and frequently accessed data using Redis and in-memory stores in the HeyGen AI FastAPI application.

## 🎯 Overview

This guide covers:
- **Multi-Level Caching**: Memory, Redis, and hybrid caching strategies
- **Specialized Caches**: User data, video data, static content, and analytics
- **Cache Policies**: TTL, eviction strategies, and compression
- **Cache Decorators**: Easy-to-use caching utilities
- **Performance Optimization**: Cache hit rates and monitoring
- **Best Practices**: Cache invalidation and data consistency

## 📋 Table of Contents

1. [Cache Architecture](#cache-architecture)
2. [Cache Types and Strategies](#cache-types-and-strategies)
3. [Specialized Caches](#specialized-caches)
4. [Cache Decorators](#cache-decorators)
5. [Integration Examples](#integration-examples)
6. [Performance Monitoring](#performance-monitoring)
7. [Best Practices](#best-practices)
8. [Cache Invalidation](#cache-invalidation)

## 🏗️ Cache Architecture

### Overview

The caching system provides multiple levels of caching with different strategies for optimal performance.

### Multi-Level Caching

#### **Memory Cache (Fastest)**
```python
from api.caching.cache_manager import MemoryCache, CacheStrategy

# LRU memory cache
memory_cache = MemoryCache(
    max_size=1000,
    strategy=CacheStrategy.LRU,
    ttl_enabled=True
)

# Store data in memory
memory_cache.set("user:123", user_data, ttl=600)

# Retrieve from memory
user_data = memory_cache.get("user:123")
```

#### **Redis Cache (Persistent)**
```python
from api.caching.cache_manager import RedisCache

# Redis cache with connection pooling
redis_cache = RedisCache(
    redis_url="redis://localhost:6379",
    connection_pool_size=10,
    compression_enabled=True
)

# Store data in Redis
await redis_cache.set("video:456", video_data, ttl=1800)

# Retrieve from Redis
video_data = await redis_cache.get("video:456")
```

#### **Hybrid Cache (Recommended)**
```python
from api.caching.cache_manager import HybridCache

# Hybrid cache (memory + Redis)
hybrid_cache = HybridCache(
    memory_cache=memory_cache,
    redis_cache=redis_cache,
    memory_first=True
)

# Store data in both memory and Redis
await hybrid_cache.set("user:123", user_data, ttl=600)

# Retrieve (tries memory first, then Redis)
user_data = await hybrid_cache.get("user:123")
```

#### **Tiered Cache (Advanced)**
```python
from api.caching.cache_manager import TieredCache

# Multi-tier cache system
tiered_cache = TieredCache([
    memory_cache,  # L1: Fastest
    redis_cache    # L2: Persistent
])

# Automatic promotion between tiers
user_data = await tiered_cache.get("user:123")
```

## 🗄️ Cache Types and Strategies

### Cache Configuration

#### **Cache Policy Configuration**
```python
from api.caching.specialized_caches import CachePolicy, DataType, CacheStrategy

# User data cache policy
user_cache_policy = CachePolicy(
    data_type=DataType.USER_DATA,
    ttl=1800,  # 30 minutes
    max_size=1000,
    strategy=CacheStrategy.LRU,
    priority="high",
    compression_enabled=True,
    auto_refresh=True,
    refresh_interval=900  # 15 minutes
)

# Video data cache policy
video_cache_policy = CachePolicy(
    data_type=DataType.VIDEO_DATA,
    ttl=3600,  # 1 hour
    max_size=500,
    strategy=CacheStrategy.LRU,
    priority="high",
    compression_enabled=True,
    auto_refresh=False
)

# Static content cache policy
static_cache_policy = CachePolicy(
    data_type=DataType.STATIC_CONTENT,
    ttl=86400,  # 24 hours
    max_size=200,
    strategy=CacheStrategy.LRU,
    priority="normal",
    compression_enabled=True,
    auto_refresh=False
)
```

### Cache Manager Setup

#### **Main Cache Manager**
```python
from api.caching.cache_manager import CacheManager, CacheConfig, CacheType

# Configure cache manager
cache_config = CacheConfig(
    cache_type=CacheType.HYBRID,
    strategy=CacheStrategy.LRU,
    max_size=1000,
    default_ttl=300,
    compression_enabled=True,
    enable_stats=True,
    redis_url="redis://localhost:6379",
    redis_connection_pool_size=10
)

# Initialize cache manager
cache_manager = CacheManager(cache_config)

# Use cache manager
await cache_manager.set("key", value, ttl=600)
cached_value = await cache_manager.get("key")
```

## 🎯 Specialized Caches

### User Data Cache

#### **User Profile Caching**
```python
from api.caching.specialized_caches import UserDataCache

user_cache = UserDataCache(cache_manager)

# Cache user profile
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile with caching."""
    # Try cache first
    cached_profile = await user_cache.get_user_profile(user_id)
    if cached_profile:
        return cached_profile
    
    # Fetch from database
    profile = await user_service.get_profile(user_id)
    
    # Cache result
    await user_cache.set_user_profile(user_id, profile)
    
    return profile

# Cache user videos
async def get_user_videos(user_id: str, page: int = 1) -> Dict[str, Any]:
    """Get user videos with caching."""
    cached_videos = await user_cache.get_user_videos(user_id, page)
    if cached_videos:
        return cached_videos
    
    videos = await video_service.get_user_videos(user_id, page)
    await user_cache.set_user_videos(user_id, videos, page)
    
    return videos

# Cache user analytics
async def get_user_analytics(user_id: str) -> Dict[str, Any]:
    """Get user analytics with caching."""
    cached_analytics = await user_cache.get_user_analytics(user_id)
    if cached_analytics:
        return cached_analytics
    
    analytics = await analytics_service.get_user_analytics(user_id)
    await user_cache.set_user_analytics(user_id, analytics)
    
    return analytics
```

#### **Session Management**
```python
# Cache user sessions
async def create_user_session(user_id: str, session_data: Dict[str, Any]) -> str:
    """Create user session with caching."""
    session_id = str(uuid.uuid4())
    await user_cache.set_user_session(session_id, session_data, ttl=86400)
    return session_id

async def get_user_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get user session from cache."""
    return await user_cache.get_user_session(session_id)

async def delete_user_session(session_id: str) -> bool:
    """Delete user session from cache."""
    return await user_cache.delete_user_session(session_id)
```

### Video Data Cache

#### **Video Metadata Caching**
```python
from api.caching.specialized_caches import VideoDataCache

video_cache = VideoDataCache(cache_manager)

# Cache video metadata
async def get_video_metadata(video_id: str) -> Dict[str, Any]:
    """Get video metadata with caching."""
    cached_metadata = await video_cache.get_video_metadata(video_id)
    if cached_metadata:
        return cached_metadata
    
    metadata = await video_service.get_metadata(video_id)
    await video_cache.set_video_metadata(video_id, metadata)
    
    return metadata

# Cache video status
async def get_video_status(video_id: str) -> Dict[str, Any]:
    """Get video status with caching."""
    cached_status = await video_cache.get_video_status(video_id)
    if cached_status:
        return cached_status
    
    status = await video_service.get_status(video_id)
    await video_cache.set_video_status(video_id, status)
    
    return status

# Cache video analytics
async def get_video_analytics(video_id: str) -> Dict[str, Any]:
    """Get video analytics with caching."""
    cached_analytics = await video_cache.get_video_analytics(video_id)
    if cached_analytics:
        return cached_analytics
    
    analytics = await analytics_service.get_video_analytics(video_id)
    await video_cache.set_video_analytics(video_id, analytics)
    
    return analytics
```

#### **Processing Queue Management**
```python
# Manage video processing queue
async def add_video_to_queue(user_id: str, video_id: str):
    """Add video to processing queue."""
    await video_cache.add_to_processing_queue(user_id, video_id)

async def get_processing_queue(user_id: str) -> List[str]:
    """Get user's processing queue."""
    return await video_cache.get_video_processing_queue(user_id) or []

async def remove_from_queue(user_id: str, video_id: str):
    """Remove video from processing queue."""
    await video_cache.remove_from_processing_queue(user_id, video_id)
```

### Static Content Cache

#### **Template and Configuration Caching**
```python
from api.caching.specialized_caches import StaticContentCache

static_cache = StaticContentCache(cache_manager)

# Cache template data
async def get_template_data(template_id: str) -> Dict[str, Any]:
    """Get template data with caching."""
    cached_template = await static_cache.get_template_data(template_id)
    if cached_template:
        return cached_template
    
    template = await template_service.get_template(template_id)
    await static_cache.set_template_data(template_id, template)
    
    return template

# Cache configuration
async def get_configuration(config_key: str) -> Dict[str, Any]:
    """Get configuration with caching."""
    cached_config = await static_cache.get_configuration(config_key)
    if cached_config:
        return cached_config
    
    config = await config_service.get_config(config_key)
    await static_cache.set_configuration(config_key, config)
    
    return config

# Cache API documentation
async def get_api_documentation(version: str = "v1") -> Dict[str, Any]:
    """Get API documentation with caching."""
    cached_docs = await static_cache.get_api_documentation(version)
    if cached_docs:
        return cached_docs
    
    docs = await docs_service.get_documentation(version)
    await static_cache.set_api_documentation(version, docs)
    
    return docs
```

#### **System Status and Rate Limits**
```python
# Cache system status
async def get_system_status() -> Dict[str, Any]:
    """Get system status with caching."""
    cached_status = await static_cache.get_system_status()
    if cached_status:
        return cached_status
    
    status = await system_service.get_status()
    await static_cache.set_system_status(status)
    
    return status

# Cache rate limits
async def check_rate_limit(user_id: str, action: str) -> bool:
    """Check rate limit with caching."""
    limit_info = await static_cache.get_rate_limits(user_id, action)
    if limit_info:
        return limit_info["allowed"]
    
    # Calculate rate limit
    allowed = await rate_limit_service.check_limit(user_id, action)
    await static_cache.set_rate_limits(user_id, action, {"allowed": allowed})
    
    return allowed
```

### Frequently Accessed Data Cache

#### **Search Results and Recommendations**
```python
from api.caching.specialized_caches import FrequentlyAccessedCache

frequent_cache = FrequentlyAccessedCache(cache_manager)

# Cache search results
async def search_videos(query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Search videos with caching."""
    cached_results = await frequent_cache.get_search_results(query, filters)
    if cached_results:
        return cached_results
    
    results = await search_service.search_videos(query, filters)
    await frequent_cache.set_search_results(query, results, filters)
    
    return results

# Cache popular videos
async def get_popular_videos(category: str = "all", limit: int = 20) -> List[Dict[str, Any]]:
    """Get popular videos with caching."""
    cached_videos = await frequent_cache.get_popular_videos(category, limit)
    if cached_videos:
        return cached_videos
    
    videos = await video_service.get_popular_videos(category, limit)
    await frequent_cache.set_popular_videos(category, videos, limit)
    
    return videos

# Cache user recommendations
async def get_user_recommendations(user_id: str) -> List[Dict[str, Any]]:
    """Get user recommendations with caching."""
    cached_recommendations = await frequent_cache.get_user_recommendations(user_id)
    if cached_recommendations:
        return cached_recommendations
    
    recommendations = await recommendation_service.get_recommendations(user_id)
    await frequent_cache.set_user_recommendations(user_id, recommendations)
    
    return recommendations
```

#### **Trending Topics and Analytics**
```python
# Cache trending topics
async def get_trending_topics() -> List[str]:
    """Get trending topics with caching."""
    cached_topics = await frequent_cache.get_trending_topics()
    if cached_topics:
        return cached_topics
    
    topics = await analytics_service.get_trending_topics()
    await frequent_cache.set_trending_topics(topics)
    
    return topics

# Cache analytics summary
async def get_analytics_summary(time_range: str = "24h") -> Dict[str, Any]:
    """Get analytics summary with caching."""
    cached_summary = await frequent_cache.get_analytics_summary(time_range)
    if cached_summary:
        return cached_summary
    
    summary = await analytics_service.get_summary(time_range)
    await frequent_cache.set_analytics_summary(time_range, summary)
    
    return summary
```

### Analytics Data Cache

#### **User and Video Analytics**
```python
from api.caching.specialized_caches import AnalyticsDataCache

analytics_cache = AnalyticsDataCache(cache_manager)

# Cache user analytics
async def get_user_analytics(user_id: str, time_range: str = "7d") -> Dict[str, Any]:
    """Get user analytics with caching."""
    cached_analytics = await analytics_cache.get_user_analytics(user_id, time_range)
    if cached_analytics:
        return cached_analytics
    
    analytics = await analytics_service.get_user_analytics(user_id, time_range)
    await analytics_cache.set_user_analytics(user_id, analytics, time_range)
    
    return analytics

# Cache video analytics
async def get_video_analytics(video_id: str) -> Dict[str, Any]:
    """Get video analytics with caching."""
    cached_analytics = await analytics_cache.get_video_analytics(video_id)
    if cached_analytics:
        return cached_analytics
    
    analytics = await analytics_service.get_video_analytics(video_id)
    await analytics_cache.set_video_analytics(video_id, analytics)
    
    return analytics

# Cache platform analytics
async def get_platform_analytics(metric: str, time_range: str = "24h") -> Dict[str, Any]:
    """Get platform analytics with caching."""
    cached_analytics = await analytics_cache.get_platform_analytics(metric, time_range)
    if cached_analytics:
        return cached_analytics
    
    analytics = await analytics_service.get_platform_analytics(metric, time_range)
    await analytics_cache.set_platform_analytics(metric, analytics, time_range)
    
    return analytics
```

#### **Performance Metrics**
```python
# Cache performance metrics
async def get_performance_metrics(component: str) -> Dict[str, Any]:
    """Get performance metrics with caching."""
    cached_metrics = await analytics_cache.get_performance_metrics(component)
    if cached_metrics:
        return cached_metrics
    
    metrics = await monitoring_service.get_performance_metrics(component)
    await analytics_cache.set_performance_metrics(component, metrics)
    
    return metrics
```

## 🎯 Cache Decorators

### Basic Cache Decorators

#### **User Data Caching**
```python
from api.caching.specialized_caches import cache_user_data, invalidate_user_cache

@cache_user_data(ttl=1800)
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile with automatic caching."""
    return await user_service.get_profile(user_id)

@invalidate_user_cache()
async def update_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """Update user profile and invalidate cache."""
    await user_service.update_profile(user_id, profile_data)
    return {"message": "Profile updated successfully"}
```

#### **Video Data Caching**
```python
from api.caching.specialized_caches import cache_video_data, invalidate_video_cache

@cache_video_data(ttl=3600)
async def get_video_metadata(video_id: str) -> Dict[str, Any]:
    """Get video metadata with automatic caching."""
    return await video_service.get_metadata(video_id)

@invalidate_video_cache()
async def update_video_metadata(video_id: str, metadata: Dict[str, Any]):
    """Update video metadata and invalidate cache."""
    await video_service.update_metadata(video_id, metadata)
    return {"message": "Metadata updated successfully"}
```

#### **Static Content Caching**
```python
from api.caching.specialized_caches import cache_static_content

@cache_static_content(ttl=86400)
async def get_template_data(template_id: str) -> Dict[str, Any]:
    """Get template data with automatic caching."""
    return await template_service.get_template(template_id)

@cache_static_content(ttl=3600)
async def get_configuration(config_key: str) -> Dict[str, Any]:
    """Get configuration with automatic caching."""
    return await config_service.get_config(config_key)
```

### Advanced Cache Decorators

#### **Custom Cache Keys**
```python
from api.caching.cache_manager import cache_result

def generate_user_videos_key(user_id: str, page: int, per_page: int) -> str:
    """Generate custom cache key for user videos."""
    return f"user_videos:{user_id}:{page}:{per_page}"

@cache_result(ttl=300, key_generator=generate_user_videos_key)
async def get_user_videos(user_id: str, page: int = 1, per_page: int = 20):
    """Get user videos with custom cache key."""
    return await video_service.get_user_videos(user_id, page, per_page)
```

#### **Conditional Caching**
```python
def should_cache_user_data(user_id: str, include_sensitive: bool = False) -> bool:
    """Determine if user data should be cached."""
    return not include_sensitive  # Don't cache sensitive data

@cache_result(ttl=1800, condition=should_cache_user_data)
async def get_user_data(user_id: str, include_sensitive: bool = False):
    """Get user data with conditional caching."""
    return await user_service.get_user_data(user_id, include_sensitive)
```

## 🔗 Integration Examples

### FastAPI Application Setup

#### **Main Application Configuration**
```python
from fastapi import FastAPI, Depends
from api.caching.cache_manager import CacheManager, CacheConfig, CacheType
from api.caching.specialized_caches import CacheFactory

app = FastAPI(title="HeyGen AI API")

# Initialize cache manager
@app.on_event("startup")
async def startup_event():
    cache_config = CacheConfig(
        cache_type=CacheType.HYBRID,
        strategy=CacheStrategy.LRU,
        max_size=1000,
        default_ttl=300,
        redis_url=os.getenv("REDIS_URL"),
        redis_connection_pool_size=10
    )
    
    cache_manager = CacheManager(cache_config)
    cache_factory = CacheFactory(cache_manager)
    
    app.state.cache_manager = cache_manager
    app.state.cache_factory = cache_factory

# Dependency injection
def get_cache_manager() -> CacheManager:
    return app.state.cache_manager

def get_cache_factory() -> CacheFactory:
    return app.state.cache_factory
```

#### **Optimized Endpoints**
```python
from api.caching.specialized_caches import (
    cache_user_data, cache_video_data, cache_static_content
)

@router.get("/users/{user_id}")
@cache_user_data(ttl=1800)
async def get_user(
    user_id: str,
    cache_factory: CacheFactory = Depends(get_cache_factory)
):
    """Get user with caching."""
    user_cache = cache_factory.get_user_cache()
    
    # Try cache first
    cached_user = await user_cache.get_user_profile(user_id)
    if cached_user:
        return cached_user
    
    # Fetch from database
    user = await user_service.get_user(user_id)
    
    # Cache result
    await user_cache.set_user_profile(user_id, user)
    
    return user

@router.get("/videos/{video_id}")
@cache_video_data(ttl=3600)
async def get_video(
    video_id: str,
    cache_factory: CacheFactory = Depends(get_cache_factory)
):
    """Get video with caching."""
    video_cache = cache_factory.get_video_cache()
    
    # Try cache first
    cached_video = await video_cache.get_video_metadata(video_id)
    if cached_video:
        return cached_video
    
    # Fetch from database
    video = await video_service.get_video(video_id)
    
    # Cache result
    await video_cache.set_video_metadata(video_id, video)
    
    return video

@router.get("/templates/{template_id}")
@cache_static_content(ttl=86400)
async def get_template(
    template_id: str,
    cache_factory: CacheFactory = Depends(get_cache_factory)
):
    """Get template with caching."""
    static_cache = cache_factory.get_static_cache()
    
    # Try cache first
    cached_template = await static_cache.get_template_data(template_id)
    if cached_template:
        return cached_template
    
    # Fetch from database
    template = await template_service.get_template(template_id)
    
    # Cache result
    await static_cache.set_template_data(template_id, template)
    
    return template
```

### Service Layer Integration

#### **Optimized User Service**
```python
class CachedUserService:
    def __init__(self, cache_factory: CacheFactory):
        self.cache_factory = cache_factory
        self.user_cache = cache_factory.get_user_cache()
    
    @cache_user_data(ttl=1800)
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with caching."""
        # Try cache first
        cached_profile = await self.user_cache.get_user_profile(user_id)
        if cached_profile:
            return cached_profile
        
        # Fetch from database
        profile = await self.user_service.get_profile(user_id)
        
        # Cache result
        await self.user_cache.set_user_profile(user_id, profile)
        
        return profile
    
    @invalidate_user_cache()
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Update user profile and invalidate cache."""
        # Update in database
        await self.user_service.update_profile(user_id, profile_data)
        
        # Invalidate cache
        await self.user_cache.invalidate_user_cache(user_id)
        
        return {"message": "Profile updated successfully"}
    
    async def get_user_videos(self, user_id: str, page: int = 1) -> Dict[str, Any]:
        """Get user videos with caching."""
        # Try cache first
        cached_videos = await self.user_cache.get_user_videos(user_id, page)
        if cached_videos:
            return cached_videos
        
        # Fetch from database
        videos = await self.video_service.get_user_videos(user_id, page)
        
        # Cache result
        await self.user_cache.set_user_videos(user_id, videos, page)
        
        return videos
```

#### **Optimized Video Service**
```python
class CachedVideoService:
    def __init__(self, cache_factory: CacheFactory):
        self.cache_factory = cache_factory
        self.video_cache = cache_factory.get_video_cache()
    
    @cache_video_data(ttl=3600)
    async def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get video metadata with caching."""
        # Try cache first
        cached_metadata = await self.video_cache.get_video_metadata(video_id)
        if cached_metadata:
            return cached_metadata
        
        # Fetch from database
        metadata = await self.video_service.get_metadata(video_id)
        
        # Cache result
        await self.video_cache.set_video_metadata(video_id, metadata)
        
        return metadata
    
    @invalidate_video_cache()
    async def update_video_metadata(self, video_id: str, metadata: Dict[str, Any]):
        """Update video metadata and invalidate cache."""
        # Update in database
        await self.video_service.update_metadata(video_id, metadata)
        
        # Invalidate cache
        await self.video_cache.invalidate_video_cache(video_id)
        
        return {"message": "Metadata updated successfully"}
    
    async def get_video_status(self, video_id: str) -> Dict[str, Any]:
        """Get video status with caching."""
        # Try cache first
        cached_status = await self.video_cache.get_video_status(video_id)
        if cached_status:
            return cached_status
        
        # Fetch from external API
        status = await self.heygen_api.get_video_status(video_id)
        
        # Cache result (shorter TTL for status)
        await self.video_cache.set_video_status(video_id, status)
        
        return status
```

## 📊 Performance Monitoring

### Cache Statistics

#### **Cache Performance Metrics**
```python
async def get_cache_performance_stats() -> Dict[str, Any]:
    """Get comprehensive cache performance statistics."""
    cache_manager = get_cache_manager()
    stats = cache_manager.get_stats()
    
    return {
        "overall_hit_rate": stats["hit_rate"],
        "total_operations": stats["hits"] + stats["misses"],
        "cache_breakdown": stats["caches"],
        "performance_metrics": {
            "memory_usage_mb": stats.get("memory_usage_mb", 0),
            "redis_operations": stats.get("redis_operations", 0),
            "evictions": stats.get("evictions", 0)
        }
    }

# Example output:
{
    "overall_hit_rate": 0.85,
    "total_operations": 1250,
    "cache_breakdown": {
        "user_cache": {
            "hit_rate": 0.92,
            "hits": 450,
            "misses": 40
        },
        "video_cache": {
            "hit_rate": 0.78,
            "hits": 320,
            "misses": 90
        }
    },
    "performance_metrics": {
        "memory_usage_mb": 45.2,
        "redis_operations": 1250,
        "evictions": 12
    }
}
```

#### **Cache Health Monitoring**
```python
async def monitor_cache_health() -> Dict[str, str]:
    """Monitor health of all cache components."""
    cache_manager = get_cache_manager()
    
    health_status = {}
    
    # Check memory cache
    try:
        memory_stats = cache_manager.caches.get("memory", {}).get_stats()
        if memory_stats["hit_rate"] > 0.8:
            health_status["memory_cache"] = "healthy"
        else:
            health_status["memory_cache"] = "low_hit_rate"
    except Exception as e:
        health_status["memory_cache"] = f"error: {str(e)}"
    
    # Check Redis cache
    try:
        redis_stats = cache_manager.caches.get("redis", {}).get_stats()
        if redis_stats["hit_rate"] > 0.7:
            health_status["redis_cache"] = "healthy"
        else:
            health_status["redis_cache"] = "low_hit_rate"
    except Exception as e:
        health_status["redis_cache"] = f"error: {str(e)}"
    
    return health_status
```

### Cache Performance Alerts

#### **Performance Monitoring**
```python
async def check_cache_performance():
    """Check cache performance and alert on issues."""
    stats = await get_cache_performance_stats()
    
    # Alert on low hit rate
    if stats["overall_hit_rate"] < 0.8:
        logger.warning(f"Low cache hit rate: {stats['overall_hit_rate']:.2%}")
    
    # Alert on high memory usage
    memory_usage = stats["performance_metrics"]["memory_usage_mb"]
    if memory_usage > 100:  # 100MB threshold
        logger.warning(f"High memory usage: {memory_usage:.2f}MB")
    
    # Alert on high eviction rate
    evictions = stats["performance_metrics"]["evictions"]
    total_ops = stats["total_operations"]
    if total_ops > 0 and (evictions / total_ops) > 0.1:
        logger.warning(f"High eviction rate: {evictions/total_ops:.2%}")
```

## 🏆 Best Practices

### 1. Cache Key Design

#### **✅ Good: Descriptive Cache Keys**
```python
# Clear, descriptive cache keys
cache_keys = {
    "user_profile": f"user:{user_id}:profile",
    "user_videos": f"user:{user_id}:videos:{page}:{per_page}",
    "video_metadata": f"video:{video_id}:metadata",
    "video_status": f"video:{video_id}:status",
    "search_results": f"search:{query_hash}:filters:{filter_hash}"
}

# ❌ Bad: Generic cache keys
cache_keys = {
    "user_data": f"data:{user_id}",
    "video_data": f"data:{video_id}"
}
```

### 2. TTL Strategy

#### **Different TTL for Different Data Types**
```python
# TTL configuration for different data types
CACHE_TTL = {
    "user_profile": 1800,      # 30 minutes
    "user_videos": 300,        # 5 minutes
    "video_metadata": 3600,    # 1 hour
    "video_status": 300,       # 5 minutes (frequently updated)
    "static_content": 86400,   # 24 hours
    "analytics": 1800,         # 30 minutes
    "search_results": 600,     # 10 minutes
    "session_data": 86400      # 24 hours
}

# Use appropriate TTL
await cache_manager.set("user:123:profile", user_data, CACHE_TTL["user_profile"])
await cache_manager.set("video:456:status", status_data, CACHE_TTL["video_status"])
```

### 3. Cache Invalidation

#### **Proper Cache Invalidation**
```python
async def update_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """Update user profile with proper cache invalidation."""
    # Update in database
    await user_service.update_profile(user_id, profile_data)
    
    # Invalidate related cache entries
    await user_cache.invalidate_user_cache(user_id)
    
    # Clear search results that might include this user
    await frequent_cache.clear_search_cache()
    
    return {"message": "Profile updated successfully"}

async def update_video_metadata(video_id: str, metadata: Dict[str, Any]):
    """Update video metadata with proper cache invalidation."""
    # Update in database
    await video_service.update_metadata(video_id, metadata)
    
    # Invalidate video cache
    await video_cache.invalidate_video_cache(video_id)
    
    # Clear related caches
    await frequent_cache.clear_popular_videos_cache()
    await analytics_cache.clear_video_analytics_cache(video_id)
    
    return {"message": "Metadata updated successfully"}
```

### 4. Error Handling

#### **Graceful Cache Failures**
```python
async def get_user_profile_with_fallback(user_id: str) -> Dict[str, Any]:
    """Get user profile with cache fallback."""
    try:
        # Try cache first
        cached_profile = await user_cache.get_user_profile(user_id)
        if cached_profile:
            return cached_profile
    except Exception as e:
        logger.warning(f"Cache error, falling back to database: {e}")
    
    try:
        # Fallback to database
        profile = await user_service.get_profile(user_id)
        
        # Try to cache result (don't fail if cache is down)
        try:
            await user_cache.set_user_profile(user_id, profile)
        except Exception as e:
            logger.warning(f"Failed to cache user profile: {e}")
        
        return profile
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
```

### 5. Memory Management

#### **Memory-Efficient Caching**
```python
# Configure memory limits
memory_cache_config = CacheConfig(
    cache_type=CacheType.MEMORY,
    max_size=1000,
    memory_limit_mb=100,
    enable_eviction=True,
    eviction_policy="lru"
)

# Monitor memory usage
async def monitor_memory_usage():
    """Monitor cache memory usage."""
    stats = cache_manager.get_stats()
    memory_usage = stats.get("memory_usage_mb", 0)
    
    if memory_usage > 80:  # 80% of limit
        logger.warning(f"High memory usage: {memory_usage:.2f}MB")
        
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Clear least used items
        if memory_usage > 90:
            logger.warning("Clearing least used cache items")
            await cache_manager.clear_least_used()
```

## 🗑️ Cache Invalidation

### Invalidation Strategies

#### **Time-Based Invalidation**
```python
# Automatic TTL-based invalidation
await cache_manager.set("user:123:profile", user_data, ttl=1800)  # 30 minutes

# Manual invalidation after updates
async def update_user_profile(user_id: str, profile_data: Dict[str, Any]):
    await user_service.update_profile(user_id, profile_data)
    await cache_manager.delete(f"user:{user_id}:profile")
```

#### **Pattern-Based Invalidation**
```python
async def invalidate_user_related_cache(user_id: str):
    """Invalidate all user-related cache entries."""
    patterns = [
        f"user:{user_id}:profile",
        f"user:{user_id}:videos:*",
        f"user:{user_id}:analytics",
        f"search:*user:{user_id}*"
    ]
    
    for pattern in patterns:
        await cache_manager.clear_pattern(pattern)
```

#### **Event-Based Invalidation**
```python
async def handle_user_update_event(user_id: str, event_type: str):
    """Handle user update events and invalidate cache accordingly."""
    if event_type == "profile_update":
        await user_cache.invalidate_user_cache(user_id)
    elif event_type == "video_created":
        await user_cache.invalidate_user_cache(user_id)
        await frequent_cache.clear_popular_videos_cache()
    elif event_type == "video_deleted":
        await user_cache.invalidate_user_cache(user_id)
        await video_cache.invalidate_video_cache(user_id)
```

### Cache Warming

#### **Preload Frequently Accessed Data**
```python
async def warm_cache():
    """Warm up cache with frequently accessed data."""
    logger.info("Starting cache warming...")
    
    # Warm user profiles for active users
    active_users = await user_service.get_active_users(limit=100)
    for user in active_users:
        try:
            profile = await user_service.get_profile(user["id"])
            await user_cache.set_user_profile(user["id"], profile)
        except Exception as e:
            logger.warning(f"Failed to warm user cache for {user['id']}: {e}")
    
    # Warm popular videos
    popular_videos = await video_service.get_popular_videos(limit=50)
    await frequent_cache.set_popular_videos("all", popular_videos)
    
    # Warm system configuration
    configs = await config_service.get_all_configs()
    for config in configs:
        await static_cache.set_configuration(config["key"], config["value"])
    
    logger.info("Cache warming completed")

# Schedule cache warming
@app.on_event("startup")
async def startup_event():
    # Warm cache on startup
    await warm_cache()
    
    # Schedule periodic warming
    asyncio.create_task(periodic_cache_warming())

async def periodic_cache_warming():
    """Periodically warm cache."""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            await warm_cache()
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
```

## 📈 Expected Performance Improvements

### 1. Response Time Reduction
- **User Data**: 70-90% reduction for cached user profiles
- **Video Data**: 60-80% reduction for cached video metadata
- **Static Content**: 80-95% reduction for cached templates and configs
- **Search Results**: 50-70% reduction for cached search queries

### 2. Database Load Reduction
- **Query Reduction**: 60-80% reduction in database queries
- **Connection Pool**: Better connection pool utilization
- **Index Usage**: Reduced index pressure on frequently accessed data

### 3. Scalability Improvements
- **Concurrent Requests**: 3-5x increase in handling capacity
- **Memory Efficiency**: Better memory utilization with LRU/LFU strategies
- **Horizontal Scaling**: Better support for multiple application instances

### 4. User Experience
- **Faster Page Loads**: Reduced API response times
- **Consistent Performance**: Stable response times under load
- **Better Availability**: Graceful degradation when services are slow

## 🚀 Next Steps

1. **Implement the caching system** in your FastAPI application
2. **Configure cache policies** for different data types
3. **Add cache decorators** to existing endpoints
4. **Set up monitoring** for cache performance
5. **Implement cache invalidation** strategies
6. **Add cache warming** for frequently accessed data
7. **Monitor and optimize** based on real usage patterns

This comprehensive caching system provides your HeyGen AI API with:
- **Multi-level caching** for optimal performance
- **Specialized caches** for different data types
- **Automatic cache management** with decorators
- **Performance monitoring** and health checks
- **Graceful degradation** when cache is unavailable
- **Efficient memory management** with eviction policies

The system is designed to maximize performance while maintaining data consistency and providing excellent user experience across all components. 