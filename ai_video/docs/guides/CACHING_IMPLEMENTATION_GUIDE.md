# 🚀 CACHING IMPLEMENTATION GUIDE - STATIC & FREQUENTLY ACCESSED DATA

## Overview

This guide provides comprehensive strategies for implementing caching for static and frequently accessed data in the AI Video system using Redis and in-memory stores.

## Table of Contents

1. [Caching Strategy Overview](#caching-strategy-overview)
2. [Static Data Caching](#static-data-caching)
3. [Frequently Accessed Data Caching](#frequently-accessed-data-caching)
4. [Multi-Tier Caching Architecture](#multi-tier-caching-architecture)
5. [Cache Warming Strategies](#cache-warming-strategies)
6. [Predictive Caching](#predictive-caching)
7. [Cache Invalidation](#cache-invalidation)
8. [Performance Monitoring](#performance-monitoring)
9. [Best Practices](#best-practices)
10. [Implementation Examples](#implementation-examples)

## Caching Strategy Overview

### Types of Data to Cache

1. **Static Data** - Never or rarely changes
   - Configuration files
   - Model metadata
   - System constants
   - API documentation

2. **Frequently Accessed Data** - Accessed often but may change
   - User sessions
   - Video metadata
   - Processing results
   - API responses

3. **Computed Data** - Expensive to compute
   - Video thumbnails
   - Transcoding results
   - Analytics data
   - Search indexes

### Cache Tiers

```
┌─────────────────┐
│   Memory Cache  │ ← Fastest (L1)
│   (In-Memory)   │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Redis Cache   │ ← Fast (L2)
│   (Distributed) │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Database      │ ← Slowest (L3)
│   (Persistent)  │
└─────────────────┘
```

## Static Data Caching

### Characteristics
- **Lifetime**: Very long (days/weeks/months)
- **Size**: Small to medium
- **Access Pattern**: Read-only, infrequent
- **Consistency**: High (rarely changes)

### Implementation

```python
# Static Data Manager
class StaticDataManager:
    def __init__(self, memory_cache, redis_cache):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.static_keys = set()
    
    async def register_static_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """Register static data that should be cached permanently."""
        self.static_keys.add(key)
        
        # Store in both caches with long TTL
        await self.memory_cache.set(key, data, ttl or 86400)  # 24 hours
        await self.redis_cache.set(key, data, ttl or 604800)  # 7 days
    
    async def get_static_data(self, key: str) -> Optional[Any]:
        """Get static data with fallback strategy."""
        # Try memory cache first (fastest)
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache (fast)
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster future access
            await self.memory_cache.set(key, value)
            return value
        
        return None

# Usage Example
async def cache_static_config():
    static_manager = StaticDataManager(memory_cache, redis_cache)
    
    # Cache configuration data
    config_data = {
        "api_version": "2.0.0",
        "supported_formats": ["mp4", "avi", "mov"],
        "max_file_size": 1024 * 1024 * 100,  # 100MB
        "processing_timeout": 300
    }
    
    await static_manager.register_static_data("system_config", config_data)
    
    # Cache model metadata
    model_metadata = {
        "video_enhancement": {
            "version": "1.2.0",
            "path": "/models/enhancement_v1.2.0.pth",
            "input_size": [1920, 1080],
            "output_size": [1920, 1080]
        },
        "video_compression": {
            "version": "2.1.0",
            "path": "/models/compression_v2.1.0.pth",
            "compression_ratio": 0.8
        }
    }
    
    await static_manager.register_static_data("model_metadata", model_metadata)
```

### Static Data Categories

```python
# System Configuration
STATIC_CONFIG_KEYS = {
    "system_config": "system_configuration",
    "api_config": "api_configuration", 
    "security_config": "security_settings",
    "logging_config": "logging_configuration"
}

# Model Information
STATIC_MODEL_KEYS = {
    "model_metadata": "model_information",
    "model_paths": "model_file_paths",
    "model_versions": "model_version_info",
    "model_configs": "model_configurations"
}

# Feature Flags
STATIC_FEATURE_KEYS = {
    "feature_flags": "feature_toggles",
    "experiments": "experiment_configs",
    "permissions": "permission_matrix"
}

# Reference Data
STATIC_REFERENCE_KEYS = {
    "video_formats": "supported_video_formats",
    "audio_codecs": "supported_audio_codecs",
    "quality_presets": "quality_preset_configs"
}
```

## Frequently Accessed Data Caching

### Characteristics
- **Lifetime**: Short to medium (minutes to hours)
- **Size**: Small to large
- **Access Pattern**: Read/write, frequent
- **Consistency**: Medium (may change)

### Implementation

```python
# Frequently Accessed Data Manager
class FrequentDataManager:
    def __init__(self, memory_cache, redis_cache, predictive_cache):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.predictive_cache = predictive_cache
        self.frequent_keys = set()
    
    async def register_frequent_data(self, key: str, data: Any, ttl: int = 1800):
        """Register frequently accessed data."""
        self.frequent_keys.add(key)
        
        # Store in memory cache with shorter TTL for speed
        await self.memory_cache.set(key, data, ttl // 2)
        
        # Store in Redis cache with longer TTL for persistence
        await self.redis_cache.set(key, data, ttl)
    
    async def get_frequent_data(self, key: str, context: Optional[str] = None) -> Optional[Any]:
        """Get frequently accessed data with predictive caching."""
        # Record access pattern for prediction
        self.predictive_cache.record_access(key, context)
        
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster future access
            await self.memory_cache.set(key, value)
            
            # Predict and preload related data
            await self._predict_and_preload(key, context)
            
            return value
        
        return None
    
    async def _predict_and_preload(self, key: str, context: Optional[str] = None):
        """Predict and preload related data."""
        related_keys = self.predictive_cache.get_related_keys(key, context)
        
        for related_key in related_keys[:5]:  # Limit to 5 keys
            prediction = self.predictive_cache.predict_next_access(related_key, context)
            if prediction > 0.7:  # High probability threshold
                await self._preload_key(related_key)

# Usage Example
async def cache_frequent_data():
    frequent_manager = FrequentDataManager(memory_cache, redis_cache, predictive_cache)
    
    # Cache user session data
    user_session = {
        "user_id": "user123",
        "permissions": ["video_upload", "video_process"],
        "preferences": {"quality": "high", "format": "mp4"},
        "last_activity": time.time()
    }
    
    await frequent_manager.register_frequent_data("user_session:user123", user_session, 3600)
    
    # Cache video metadata
    video_metadata = {
        "video_id": "video456",
        "title": "Sample Video",
        "duration": 120,
        "format": "mp4",
        "size": 1024 * 1024 * 50,  # 50MB
        "status": "processed",
        "thumbnail_url": "https://example.com/thumb.jpg"
    }
    
    await frequent_manager.register_frequent_data("video_metadata:video456", video_metadata, 1800)
```

### Frequently Accessed Data Categories

```python
# User Data
FREQUENT_USER_KEYS = {
    "user_sessions": "user_session_data",
    "user_preferences": "user_preference_settings",
    "user_permissions": "user_permission_data",
    "user_activity": "user_activity_logs"
}

# Video Data
FREQUENT_VIDEO_KEYS = {
    "video_metadata": "video_metadata_cache",
    "video_status": "video_processing_status",
    "video_thumbnails": "video_thumbnail_cache",
    "video_analytics": "video_analytics_data"
}

# Processing Data
FREQUENT_PROCESSING_KEYS = {
    "processing_queue": "processing_queue_status",
    "processing_results": "processing_result_cache",
    "processing_configs": "processing_configuration_cache"
}

# API Data
FREQUENT_API_KEYS = {
    "api_responses": "cached_api_responses",
    "api_rates": "api_rate_limit_data",
    "api_tokens": "api_token_cache"
}
```

## Multi-Tier Caching Architecture

### Architecture Design

```python
class MultiTierCache:
    """Multi-tier caching system with intelligent fallback."""
    
    def __init__(self, memory_cache, redis_cache, database):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.database = database
        self.cache_stats = defaultdict(int)
    
    async def get(self, key: str, context: Optional[str] = None) -> Optional[Any]:
        """Get data from multi-tier cache with fallback."""
        # Tier 1: Memory Cache (Fastest)
        value = await self.memory_cache.get(key)
        if value is not None:
            self.cache_stats['memory_hits'] += 1
            return value
        
        # Tier 2: Redis Cache (Fast)
        value = await self.redis_cache.get(key)
        if value is not None:
            self.cache_stats['redis_hits'] += 1
            # Store in memory cache for faster future access
            await self.memory_cache.set(key, value)
            return value
        
        # Tier 3: Database (Slowest)
        value = await self.database.get(key)
        if value is not None:
            self.cache_stats['database_hits'] += 1
            # Store in both caches
            await self.redis_cache.set(key, value)
            await self.memory_cache.set(key, value)
            return value
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set data in all cache tiers."""
        # Set in all tiers
        await self.memory_cache.set(key, value, ttl)
        await self.redis_cache.set(key, value, ttl)
        
        # Optionally store in database for persistence
        if self._should_persist(key):
            await self.database.set(key, value)
    
    def _should_persist(self, key: str) -> bool:
        """Determine if data should be persisted to database."""
        return key.startswith(('static_', 'config_', 'user_'))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_requests = sum(self.cache_stats.values())
        
        return {
            "memory_hit_rate": self.cache_stats['memory_hits'] / total_requests if total_requests > 0 else 0,
            "redis_hit_rate": self.cache_stats['redis_hits'] / total_requests if total_requests > 0 else 0,
            "database_hit_rate": self.cache_stats['database_hits'] / total_requests if total_requests > 0 else 0,
            "miss_rate": self.cache_stats['misses'] / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests
        }
```

### Cache Tier Configuration

```python
# Memory Cache Configuration
MEMORY_CACHE_CONFIG = {
    "static_data": {
        "max_size": 500,
        "ttl": 86400,  # 24 hours
        "eviction_policy": "lru"
    },
    "frequent_data": {
        "max_size": 1000,
        "ttl": 1800,  # 30 minutes
        "eviction_policy": "lfu"
    },
    "temporary_data": {
        "max_size": 2000,
        "ttl": 300,  # 5 minutes
        "eviction_policy": "ttl"
    }
}

# Redis Cache Configuration
REDIS_CACHE_CONFIG = {
    "static_data": {
        "ttl": 604800,  # 7 days
        "compression": True
    },
    "frequent_data": {
        "ttl": 3600,  # 1 hour
        "compression": True
    },
    "temporary_data": {
        "ttl": 1800,  # 30 minutes
        "compression": False
    }
}
```

## Cache Warming Strategies

### Pre-Warming Static Data

```python
class CacheWarmer:
    """Warms up cache with frequently accessed data."""
    
    def __init__(self, static_manager, frequent_manager):
        self.static_manager = static_manager
        self.frequent_manager = frequent_manager
        self.warming_tasks = set()
    
    async def warm_static_cache(self):
        """Warm cache with static data."""
        static_data_sources = {
            "system_config": self._load_system_config,
            "model_metadata": self._load_model_metadata,
            "feature_flags": self._load_feature_flags,
            "api_config": self._load_api_config
        }
        
        for key, data_source in static_data_sources.items():
            try:
                data = await data_source()
                await self.static_manager.register_static_data(key, data)
                logger.info(f"Warmed static cache: {key}")
            except Exception as e:
                logger.error(f"Failed to warm static cache for {key}: {e}")
    
    async def warm_frequent_cache(self):
        """Warm cache with frequently accessed data."""
        frequent_data_sources = {
            "active_sessions": self._load_active_sessions,
            "popular_videos": self._load_popular_videos,
            "processing_queue": self._load_processing_queue
        }
        
        for key, data_source in frequent_data_sources.items():
            try:
                data = await data_source()
                await self.frequent_manager.register_frequent_data(key, data)
                logger.info(f"Warmed frequent cache: {key}")
            except Exception as e:
                logger.error(f"Failed to warm frequent cache for {key}: {e}")
    
    async def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        return {
            "api_version": "2.0.0",
            "max_file_size": 1024 * 1024 * 100,
            "supported_formats": ["mp4", "avi", "mov", "mkv"],
            "processing_timeout": 300
        }
    
    async def _load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata."""
        return {
            "video_enhancement": {
                "version": "1.2.0",
                "path": "/models/enhancement_v1.2.0.pth"
            },
            "video_compression": {
                "version": "2.1.0",
                "path": "/models/compression_v2.1.0.pth"
            }
        }
    
    async def _load_active_sessions(self) -> List[Dict[str, Any]]:
        """Load active user sessions."""
        # This would query the database for active sessions
        return [
            {"user_id": "user1", "last_activity": time.time()},
            {"user_id": "user2", "last_activity": time.time()}
        ]
```

### Predictive Warming

```python
class PredictiveWarmer:
    """Warms cache based on access patterns."""
    
    def __init__(self, predictive_cache, frequent_manager):
        self.predictive_cache = predictive_cache
        self.frequent_manager = frequent_manager
    
    async def warm_based_on_patterns(self, current_key: str, context: Optional[str] = None):
        """Warm cache based on current access patterns."""
        # Get related keys that are likely to be accessed
        related_keys = self.predictive_cache.get_related_keys(current_key, context)
        
        for related_key in related_keys:
            prediction = self.predictive_cache.predict_next_access(related_key, context)
            
            if prediction > 0.8:  # Very high probability
                await self._warm_key(related_key, "high_priority")
            elif prediction > 0.6:  # High probability
                await self._warm_key(related_key, "medium_priority")
    
    async def _warm_key(self, key: str, priority: str):
        """Warm a specific key."""
        try:
            # Load data from database or external source
            data = await self._load_data_for_key(key)
            
            if data is not None:
                # Set TTL based on priority
                ttl = 3600 if priority == "high_priority" else 1800
                await self.frequent_manager.register_frequent_data(key, data, ttl)
                
                logger.info(f"Predictively warmed key: {key} (priority: {priority})")
        except Exception as e:
            logger.error(f"Failed to warm key {key}: {e}")
    
    async def _load_data_for_key(self, key: str) -> Optional[Any]:
        """Load data for a specific key."""
        # This would implement the actual data loading logic
        # based on the key pattern
        if key.startswith("user_session:"):
            return await self._load_user_session(key.split(":")[1])
        elif key.startswith("video_metadata:"):
            return await self._load_video_metadata(key.split(":")[1])
        
        return None
```

## Predictive Caching

### Access Pattern Analysis

```python
class AccessPatternAnalyzer:
    """Analyzes access patterns for predictive caching."""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.correlation_matrix = defaultdict(dict)
    
    def record_access(self, key: str, context: Optional[str] = None, timestamp: Optional[float] = None):
        """Record an access to a key."""
        pattern_key = f"{context}:{key}" if context else key
        timestamp = timestamp or time.time()
        
        self.access_patterns[pattern_key].append(timestamp)
        
        # Keep only recent accesses (last 1000)
        if len(self.access_patterns[pattern_key]) > 1000:
            self.access_patterns[pattern_key] = self.access_patterns[pattern_key][-1000:]
    
    def analyze_patterns(self):
        """Analyze access patterns and build correlation matrix."""
        keys = list(self.access_patterns.keys())
        
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys[i+1:], i+1):
                correlation = self._calculate_correlation(
                    self.access_patterns[key1],
                    self.access_patterns[key2]
                )
                
                if correlation > 0.5:  # Significant correlation
                    self.correlation_matrix[key1][key2] = correlation
                    self.correlation_matrix[key2][key1] = correlation
    
    def get_related_keys(self, key: str, context: Optional[str] = None) -> List[tuple]:
        """Get keys that are correlated with the given key."""
        pattern_key = f"{context}:{key}" if context else key
        
        if pattern_key not in self.correlation_matrix:
            return []
        
        # Return keys sorted by correlation strength
        related = self.correlation_matrix[pattern_key].items()
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def predict_next_access(self, key: str, context: Optional[str] = None) -> float:
        """Predict likelihood of next access."""
        pattern_key = f"{context}:{key}" if context else key
        
        if pattern_key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[pattern_key]
        if len(accesses) < 2:
            return 0.5
        
        # Calculate time intervals between accesses
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        
        if not intervals:
            return 0.0
        
        # Calculate average interval
        avg_interval = statistics.mean(intervals)
        
        # Calculate time since last access
        time_since_last = time.time() - accesses[-1]
        
        # Predict likelihood based on time since last access vs average interval
        if avg_interval > 0:
            likelihood = max(0.0, 1.0 - (time_since_last / avg_interval))
            return min(1.0, likelihood)
        
        return 0.0
    
    def _calculate_correlation(self, accesses1: List[float], accesses2: List[float]) -> float:
        """Calculate correlation between two access patterns."""
        if len(accesses1) < 2 or len(accesses2) < 2:
            return 0.0
        
        try:
            return statistics.correlation(accesses1, accesses2)
        except:
            return 0.0
```

## Cache Invalidation

### Invalidation Strategies

```python
class CacheInvalidator:
    """Handles cache invalidation strategies."""
    
    def __init__(self, memory_cache, redis_cache):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.invalidation_patterns = {}
    
    async def invalidate_by_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        # Invalidate from memory cache
        await self.memory_cache.invalidate(pattern)
        
        # Invalidate from Redis cache
        await self.redis_cache.invalidate(pattern)
        
        logger.info(f"Invalidated cache entries matching pattern: {pattern}")
    
    async def invalidate_static_data(self, key: str):
        """Invalidate static data."""
        await self.memory_cache.delete(key)
        await self.redis_cache.delete(key)
        
        logger.info(f"Invalidated static data: {key}")
    
    async def invalidate_frequent_data(self, key: str):
        """Invalidate frequent data."""
        await self.memory_cache.delete(key)
        await self.redis_cache.delete(key)
        
        logger.info(f"Invalidated frequent data: {key}")
    
    async def invalidate_user_data(self, user_id: str):
        """Invalidate all data related to a user."""
        patterns = [
            f"user_session:{user_id}",
            f"user_preferences:{user_id}",
            f"user_activity:{user_id}*"
        ]
        
        for pattern in patterns:
            await self.invalidate_by_pattern(pattern)
        
        logger.info(f"Invalidated all data for user: {user_id}")
    
    async def invalidate_video_data(self, video_id: str):
        """Invalidate all data related to a video."""
        patterns = [
            f"video_metadata:{video_id}",
            f"video_status:{video_id}",
            f"video_thumbnail:{video_id}",
            f"video_analytics:{video_id}"
        ]
        
        for pattern in patterns:
            await self.invalidate_by_pattern(pattern)
        
        logger.info(f"Invalidated all data for video: {video_id}")
    
    def register_invalidation_pattern(self, event: str, pattern: str):
        """Register invalidation pattern for specific events."""
        self.invalidation_patterns[event] = pattern
    
    async def handle_event(self, event: str, **kwargs):
        """Handle events and trigger cache invalidation."""
        if event in self.invalidation_patterns:
            pattern = self.invalidation_patterns[event]
            
            # Replace placeholders in pattern
            for key, value in kwargs.items():
                pattern = pattern.replace(f"{{{key}}}", str(value))
            
            await self.invalidate_by_pattern(pattern)
```

### Event-Driven Invalidation

```python
# Register invalidation patterns
cache_invalidator = CacheInvalidator(memory_cache, redis_cache)

# User-related events
cache_invalidator.register_invalidation_pattern("user_login", "user_session:{user_id}")
cache_invalidator.register_invalidation_pattern("user_logout", "user_session:{user_id}")
cache_invalidator.register_invalidation_pattern("user_update", "user_preferences:{user_id}")

# Video-related events
cache_invalidator.register_invalidation_pattern("video_upload", "video_metadata:{video_id}")
cache_invalidator.register_invalidation_pattern("video_process", "video_status:{video_id}")
cache_invalidator.register_invalidation_pattern("video_delete", "video_*:{video_id}")

# System events
cache_invalidator.register_invalidation_pattern("config_update", "system_config")
cache_invalidator.register_invalidation_pattern("model_update", "model_metadata")

# Usage
await cache_invalidator.handle_event("user_login", user_id="user123")
await cache_invalidator.handle_event("video_upload", video_id="video456")
```

## Performance Monitoring

### Cache Performance Metrics

```python
class CachePerformanceMonitor:
    """Monitors cache performance and provides insights."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a performance metric."""
        timestamp = timestamp or time.time()
        self.metrics[metric_name].append((timestamp, value))
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def get_hit_rate(self, cache_name: str) -> float:
        """Get hit rate for a specific cache."""
        hits = self.metrics[f"{cache_name}_hits"]
        misses = self.metrics[f"{cache_name}_misses"]
        
        total_hits = sum(value for _, value in hits)
        total_misses = sum(value for _, value in misses)
        
        total_requests = total_hits + total_misses
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def get_average_response_time(self, cache_name: str) -> float:
        """Get average response time for a specific cache."""
        response_times = self.metrics[f"{cache_name}_response_time"]
        
        if not response_times:
            return 0.0
        
        total_time = sum(value for _, value in response_times)
        return total_time / len(response_times)
    
    def get_cache_size(self, cache_name: str) -> int:
        """Get current cache size."""
        sizes = self.metrics[f"{cache_name}_size"]
        return sizes[-1][1] if sizes else 0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "caches": {}
        }
        
        cache_names = ["memory", "redis", "database"]
        
        for cache_name in cache_names:
            report["caches"][cache_name] = {
                "hit_rate": self.get_hit_rate(cache_name),
                "avg_response_time": self.get_average_response_time(cache_name),
                "current_size": self.get_cache_size(cache_name),
                "total_requests": sum(value for _, value in self.metrics[f"{cache_name}_hits"]) +
                                sum(value for _, value in self.metrics[f"{cache_name}_misses"])
            }
        
        return report
    
    def check_alerts(self) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        # Check hit rates
        for cache_name in ["memory", "redis"]:
            hit_rate = self.get_hit_rate(cache_name)
            if hit_rate < 0.8:  # Less than 80% hit rate
                alerts.append(f"Low hit rate for {cache_name} cache: {hit_rate:.2%}")
        
        # Check response times
        for cache_name in ["memory", "redis"]:
            avg_response_time = self.get_average_response_time(cache_name)
            if avg_response_time > 0.1:  # More than 100ms
                alerts.append(f"High response time for {cache_name} cache: {avg_response_time:.3f}s")
        
        return alerts
```

## Best Practices

### 1. Cache Key Design

```python
# Good cache key patterns
GOOD_CACHE_KEYS = {
    "static": "static:system_config",
    "user": "user:session:{user_id}",
    "video": "video:metadata:{video_id}",
    "processing": "processing:status:{job_id}",
    "api": "api:response:{endpoint}:{params_hash}"
}

# Bad cache key patterns
BAD_CACHE_KEYS = {
    "too_long": "very_long_cache_key_that_is_hard_to_read_and_manage",
    "no_structure": "randomkey123",
    "inconsistent": "user_session_123",  # Inconsistent naming
    "sensitive": "user:password:123"  # Contains sensitive data
}
```

### 2. TTL Strategy

```python
# TTL Configuration
TTL_STRATEGY = {
    "static_data": {
        "memory": 86400,    # 24 hours
        "redis": 604800     # 7 days
    },
    "frequent_data": {
        "memory": 1800,     # 30 minutes
        "redis": 3600       # 1 hour
    },
    "temporary_data": {
        "memory": 300,      # 5 minutes
        "redis": 1800       # 30 minutes
    },
    "session_data": {
        "memory": 3600,     # 1 hour
        "redis": 7200       # 2 hours
    }
}
```

### 3. Memory Management

```python
class CacheMemoryManager:
    """Manages memory usage for caches."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        return self.current_memory_mb < self.max_memory_mb
    
    def estimate_memory_usage(self, data: Any) -> int:
        """Estimate memory usage for data in MB."""
        try:
            size_bytes = len(pickle.dumps(data))
            return size_bytes / (1024 * 1024)
        except:
            return 1  # Default estimate
    
    def can_store(self, data: Any) -> bool:
        """Check if data can be stored without exceeding memory limit."""
        estimated_size = self.estimate_memory_usage(data)
        return (self.current_memory_mb + estimated_size) <= self.max_memory_mb
```

### 4. Error Handling

```python
class CacheErrorHandler:
    """Handles cache errors gracefully."""
    
    @staticmethod
    async def safe_cache_operation(operation: Callable, fallback: Optional[Callable] = None):
        """Execute cache operation with error handling."""
        try:
            return await operation()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            if fallback:
                return await fallback()
            return None
        except Exception as e:
            logger.error(f"Cache operation error: {e}")
            if fallback:
                return await fallback()
            return None
    
    @staticmethod
    async def cache_with_fallback(key: str, cache_get: Callable, 
                                fallback_get: Callable, cache_set: Callable):
        """Get from cache with fallback to data source."""
        # Try cache first
        cached_value = await CacheErrorHandler.safe_cache_operation(
            lambda: cache_get(key)
        )
        
        if cached_value is not None:
            return cached_value
        
        # Fallback to data source
        value = await fallback_get(key)
        
        if value is not None:
            # Store in cache for future use
            await CacheErrorHandler.safe_cache_operation(
                lambda: cache_set(key, value)
            )
        
        return value
```

## Implementation Examples

### Complete Caching System Example

```python
async def implement_complete_caching_system():
    """Complete example of implementing caching for AI Video system."""
    
    # Initialize caching system
    caching_system = EnhancedCachingSystem("redis://localhost:6379")
    await caching_system.initialize()
    
    # Define data sources for cache warming
    data_sources = {
        "static_config": lambda: {
            "api_version": "2.0.0",
            "max_file_size": 1024 * 1024 * 100,
            "supported_formats": ["mp4", "avi", "mov", "mkv"],
            "processing_timeout": 300
        },
        "static_models": lambda: {
            "video_enhancement": {
                "version": "1.2.0",
                "path": "/models/enhancement_v1.2.0.pth"
            },
            "video_compression": {
                "version": "2.1.0",
                "path": "/models/compression_v2.1.0.pth"
            }
        },
        "frequent_sessions": lambda: [
            {"user_id": "user1", "last_activity": time.time()},
            {"user_id": "user2", "last_activity": time.time()}
        ],
        "frequent_videos": lambda: [
            {"video_id": "video1", "title": "Sample Video 1"},
            {"video_id": "video2", "title": "Sample Video 2"}
        ]
    }
    
    # Warm cache
    await caching_system.warm_cache(data_sources)
    
    # Use cached data
    config = await caching_system.get_static_data("static_config")
    print(f"System config: {config}")
    
    session_data = await caching_system.get_frequent_data("frequent_sessions", "user_context")
    print(f"Session data: {session_data}")
    
    # Get performance statistics
    stats = await caching_system.get_cache_stats()
    print(f"Cache performance: {stats}")
    
    # Cleanup
    await caching_system.cleanup()

# Run the example
if __name__ == "__main__":
    asyncio.run(implement_complete_caching_system())
```

This comprehensive caching implementation provides:

1. **Multi-tier caching** with memory and Redis
2. **Static data caching** for configuration and metadata
3. **Frequent data caching** with predictive preloading
4. **Cache warming** strategies for optimal performance
5. **Performance monitoring** and alerting
6. **Intelligent invalidation** based on events
7. **Error handling** and fallback mechanisms

The system ensures that static and frequently accessed data is always available with minimal latency while maintaining data consistency and optimal resource usage. 