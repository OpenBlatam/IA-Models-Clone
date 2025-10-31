from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from .cache_manager import CacheManager, CacheConfig, CacheType, CacheStrategy
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Specialized Caches for HeyGen AI API
Specialized cache implementations for different data types.
"""



logger = structlog.get_logger()

# =============================================================================
# Specialized Cache Types
# =============================================================================

class DataType(Enum):
    """Data type enumeration."""
    USER_DATA = "user_data"
    VIDEO_DATA = "video_data"
    STATIC_CONTENT = "static_content"
    FREQUENTLY_ACCESSED = "frequently_accessed"
    SESSION_DATA = "session_data"
    ANALYTICS_DATA = "analytics_data"
    CONFIGURATION = "configuration"
    TEMPLATE_DATA = "template_data"

@dataclass
class CachePolicy:
    """Cache policy configuration."""
    data_type: DataType
    ttl: int
    max_size: int
    strategy: CacheStrategy
    priority: str
    compression_enabled: bool = True
    auto_refresh: bool = False
    refresh_interval: Optional[int] = None
    invalidation_patterns: List[str] = None

# =============================================================================
# User Data Cache
# =============================================================================

class UserDataCache:
    """Specialized cache for user-related data."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.user_cache_config = CachePolicy(
            data_type=DataType.USER_DATA,
            ttl=1800,  # 30 minutes
            max_size=1000,
            strategy=CacheStrategy.LRU,
            priority="high",
            compression_enabled=True,
            auto_refresh=True,
            refresh_interval=900  # 15 minutes
        )
    
    def _generate_user_key(self, user_id: str, data_type: str = "profile") -> str:
        """Generate cache key for user data."""
        return f"user:{user_id}:{data_type}"
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from cache."""
        key = self._generate_user_key(user_id, "profile")
        return await self.cache_manager.get(key, "user_cache")
    
    async def set_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Set user profile in cache."""
        key = self._generate_user_key(user_id, "profile")
        await self.cache_manager.set(key, profile_data, self.user_cache_config.ttl, "user_cache")
    
    async def get_user_videos(self, user_id: str, page: int = 1, per_page: int = 20) -> Optional[Dict[str, Any]]:
        """Get user videos from cache."""
        key = self._generate_user_key(user_id, f"videos:{page}:{per_page}")
        return await self.cache_manager.get(key, "user_cache")
    
    async def set_user_videos(self, user_id: str, videos_data: Dict[str, Any], page: int = 1, per_page: int = 20):
        """Set user videos in cache."""
        key = self._generate_user_key(user_id, f"videos:{page}:{per_page}")
        await self.cache_manager.set(key, videos_data, self.user_cache_config.ttl, "user_cache")
    
    async def get_user_analytics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user analytics from cache."""
        key = self._generate_user_key(user_id, "analytics")
        return await self.cache_manager.get(key, "user_cache")
    
    async def set_user_analytics(self, user_id: str, analytics_data: Dict[str, Any]):
        """Set user analytics in cache."""
        key = self._generate_user_key(user_id, "analytics")
        await self.cache_manager.set(key, analytics_data, self.user_cache_config.ttl, "user_cache")
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all user-related cache entries."""
        patterns = [
            f"user:{user_id}:profile",
            f"user:{user_id}:videos:*",
            f"user:{user_id}:analytics"
        ]
        
        for pattern in patterns:
            await self.cache_manager.clear("user_cache")
    
    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user session from cache."""
        key = f"session:{session_id}"
        return await self.cache_manager.get(key, "session_cache")
    
    async def set_user_session(self, session_id: str, session_data: Dict[str, Any], ttl: int = 86400):
        """Set user session in cache."""
        key = f"session:{session_id}"
        await self.cache_manager.set(key, session_data, ttl, "session_cache")
    
    async def delete_user_session(self, session_id: str) -> bool:
        """Delete user session from cache."""
        key = f"session:{session_id}"
        return await self.cache_manager.delete(key, "session_cache")

# =============================================================================
# Video Data Cache
# =============================================================================

class VideoDataCache:
    """Specialized cache for video-related data."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.video_cache_config = CachePolicy(
            data_type=DataType.VIDEO_DATA,
            ttl=3600,  # 1 hour
            max_size=500,
            strategy=CacheStrategy.LRU,
            priority="high",
            compression_enabled=True,
            auto_refresh=False
        )
    
    def _generate_video_key(self, video_id: str, data_type: str = "metadata") -> str:
        """Generate cache key for video data."""
        return f"video:{video_id}:{data_type}"
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video metadata from cache."""
        key = self._generate_video_key(video_id, "metadata")
        return await self.cache_manager.get(key, "video_cache")
    
    async def set_video_metadata(self, video_id: str, metadata: Dict[str, Any]):
        """Set video metadata in cache."""
        key = self._generate_video_key(video_id, "metadata")
        await self.cache_manager.set(key, metadata, self.video_cache_config.ttl, "video_cache")
    
    async def get_video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video status from cache."""
        key = self._generate_video_key(video_id, "status")
        return await self.cache_manager.get(key, "video_cache")
    
    async def set_video_status(self, video_id: str, status: Dict[str, Any]):
        """Set video status in cache."""
        key = self._generate_video_key(video_id, "status")
        # Shorter TTL for status updates
        await self.cache_manager.set(key, status, 300, "video_cache")
    
    async def get_video_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video analytics from cache."""
        key = self._generate_video_key(video_id, "analytics")
        return await self.cache_manager.get(key, "video_cache")
    
    async def set_video_analytics(self, video_id: str, analytics: Dict[str, Any]):
        """Set video analytics in cache."""
        key = self._generate_video_key(video_id, "analytics")
        await self.cache_manager.set(key, analytics, self.video_cache_config.ttl, "video_cache")
    
    async def get_video_processing_queue(self, user_id: str) -> Optional[List[str]]:
        """Get user's video processing queue from cache."""
        key = f"processing_queue:{user_id}"
        return await self.cache_manager.get(key, "video_cache")
    
    async def set_video_processing_queue(self, user_id: str, queue: List[str]):
        """Set user's video processing queue in cache."""
        key = f"processing_queue:{user_id}"
        await self.cache_manager.set(key, queue, 1800, "video_cache")
    
    async def add_to_processing_queue(self, user_id: str, video_id: str):
        """Add video to processing queue."""
        queue = await self.get_video_processing_queue(user_id) or []
        if video_id not in queue:
            queue.append(video_id)
            await self.set_video_processing_queue(user_id, queue)
    
    async def remove_from_processing_queue(self, user_id: str, video_id: str):
        """Remove video from processing queue."""
        queue = await self.get_video_processing_queue(user_id) or []
        if video_id in queue:
            queue.remove(video_id)
            await self.set_video_processing_queue(user_id, queue)
    
    async def invalidate_video_cache(self, video_id: str):
        """Invalidate all video-related cache entries."""
        patterns = [
            f"video:{video_id}:metadata",
            f"video:{video_id}:status",
            f"video:{video_id}:analytics"
        ]
        
        for pattern in patterns:
            await self.cache_manager.clear("video_cache")

# =============================================================================
# Static Content Cache
# =============================================================================

class StaticContentCache:
    """Specialized cache for static content."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.static_cache_config = CachePolicy(
            data_type=DataType.STATIC_CONTENT,
            ttl=86400,  # 24 hours
            max_size=200,
            strategy=CacheStrategy.LRU,
            priority="normal",
            compression_enabled=True,
            auto_refresh=False
        )
    
    async def get_template_data(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template data from cache."""
        key = f"template:{template_id}"
        return await self.cache_manager.get(key, "static_cache")
    
    async def set_template_data(self, template_id: str, template_data: Dict[str, Any]):
        """Set template data in cache."""
        key = f"template:{template_id}"
        await self.cache_manager.set(key, template_data, self.static_cache_config.ttl, "static_cache")
    
    async def get_configuration(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get configuration from cache."""
        key = f"config:{config_key}"
        return await self.cache_manager.get(key, "static_cache")
    
    async def set_configuration(self, config_key: str, config_data: Dict[str, Any]):
        """Set configuration in cache."""
        key = f"config:{config_key}"
        await self.cache_manager.set(key, config_data, self.static_cache_config.ttl, "static_cache")
    
    async async def get_api_documentation(self, version: str = "v1") -> Optional[Dict[str, Any]]:
        """Get API documentation from cache."""
        key = f"api_docs:{version}"
        return await self.cache_manager.get(key, "static_cache")
    
    async def set_api_documentation(self, version: str, docs_data: Dict[str, Any]):
        """Set API documentation in cache."""
        key = f"api_docs:{version}"
        await self.cache_manager.set(key, docs_data, self.static_cache_config.ttl, "static_cache")
    
    async def get_system_status(self) -> Optional[Dict[str, Any]]:
        """Get system status from cache."""
        key = "system:status"
        return await self.cache_manager.get(key, "static_cache")
    
    async def set_system_status(self, status_data: Dict[str, Any]):
        """Set system status in cache."""
        key = "system:status"
        # Shorter TTL for system status
        await self.cache_manager.set(key, status_data, 300, "static_cache")
    
    async def get_rate_limits(self, user_id: str, action: str) -> Optional[Dict[str, Any]]:
        """Get rate limit information from cache."""
        key = f"rate_limit:{user_id}:{action}"
        return await self.cache_manager.get(key, "static_cache")
    
    async def set_rate_limits(self, user_id: str, action: str, limit_data: Dict[str, Any]):
        """Set rate limit information in cache."""
        key = f"rate_limit:{user_id}:{action}"
        await self.cache_manager.set(key, limit_data, 3600, "static_cache")

# =============================================================================
# Frequently Accessed Data Cache
# =============================================================================

class FrequentlyAccessedCache:
    """Specialized cache for frequently accessed data."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.frequent_cache_config = CachePolicy(
            data_type=DataType.FREQUENTLY_ACCESSED,
            ttl=600,  # 10 minutes
            max_size=2000,
            strategy=CacheStrategy.LFU,
            priority="high",
            compression_enabled=True,
            auto_refresh=True,
            refresh_interval=300  # 5 minutes
        )
    
    async def get_search_results(self, query: str, filters: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """Get search results from cache."""
        query_hash = hashlib.md5(f"{query}:{json.dumps(filters or {}, sort_keys=True)}".encode()).hexdigest()
        key = f"search:{query_hash}"
        return await self.cache_manager.get(key, "frequent_cache")
    
    async def set_search_results(self, query: str, results: List[Dict[str, Any]], filters: Dict[str, Any] = None):
        """Set search results in cache."""
        query_hash = hashlib.md5(f"{query}:{json.dumps(filters or {}, sort_keys=True)}".encode()).hexdigest()
        key = f"search:{query_hash}"
        await self.cache_manager.set(key, results, self.frequent_cache_config.ttl, "frequent_cache")
    
    async def get_popular_videos(self, category: str = "all", limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """Get popular videos from cache."""
        key = f"popular_videos:{category}:{limit}"
        return await self.cache_manager.get(key, "frequent_cache")
    
    async def set_popular_videos(self, category: str, videos: List[Dict[str, Any]], limit: int = 20):
        """Set popular videos in cache."""
        key = f"popular_videos:{category}:{limit}"
        await self.cache_manager.set(key, videos, self.frequent_cache_config.ttl, "frequent_cache")
    
    async def get_user_recommendations(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get user recommendations from cache."""
        key = f"recommendations:{user_id}"
        return await self.cache_manager.get(key, "frequent_cache")
    
    async def set_user_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]]):
        """Set user recommendations in cache."""
        key = f"recommendations:{user_id}"
        await self.cache_manager.set(key, recommendations, self.frequent_cache_config.ttl, "frequent_cache")
    
    async def get_trending_topics(self) -> Optional[List[str]]:
        """Get trending topics from cache."""
        key = "trending_topics"
        return await self.cache_manager.get(key, "frequent_cache")
    
    async def set_trending_topics(self, topics: List[str]):
        """Set trending topics in cache."""
        key = "trending_topics"
        await self.cache_manager.set(key, topics, self.frequent_cache_config.ttl, "frequent_cache")
    
    async def get_analytics_summary(self, time_range: str = "24h") -> Optional[Dict[str, Any]]:
        """Get analytics summary from cache."""
        key = f"analytics_summary:{time_range}"
        return await self.cache_manager.get(key, "frequent_cache")
    
    async def set_analytics_summary(self, time_range: str, summary: Dict[str, Any]):
        """Set analytics summary in cache."""
        key = f"analytics_summary:{time_range}"
        await self.cache_manager.set(key, summary, self.frequent_cache_config.ttl, "frequent_cache")

# =============================================================================
# Analytics Data Cache
# =============================================================================

class AnalyticsDataCache:
    """Specialized cache for analytics data."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.analytics_cache_config = CachePolicy(
            data_type=DataType.ANALYTICS_DATA,
            ttl=1800,  # 30 minutes
            max_size=500,
            strategy=CacheStrategy.LRU,
            priority="normal",
            compression_enabled=True,
            auto_refresh=True,
            refresh_interval=900  # 15 minutes
        )
    
    async def get_user_analytics(self, user_id: str, time_range: str = "7d") -> Optional[Dict[str, Any]]:
        """Get user analytics from cache."""
        key = f"user_analytics:{user_id}:{time_range}"
        return await self.cache_manager.get(key, "analytics_cache")
    
    async def set_user_analytics(self, user_id: str, analytics: Dict[str, Any], time_range: str = "7d"):
        """Set user analytics in cache."""
        key = f"user_analytics:{user_id}:{time_range}"
        await self.cache_manager.set(key, analytics, self.analytics_cache_config.ttl, "analytics_cache")
    
    async def get_video_analytics(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video analytics from cache."""
        key = f"video_analytics:{video_id}"
        return await self.cache_manager.get(key, "analytics_cache")
    
    async def set_video_analytics(self, video_id: str, analytics: Dict[str, Any]):
        """Set video analytics in cache."""
        key = f"video_analytics:{video_id}"
        await self.cache_manager.set(key, analytics, self.analytics_cache_config.ttl, "analytics_cache")
    
    async def get_platform_analytics(self, metric: str, time_range: str = "24h") -> Optional[Dict[str, Any]]:
        """Get platform analytics from cache."""
        key = f"platform_analytics:{metric}:{time_range}"
        return await self.cache_manager.get(key, "analytics_cache")
    
    async def set_platform_analytics(self, metric: str, analytics: Dict[str, Any], time_range: str = "24h"):
        """Set platform analytics in cache."""
        key = f"platform_analytics:{metric}:{time_range}"
        await self.cache_manager.set(key, analytics, self.analytics_cache_config.ttl, "analytics_cache")
    
    async def get_performance_metrics(self, component: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics from cache."""
        key = f"performance_metrics:{component}"
        return await self.cache_manager.get(key, "analytics_cache")
    
    async def set_performance_metrics(self, component: str, metrics: Dict[str, Any]):
        """Set performance metrics in cache."""
        key = f"performance_metrics:{component}"
        await self.cache_manager.set(key, metrics, 300, "analytics_cache")  # 5 minutes TTL

# =============================================================================
# Cache Factory
# =============================================================================

class CacheFactory:
    """Factory for creating specialized caches."""
    
    def __init__(self, cache_manager: CacheManager):
        
    """__init__ function."""
self.cache_manager = cache_manager
        self.specialized_caches: Dict[DataType, Any] = {}
    
    def get_user_cache(self) -> UserDataCache:
        """Get user data cache."""
        if DataType.USER_DATA not in self.specialized_caches:
            self.specialized_caches[DataType.USER_DATA] = UserDataCache(self.cache_manager)
        return self.specialized_caches[DataType.USER_DATA]
    
    def get_video_cache(self) -> VideoDataCache:
        """Get video data cache."""
        if DataType.VIDEO_DATA not in self.specialized_caches:
            self.specialized_caches[DataType.VIDEO_DATA] = VideoDataCache(self.cache_manager)
        return self.specialized_caches[DataType.VIDEO_DATA]
    
    def get_static_cache(self) -> StaticContentCache:
        """Get static content cache."""
        if DataType.STATIC_CONTENT not in self.specialized_caches:
            self.specialized_caches[DataType.STATIC_CONTENT] = StaticContentCache(self.cache_manager)
        return self.specialized_caches[DataType.STATIC_CONTENT]
    
    def get_frequent_cache(self) -> FrequentlyAccessedCache:
        """Get frequently accessed data cache."""
        if DataType.FREQUENTLY_ACCESSED not in self.specialized_caches:
            self.specialized_caches[DataType.FREQUENTLY_ACCESSED] = FrequentlyAccessedCache(self.cache_manager)
        return self.specialized_caches[DataType.FREQUENTLY_ACCESSED]
    
    def get_analytics_cache(self) -> AnalyticsDataCache:
        """Get analytics data cache."""
        if DataType.ANALYTICS_DATA not in self.specialized_caches:
            self.specialized_caches[DataType.ANALYTICS_DATA] = AnalyticsDataCache(self.cache_manager)
        return self.specialized_caches[DataType.ANALYTICS_DATA]
    
    def get_all_caches(self) -> Dict[DataType, Any]:
        """Get all specialized caches."""
        return self.specialized_caches

# =============================================================================
# Cache Decorators for Specialized Caches
# =============================================================================

def cache_user_data(ttl: int = 1800):
    """Decorator to cache user data."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract user_id from arguments
            user_id = None
            for arg in args:
                if isinstance(arg, str) and arg.startswith("user_"):
                    user_id = arg
                    break
            
            if not user_id:
                for key, value in kwargs.items():
                    if key == "user_id" or key.endswith("_id"):
                        user_id = value
                        break
            
            if user_id:
                # Try to get from cache
                cache_factory = globals().get('cache_factory')
                if cache_factory:
                    user_cache = cache_factory.get_user_cache()
                    cached_result = await user_cache.get_user_profile(user_id)
                    if cached_result is not None:
                        return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if user_id and cache_factory:
                user_cache = cache_factory.get_user_cache()
                await user_cache.set_user_profile(user_id, result)
            
            return result
        
        return wrapper
    return decorator

def cache_video_data(ttl: int = 3600):
    """Decorator to cache video data."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract video_id from arguments
            video_id = None
            for arg in args:
                if isinstance(arg, str) and arg.startswith("video_"):
                    video_id = arg
                    break
            
            if not video_id:
                for key, value in kwargs.items():
                    if key == "video_id" or key.endswith("_id"):
                        video_id = value
                        break
            
            if video_id:
                # Try to get from cache
                cache_factory = globals().get('cache_factory')
                if cache_factory:
                    video_cache = cache_factory.get_video_cache()
                    cached_result = await video_cache.get_video_metadata(video_id)
                    if cached_result is not None:
                        return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if video_id and cache_factory:
                video_cache = cache_factory.get_video_cache()
                await video_cache.set_video_metadata(video_id, result)
            
            return result
        
        return wrapper
    return decorator

def cache_static_content(ttl: int = 86400):
    """Decorator to cache static content."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items()))
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cache_factory = globals().get('cache_factory')
            if cache_factory:
                static_cache = cache_factory.get_static_cache()
                cached_result = await static_cache.get_configuration(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_factory:
                static_cache = cache_factory.get_static_cache()
                await static_cache.set_configuration(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

def invalidate_user_cache():
    """Decorator to invalidate user cache after function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Extract user_id and invalidate cache
            user_id = None
            for arg in args:
                if isinstance(arg, str) and arg.startswith("user_"):
                    user_id = arg
                    break
            
            if not user_id:
                for key, value in kwargs.items():
                    if key == "user_id" or key.endswith("_id"):
                        user_id = value
                        break
            
            if user_id:
                cache_factory = globals().get('cache_factory')
                if cache_factory:
                    user_cache = cache_factory.get_user_cache()
                    await user_cache.invalidate_user_cache(user_id)
            
            return result
        
        return wrapper
    return decorator

def invalidate_video_cache():
    """Decorator to invalidate video cache after function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Extract video_id and invalidate cache
            video_id = None
            for arg in args:
                if isinstance(arg, str) and arg.startswith("video_"):
                    video_id = arg
                    break
            
            if not video_id:
                for key, value in kwargs.items():
                    if key == "video_id" or key.endswith("_id"):
                        video_id = value
                        break
            
            if video_id:
                cache_factory = globals().get('cache_factory')
                if cache_factory:
                    video_cache = cache_factory.get_video_cache()
                    await video_cache.invalidate_video_cache(video_id)
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_cache_factory() -> CacheFactory:
    """Dependency to get cache factory instance."""
    # This would be configured in your FastAPI app
    cache_manager = CacheManager(CacheConfig())
    return CacheFactory(cache_manager)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "DataType",
    "CachePolicy",
    "UserDataCache",
    "VideoDataCache",
    "StaticContentCache",
    "FrequentlyAccessedCache",
    "AnalyticsDataCache",
    "CacheFactory",
    "cache_user_data",
    "cache_video_data",
    "cache_static_content",
    "invalidate_user_cache",
    "invalidate_video_cache",
    "get_cache_factory",
] 