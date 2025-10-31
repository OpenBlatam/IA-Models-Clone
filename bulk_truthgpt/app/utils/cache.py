"""
Advanced caching utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from flask import current_app, request, g

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced cache manager with multiple strategies."""
    
    def __init__(self, app=None):
        """Initialize cache manager."""
        self.app = app
        self.cache = None
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        if app:
            self.init_app(app)
    
    def init_app(self, app) -> None:
        """Initialize cache with app."""
        self.app = app
        self.cache = app.cache
        
        # Configure cache settings
        app.config.setdefault('CACHE_DEFAULT_TIMEOUT', 300)
        app.config.setdefault('CACHE_KEY_PREFIX', 'ultimate_enhanced_supreme')
        
        logger.info("💾 Cache manager initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with early returns."""
        if not self.cache:
            return default
        
        try:
            value = self.cache.get(key)
            if value is not None:
                self.cache_stats['hits'] += 1
                logger.debug(f"📦 Cache hit: {key}")
                return value
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"📭 Cache miss: {key}")
                return default
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"❌ Cache get error: {e}")
            return default
    
    def set(self, key: str, value: Any, timeout: int = None) -> bool:
        """Set value in cache with early returns."""
        if not self.cache:
            return False
        
        try:
            timeout = timeout or self.app.config.get('CACHE_DEFAULT_TIMEOUT', 300)
            self.cache.set(key, value, timeout=timeout)
            self.cache_stats['sets'] += 1
            logger.debug(f"💾 Cache set: {key}")
            return True
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache with early returns."""
        if not self.cache:
            return False
        
        try:
            self.cache.delete(key)
            self.cache_stats['deletes'] += 1
            logger.debug(f"🗑️ Cache delete: {key}")
            return True
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"❌ Cache delete error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache with early returns."""
        if not self.cache:
            return False
        
        try:
            self.cache.clear()
            logger.info("🧹 Cache cleared")
            return True
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"❌ Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_operations = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_operations * 100) if total_operations > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'sets': self.cache_stats['sets'],
            'deletes': self.cache_stats['deletes'],
            'errors': self.cache_stats['errors'],
            'hit_rate': round(hit_rate, 2),
            'total_operations': total_operations
        }

# Global cache manager instance
cache_manager = CacheManager()

def init_cache(app) -> None:
    """Initialize cache with app."""
    cache_manager.init_app(app)

def get_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key with early returns."""
    if not prefix:
        return ""
    
    # Create key from args and kwargs
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    
    # Create hash of key parts
    key_string = ":".join(key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    return f"{prefix}:{key_hash}"

def cache_result(ttl: int = 300, key_prefix: str = None, key_func: Callable = None):
    """Decorator for caching function results with early returns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                prefix = key_prefix or func.__name__
                cache_key = get_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_invalidate(pattern: str) -> bool:
    """Invalidate cache by pattern with early returns."""
    if not pattern:
        return False
    
    try:
        # This is a simplified implementation
        # In production, you'd use Redis SCAN or similar
        logger.info(f"🗑️ Cache invalidation requested for pattern: {pattern}")
        return True
    except Exception as e:
        logger.error(f"❌ Cache invalidation error: {e}")
        return False

def cache_warmup(warmup_funcs: List[Callable]) -> Dict[str, Any]:
    """Warm up cache with multiple functions."""
    results = {}
    
    for func in warmup_funcs:
        try:
            start_time = time.perf_counter()
            func()
            execution_time = time.perf_counter() - start_time
            
            results[func.__name__] = {
                'success': True,
                'execution_time': execution_time
            }
        except Exception as e:
            results[func.__name__] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def get_cached_data(key: str, fetch_func: Callable, ttl: int = 300) -> Any:
    """Get data from cache or fetch if not available."""
    # Try cache first
    cached_data = cache_manager.get(key)
    if cached_data is not None:
        return cached_data
    
    # Fetch data and cache it
    try:
        data = fetch_func()
        cache_manager.set(key, data, ttl)
        return data
    except Exception as e:
        logger.error(f"❌ Fetch function error: {e}")
        return None

def cache_user_data(user_id: str, data: Any, ttl: int = 3600) -> bool:
    """Cache user-specific data with early returns."""
    if not user_id or not data:
        return False
    
    key = f"user:{user_id}:data"
    return cache_manager.set(key, data, ttl)

def get_user_data(user_id: str) -> Any:
    """Get user-specific data from cache with early returns."""
    if not user_id:
        return None
    
    key = f"user:{user_id}:data"
    return cache_manager.get(key)

def cache_session_data(session_id: str, data: Any, ttl: int = 1800) -> bool:
    """Cache session-specific data with early returns."""
    if not session_id or not data:
        return False
    
    key = f"session:{session_id}:data"
    return cache_manager.set(key, data, ttl)

def get_session_data(session_id: str) -> Any:
    """Get session-specific data from cache with early returns."""
    if not session_id:
        return None
    
    key = f"session:{session_id}:data"
    return cache_manager.get(key)

def cache_api_response(endpoint: str, params: Dict[str, Any], response: Any, ttl: int = 300) -> bool:
    """Cache API response with early returns."""
    if not endpoint or not response:
        return False
    
    # Create cache key from endpoint and params
    key = get_cache_key(f"api:{endpoint}", **params)
    return cache_manager.set(key, response, ttl)

def get_cached_api_response(endpoint: str, params: Dict[str, Any]) -> Any:
    """Get cached API response with early returns."""
    if not endpoint:
        return None
    
    key = get_cache_key(f"api:{endpoint}", **params)
    return cache_manager.get(key)

def cache_performance_metrics(metrics: Dict[str, Any], ttl: int = 600) -> bool:
    """Cache performance metrics with early returns."""
    if not metrics:
        return False
    
    key = f"performance:metrics:{int(time.time())}"
    return cache_manager.set(key, metrics, ttl)

def get_performance_metrics(time_range: int = 3600) -> List[Dict[str, Any]]:
    """Get cached performance metrics for time range."""
    # This is a simplified implementation
    # In production, you'd query by time range
    logger.info(f"📊 Retrieving performance metrics for last {time_range} seconds")
    return []

def cache_optimization_result(optimization_id: str, result: Dict[str, Any], ttl: int = 3600) -> bool:
    """Cache optimization result with early returns."""
    if not optimization_id or not result:
        return False
    
    key = f"optimization:{optimization_id}:result"
    return cache_manager.set(key, result, ttl)

def get_optimization_result(optimization_id: str) -> Any:
    """Get cached optimization result with early returns."""
    if not optimization_id:
        return None
    
    key = f"optimization:{optimization_id}:result"
    return cache_manager.get(key)

def cache_health_status(status: Dict[str, Any], ttl: int = 60) -> bool:
    """Cache health status with early returns."""
    if not status:
        return False
    
    key = "health:status"
    return cache_manager.set(key, status, ttl)

def get_health_status() -> Any:
    """Get cached health status with early returns."""
    key = "health:status"
    return cache_manager.get(key)

def cache_analytics_data(analytics_type: str, data: Any, ttl: int = 1800) -> bool:
    """Cache analytics data with early returns."""
    if not analytics_type or not data:
        return False
    
    key = f"analytics:{analytics_type}:data"
    return cache_manager.set(key, data, ttl)

def get_analytics_data(analytics_type: str) -> Any:
    """Get cached analytics data with early returns."""
    if not analytics_type:
        return None
    
    key = f"analytics:{analytics_type}:data"
    return cache_manager.get(key)

def clear_user_cache(user_id: str) -> bool:
    """Clear all cache for specific user with early returns."""
    if not user_id:
        return False
    
    # Clear user-specific cache
    user_key = f"user:{user_id}:*"
    return cache_invalidate(user_key)

def clear_session_cache(session_id: str) -> bool:
    """Clear all cache for specific session with early returns."""
    if not session_id:
        return False
    
    # Clear session-specific cache
    session_key = f"session:{session_id}:*"
    return cache_invalidate(session_key)

def get_cache_health() -> Dict[str, Any]:
    """Get cache health information."""
    stats = cache_manager.get_stats()
    
    return {
        'status': 'healthy' if stats['errors'] == 0 else 'unhealthy',
        'stats': stats,
        'timestamp': time.time()
    }

# Cache decorators
def cache_with_ttl(ttl: int = 300):
    """Decorator for caching with custom TTL."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            
            # Try cache first
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def cache_conditional(condition_func: Callable):
    """Decorator for conditional caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func(*args, **kwargs):
                cache_key = get_cache_key(func.__name__, *args, **kwargs)
                cached_result = cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            result = func(*args, **kwargs)
            
            if condition_func(*args, **kwargs):
                cache_key = get_cache_key(func.__name__, *args, **kwargs)
                cache_manager.set(cache_key, result)
            
            return result
        return wrapper
    return decorator









