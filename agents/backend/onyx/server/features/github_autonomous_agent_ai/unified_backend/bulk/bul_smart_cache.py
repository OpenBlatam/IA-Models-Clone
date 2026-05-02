"""
BUL - Business Universal Language (Smart Cache System)
====================================================

Intelligent caching system with advanced features.
"""

import redis
import json
import time
import hashlib
import pickle
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from functools import wraps
import asyncio
import threading

logger = logging.getLogger(__name__)

class SmartCache:
    """Intelligent caching system with advanced features."""
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize smart cache system."""
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db, 
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_type = "redis"
            logger.info("Connected to Redis cache")
        except:
            self.redis_client = None
            self.cache_type = "memory"
            self.memory_cache = {}
            logger.info("Using in-memory cache")
        
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "expires": 0
        }
        
        self.cache_policies = {
            "default_ttl": 3600,  # 1 hour
            "max_size": 10000,
            "cleanup_interval": 300,  # 5 minutes
            "eviction_policy": "lru"  # Least Recently Used
        }
        
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    self.cleanup_expired()
                    time.sleep(self.cache_policies["cleanup_interval"])
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("Cache cleanup thread started")
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.cache_type == "redis":
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats["hits"] += 1
                    return json.loads(value)
                else:
                    self.cache_stats["misses"] += 1
                    return None
            else:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry["expires"] > time.time():
                        self.cache_stats["hits"] += 1
                        return cache_entry["value"]
                    else:
                        del self.memory_cache[key]
                        self.cache_stats["expires"] += 1
                
                self.cache_stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            if ttl is None:
                ttl = self.cache_policies["default_ttl"]
            
            if self.cache_type == "redis":
                success = self.redis_client.setex(key, ttl, json.dumps(value))
                if success:
                    self.cache_stats["sets"] += 1
                return success
            else:
                self.memory_cache[key] = {
                    "value": value,
                    "expires": time.time() + ttl,
                    "created": time.time()
                }
                self.cache_stats["sets"] += 1
                
                # Check size limit
                if len(self.memory_cache) > self.cache_policies["max_size"]:
                    self.evict_oldest()
                
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.cache_type == "redis":
                success = self.redis_client.delete(key)
                if success:
                    self.cache_stats["deletes"] += 1
                return bool(success)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    self.cache_stats["deletes"] += 1
                    return True
                return False
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self.cache_type == "redis":
                return bool(self.redis_client.exists(key))
            else:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    if cache_entry["expires"] > time.time():
                        return True
                    else:
                        del self.memory_cache[key]
                        self.cache_stats["expires"] += 1
                return False
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def cleanup_expired(self):
        """Clean up expired entries."""
        if self.cache_type == "memory":
            current_time = time.time()
            expired_keys = []
            
            for key, cache_entry in self.memory_cache.items():
                if cache_entry["expires"] <= current_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.memory_cache[key]
                self.cache_stats["expires"] += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def evict_oldest(self):
        """Evict oldest entries based on policy."""
        if self.cache_type == "memory" and self.memory_cache:
            if self.cache_policies["eviction_policy"] == "lru":
                # Sort by creation time and remove oldest
                sorted_items = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1]["created"]
                )
                
                # Remove 10% of entries
                to_remove = max(1, len(sorted_items) // 10)
                for i in range(to_remove):
                    key = sorted_items[i][0]
                    del self.memory_cache[key]
                    self.cache_stats["deletes"] += 1
                
                logger.info(f"Evicted {to_remove} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests > 0:
            hit_rate = (self.cache_stats["hits"] / total_requests) * 100
        
        stats = {
            "cache_type": self.cache_type,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "cache_size": len(self.memory_cache) if self.cache_type == "memory" else "N/A",
            "policies": self.cache_policies,
            **self.cache_stats
        }
        
        return stats
    
    def clear(self):
        """Clear all cache entries."""
        try:
            if self.cache_type == "redis":
                self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

# Global cache instance
smart_cache = SmartCache()

def cached(ttl: Optional[int] = None, prefix: str = "default"):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = smart_cache.generate_key(prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = smart_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            smart_cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = smart_cache.generate_key(prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = smart_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            smart_cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

class CacheManager:
    """Advanced cache management system."""
    
    def __init__(self):
        self.cache = smart_cache
        self.cache_groups = {
            "ai_responses": {"ttl": 1800, "prefix": "ai"},  # 30 minutes
            "user_sessions": {"ttl": 3600, "prefix": "user"},  # 1 hour
            "api_responses": {"ttl": 300, "prefix": "api"},  # 5 minutes
            "system_metrics": {"ttl": 60, "prefix": "metrics"},  # 1 minute
            "document_cache": {"ttl": 7200, "prefix": "doc"}  # 2 hours
        }
    
    def get_group_cache(self, group: str, key: str) -> Optional[Any]:
        """Get value from specific cache group."""
        if group not in self.cache_groups:
            return None
        
        full_key = f"{self.cache_groups[group]['prefix']}:{key}"
        return self.cache.get(full_key)
    
    def set_group_cache(self, group: str, key: str, value: Any) -> bool:
        """Set value in specific cache group."""
        if group not in self.cache_groups:
            return False
        
        full_key = f"{self.cache_groups[group]['prefix']}:{key}"
        ttl = self.cache_groups[group]["ttl"]
        return self.cache.set(full_key, value, ttl)
    
    def invalidate_group(self, group: str) -> bool:
        """Invalidate entire cache group."""
        if group not in self.cache_groups:
            return False
        
        prefix = self.cache_groups[group]["prefix"]
        try:
            if self.cache.cache_type == "redis":
                # Get all keys with prefix
                keys = self.cache.redis_client.keys(f"{prefix}:*")
                if keys:
                    self.cache.redis_client.delete(*keys)
                return True
            else:
                # Remove from memory cache
                keys_to_remove = [k for k in self.cache.memory_cache.keys() if k.startswith(f"{prefix}:")]
                for key in keys_to_remove:
                    del self.cache.memory_cache[key]
                return True
        except Exception as e:
            logger.error(f"Error invalidating group {group}: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        return {
            "cache_stats": self.cache.get_stats(),
            "cache_groups": self.cache_groups,
            "group_sizes": self.get_group_sizes()
        }
    
    def get_group_sizes(self) -> Dict[str, int]:
        """Get sizes of cache groups."""
        group_sizes = {}
        
        try:
            if self.cache.cache_type == "redis":
                for group, config in self.cache_groups.items():
                    prefix = config["prefix"]
                    keys = self.cache.redis_client.keys(f"{prefix}:*")
                    group_sizes[group] = len(keys)
            else:
                for group, config in self.cache_groups.items():
                    prefix = config["prefix"]
                    count = sum(1 for k in self.cache.memory_cache.keys() if k.startswith(f"{prefix}:"))
                    group_sizes[group] = count
        except Exception as e:
            logger.error(f"Error getting group sizes: {e}")
        
        return group_sizes

# Global cache manager
cache_manager = CacheManager()

# Example usage functions
@cached(ttl=300, prefix="ai")
def get_ai_response(prompt: str, model: str) -> str:
    """Example AI response function with caching."""
    # Simulate AI processing
    time.sleep(0.1)
    return f"AI response for: {prompt[:50]}... using {model}"

@cached(ttl=60, prefix="metrics")
def get_system_metrics() -> Dict[str, Any]:
    """Example system metrics function with caching."""
    import psutil
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Test the cache system
    print("Testing Smart Cache System...")
    
    # Test basic operations
    cache_manager.set_group_cache("ai_responses", "test_key", {"response": "test data"})
    result = cache_manager.get_group_cache("ai_responses", "test_key")
    print(f"Cache test result: {result}")
    
    # Test decorator
    response1 = get_ai_response("Test prompt", "gpt-4")
    response2 = get_ai_response("Test prompt", "gpt-4")  # Should be cached
    print(f"Response 1: {response1}")
    print(f"Response 2 (cached): {response2}")
    
    # Print cache info
    cache_info = cache_manager.get_cache_info()
    print(f"Cache info: {json.dumps(cache_info, indent=2)}")
