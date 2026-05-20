"""
ML NLP Benchmark Cache System
Real, working advanced caching for ML NLP Benchmark system
"""

import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict
import pickle
import gzip
import base64
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class MLNLPBenchmarkCache:
    """Advanced caching system for ML NLP Benchmark"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_counts = {}
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "compressions": 0
        }
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self.timestamps.items():
            if current_time - timestamp > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_key(key)
            self.stats["evictions"] += 1
    
    def _evict_lru(self):
        """Remove least recently used entry"""
        if not self.cache:
            return
        
        # Remove least recently used (first item in OrderedDict)
        key, _ = self.cache.popitem(last=False)
        self._remove_key(key)
        self.stats["evictions"] += 1
    
    def _remove_key(self, key: str):
        """Remove key from all internal structures"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_counts.pop(key, None)
    
    def _compress_data(self, data: Any) -> str:
        """Compress data for storage"""
        try:
            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized)
            encoded = base64.b64encode(compressed).decode()
            self.stats["compressions"] += 1
            return encoded
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return str(data)
    
    def _decompress_data(self, compressed_data: str) -> Any:
        """Decompress data from storage"""
        try:
            decoded = base64.b64decode(compressed_data.encode())
            decompressed = gzip.decompress(decoded)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return compressed_data
    
    def get(self, key: Union[str, Any]) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        with self.lock:
            if cache_key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            if self._is_expired(cache_key):
                self._remove_key(cache_key)
                self.stats["misses"] += 1
                self.stats["evictions"] += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            
            # Update access count
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            
            self.stats["hits"] += 1
            
            # Decompress if needed
            if isinstance(value, str) and value.startswith('gzip:'):
                return self._decompress_data(value[5:])
            
            return value
    
    def set(self, key: Union[str, Any], value: Any, compress: bool = True) -> bool:
        """Set value in cache"""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        with self.lock:
            # Remove expired entries
            self._evict_expired()
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and cache_key not in self.cache:
                self._evict_lru()
            
            # Compress value if requested
            if compress and not isinstance(value, (str, int, float, bool)):
                compressed_value = f"gzip:{self._compress_data(value)}"
            else:
                compressed_value = value
            
            # Store in cache
            self.cache[cache_key] = compressed_value
            self.timestamps[cache_key] = time.time()
            self.access_counts[cache_key] = 0
            
            self.stats["sets"] += 1
            return True
    
    def delete(self, key: Union[str, Any]) -> bool:
        """Delete key from cache"""
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        with self.lock:
            if cache_key in self.cache:
                self._remove_key(cache_key)
                self.stats["deletes"] += 1
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_counts.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hit_rate": hit_rate,
                "stats": self.stats.copy(),
                "memory_usage": self._estimate_memory_usage(),
                "oldest_entry": min(self.timestamps.values()) if self.timestamps else None,
                "newest_entry": max(self.timestamps.values()) if self.timestamps else None
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        total_size = 0
        for key, value in self.cache.items():
            total_size += len(str(key)) + len(str(value))
        return total_size
    
    def get_most_accessed(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most accessed cache entries"""
        with self.lock:
            sorted_items = sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [
                {
                    "key": key,
                    "access_count": count,
                    "timestamp": self.timestamps.get(key),
                    "age_seconds": time.time() - self.timestamps.get(key, 0) if key in self.timestamps else 0
                }
                for key, count in sorted_items[:limit]
            ]
    
    def cleanup_expired(self) -> int:
        """Manually cleanup expired entries"""
        with self.lock:
            initial_size = len(self.cache)
            self._evict_expired()
            return initial_size - len(self.cache)

class MLNLPBenchmarkCacheManager:
    """Cache manager for multiple cache instances"""
    
    def __init__(self):
        self.caches = {}
        self.default_cache = MLNLPBenchmarkCache()
    
    def get_cache(self, name: str = "default", max_size: int = 10000, ttl: int = 3600) -> MLNLPBenchmarkCache:
        """Get or create cache instance"""
        if name not in self.caches:
            self.caches[name] = MLNLPBenchmarkCache(max_size=max_size, ttl=ttl)
        return self.caches[name]
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        stats["default"] = self.default_cache.get_stats()
        return stats
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self.caches.values():
            cache.clear()
        self.default_cache.clear()
        logger.info("All caches cleared")
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches"""
        results = {}
        for name, cache in self.caches.items():
            results[name] = cache.cleanup_expired()
        results["default"] = self.default_cache.cleanup_expired()
        return results

class MLNLPBenchmarkCacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, cache: MLNLPBenchmarkCache, ttl: int = 3600):
        self.cache = cache
        self.ttl = ttl
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result, ttl=self.ttl)
            return result
        
        return wrapper

# Global cache manager
cache_manager = MLNLPBenchmarkCacheManager()

def get_cache(name: str = "default", max_size: int = 10000, ttl: int = 3600) -> MLNLPBenchmarkCache:
    """Get cache instance"""
    return cache_manager.get_cache(name, max_size, ttl)

def cache_result(ttl: int = 3600, cache_name: str = "default"):
    """Decorator for caching function results"""
    cache = get_cache(cache_name)
    return MLNLPBenchmarkCacheDecorator(cache, ttl)

def clear_cache(name: str = "default"):
    """Clear specific cache"""
    if name == "default":
        cache_manager.default_cache.clear()
    elif name in cache_manager.caches:
        cache_manager.caches[name].clear()

def clear_all_caches():
    """Clear all caches"""
    cache_manager.clear_all()

def get_cache_stats(name: str = "default") -> Dict[str, Any]:
    """Get cache statistics"""
    if name == "default":
        return cache_manager.default_cache.get_stats()
    elif name in cache_manager.caches:
        return cache_manager.caches[name].get_stats()
    else:
        return {"error": f"Cache '{name}' not found"}

def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return cache_manager.get_all_stats()

def cleanup_expired(name: str = "default") -> int:
    """Cleanup expired entries"""
    if name == "default":
        return cache_manager.default_cache.cleanup_expired()
    elif name in cache_manager.caches:
        return cache_manager.caches[name].cleanup_expired()
    else:
        return 0

def cleanup_all_expired() -> Dict[str, int]:
    """Cleanup expired entries in all caches"""
    return cache_manager.cleanup_all()











