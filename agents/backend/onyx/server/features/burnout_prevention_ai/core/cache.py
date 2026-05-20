"""
Simple in-memory cache for performance
=====================================
Thread-safe LRU cache with TTL.
"""

import time
import threading
from typing import Any, Optional, Dict
from collections import OrderedDict
from .hashing import hash_data

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .constants import CACHE_MAX_SIZE, CACHE_DEFAULT_TTL


class SimpleCache:
    """
    Thread-safe LRU cache with TTL.
    
    Uses threading.Lock for thread-safety in multi-threaded environments.
    """
    
    def __init__(self, max_size: int = 100, default_ttl: float = 300.0):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # Cache metrics functions to avoid repeated imports
        self._metrics_functions = self._load_metrics_functions()
    
    def _load_metrics_functions(self) -> Dict[str, Any]:
        """
        Load metrics functions once to avoid repeated imports.
        
        Returns:
            Dict with metrics functions or empty dict if not available
        """
        try:
            from .metrics import record_cache_hit, record_cache_miss, update_cache_size
            return {
                "record_cache_hit": record_cache_hit,
                "record_cache_miss": record_cache_miss,
                "update_cache_size": update_cache_size
            }
        except ImportError:
            return {}
    
    def _make_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments (optimized).
        
        Uses centralized hashing utility for consistent, short keys.
        """
        return hash_data(*args, **kwargs)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired (thread-safe).
        
        Args:
            key: Cache key to look up
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                # Record miss metric (optimized)
                if "record_cache_miss" in self._metrics_functions:
                    self._metrics_functions["record_cache_miss"]()
                self._record_metrics()
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                self._misses += 1
                # Record miss metric (optimized)
                if "record_cache_miss" in self._metrics_functions:
                    self._metrics_functions["record_cache_miss"]()
                self._record_metrics()
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            # Record hit metric (optimized)
            if "record_cache_hit" in self._metrics_functions:
                self._metrics_functions["record_cache_hit"]()
            self._record_metrics()
            return entry["value"]
    
    def _record_metrics(self) -> None:
        """
        Record cache metrics (optimized to use cached functions).
        
        Only records metrics if metrics module is available.
        Uses batched recording to reduce overhead.
        """
        if "update_cache_size" in self._metrics_functions:
            self._metrics_functions["update_cache_size"](len(self._cache))
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set value in cache with TTL (thread-safe).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = {"value": value, "expires_at": expires_at}
            
            # Evict oldest if over limit
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
            
            self._update_cache_size_metric()
    
    def _update_cache_size_metric(self) -> None:
        """Update cache size metric (optimized to use cached functions)."""
        if "update_cache_size" in self._metrics_functions:
            self._metrics_functions["update_cache_size"](len(self._cache))
    
    def clear(self) -> None:
        """Clear all cache entries (thread-safe)."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def clear_expired(self) -> int:
        """
        Remove expired entries (optimized, thread-safe).
        
        Uses efficient iteration to remove expired entries in a single pass.
        Updates metrics after cleanup.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            # More efficient: iterate and delete in reverse to avoid index issues
            expired_keys = [
                key for key, entry in self._cache.items() 
                if now > entry.get("expires_at", 0)
            ]
            # Delete expired entries (safe pop to avoid KeyError)
            removed_count = 0
            for key in expired_keys:
                if self._cache.pop(key, None) is not None:
                    removed_count += 1
            
            # Update metrics if entries were removed
            if removed_count > 0:
                self._update_cache_size_metric()
            
            return removed_count
    
    def size(self) -> int:
        """Get cache size (thread-safe)."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics (size, hits, misses, hit_rate)
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2)
            }


# Global cache instance
_cache = SimpleCache(max_size=CACHE_MAX_SIZE, default_ttl=CACHE_DEFAULT_TTL)


def get_cache(key: str) -> Optional[Any]:
    """Get from global cache."""
    return _cache.get(key)


def set_cache(key: str, value: Any, ttl: Optional[float] = None) -> None:
    """Set in global cache."""
    _cache.set(key, value, ttl)


def clear_cache() -> None:
    """Clear global cache."""
    _cache.clear()


def make_messages_cache_key(
    messages: list,
    model: str,
    max_tokens: int,
    temperature: float,
    prefix: str = "api"
) -> str:
    """
    Generate cache key from messages list (optimized).
    
    Args:
        messages: List of message dictionaries
        model: Model name
        max_tokens: Maximum tokens
        temperature: Temperature value
        prefix: Cache key prefix
        
    Returns:
        Cache key string
    """
    from .hashing import hash_string
    
    # Sort messages for consistent cache keys
    messages_parts = [
        f"{m.get('role', '')}:{m.get('content', '')}"
        for m in sorted(messages, key=lambda x: (x.get("role", ""), x.get("content", "")))
    ]
    messages_str = "|".join(messages_parts)
    messages_hash = hash_string(messages_str, length=16)
    return make_cache_key(prefix, messages_hash, model, max_tokens, int(temperature * 10))


def make_cache_key(*args, **kwargs) -> str:
    """Make cache key from arguments."""
    return _cache._make_key(*args, **kwargs)


def clear_expired() -> int:
    """
    Clear expired entries from global cache.
    
    Returns:
        Number of entries removed
    """
    return _cache.clear_expired()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache.get_stats()

