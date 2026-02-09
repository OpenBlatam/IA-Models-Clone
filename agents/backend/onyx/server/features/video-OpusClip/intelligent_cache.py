"""
Intelligent Caching System for Ultimate Opus Clip

Advanced caching system that improves performance by intelligently
caching processed results, models, and intermediate data.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable
import asyncio
import time
import hashlib
import json
import pickle
import threading
from dataclasses import dataclass
from enum import Enum
import structlog
from pathlib import Path
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import yaml

logger = structlog.get_logger("intelligent_cache")

class CacheType(Enum):
    """Types of cache entries."""
    VIDEO_FRAMES = "video_frames"
    PROCESSING_RESULTS = "processing_results"
    MODEL_PREDICTIONS = "model_predictions"
    API_RESPONSES = "api_responses"
    CONFIGURATION = "configuration"
    TEMPORARY = "temporary"

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based

@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    tags: List[str] = None

@dataclass
class CacheConfig:
    """Configuration for the intelligent cache."""
    max_size_mb: int = 1024
    max_entries: int = 10000
    default_ttl: float = 3600  # 1 hour
    cleanup_interval: float = 300  # 5 minutes
    compression_enabled: bool = True
    persistence_enabled: bool = True
    cache_directory: str = "cache"
    policy: CachePolicy = CachePolicy.LRU

class IntelligentCache:
    """Intelligent caching system with multiple strategies."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.access_frequency: Dict[str, int] = {}
        self.total_size_bytes = 0
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = False
        
        # Initialize cache directory
        self.cache_dir = Path(self.config.cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Start cleanup thread
        self.start_cleanup_thread()
        
        logger.info("Intelligent cache initialized")
    
    def start_cleanup_thread(self):
        """Start the cleanup thread."""
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
            logger.info("Cache cleanup thread started")
    
    def stop_cleanup_thread(self):
        """Stop the cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("Cache cleanup thread stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop."""
        while self.running:
            try:
                self._cleanup_expired_entries()
                self._cleanup_by_size()
                self._cleanup_by_policy()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
    
    def _cleanup_by_size(self):
        """Clean up cache by size limits."""
        if self.total_size_bytes <= self.config.max_size_mb * 1024 * 1024:
            return
        
        # Calculate how much to remove
        target_size = self.config.max_size_mb * 1024 * 1024 * 0.8  # Remove to 80%
        bytes_to_remove = self.total_size_bytes - target_size
        
        # Sort entries by policy
        if self.config.policy == CachePolicy.LRU:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.config.policy == CachePolicy.LFU:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        else:
            sorted_entries = list(self.cache.items())
        
        # Remove entries until size target is met
        removed_bytes = 0
        for key, entry in sorted_entries:
            if removed_bytes >= bytes_to_remove:
                break
            
            self._remove_entry(key)
            removed_bytes += entry.size_bytes
        
        logger.info(f"Cleaned up {removed_bytes} bytes from cache")
    
    def _cleanup_by_policy(self):
        """Clean up cache by policy."""
        if len(self.cache) <= self.config.max_entries:
            return
        
        # Remove excess entries
        excess_count = len(self.cache) - self.config.max_entries
        
        if self.config.policy == CachePolicy.LRU:
            # Remove least recently used
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )
        elif self.config.policy == CachePolicy.LFU:
            # Remove least frequently used
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k].access_count
            )
        else:
            # Remove oldest
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at
            )
        
        # Remove excess entries
        for key in sorted_keys[:excess_count]:
            self._remove_entry(key)
        
        logger.info(f"Removed {excess_count} entries by policy")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                
                # Update access frequency
                if key in self.access_frequency:
                    del self.access_frequency[key]
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate the size of a value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v)
                    for k, v in value.items()
                )
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            else:
                # Fallback to pickle size
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a hash of the arguments
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                self._remove_entry(key)
                return None
            
            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Update access frequency
            self.access_frequency[key] = self.access_frequency.get(key, 0) + 1
            
            logger.debug(f"Cache hit for key: {key}")
            return entry.value
    
    def set(self, key: str, value: Any, cache_type: CacheType = CacheType.TEMPORARY, 
            ttl: Optional[float] = None, tags: List[str] = None) -> bool:
        """Set a value in the cache."""
        try:
            with self.lock:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check if we need to make space
                if size_bytes > self.config.max_size_mb * 1024 * 1024:
                    logger.warning(f"Value too large for cache: {size_bytes} bytes")
                    return False
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=cache_type,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl=ttl or self.config.default_ttl,
                    tags=tags or []
                )
                
                # Remove existing entry if it exists
                if key in self.cache:
                    self._remove_entry(key)
                
                # Add new entry
                self.cache[key] = entry
                self.total_size_bytes += size_bytes
                
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                # Update access frequency
                self.access_frequency[key] = 1
                
                logger.debug(f"Cached value for key: {key}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                logger.debug(f"Deleted cache entry: {key}")
                return True
            return False
    
    def clear(self, cache_type: Optional[CacheType] = None):
        """Clear cache entries."""
        with self.lock:
            if cache_type:
                # Clear only specific type
                keys_to_remove = [
                    key for key, entry in self.cache.items()
                    if entry.cache_type == cache_type
                ]
                for key in keys_to_remove:
                    self._remove_entry(key)
                logger.info(f"Cleared {len(keys_to_remove)} entries of type {cache_type}")
            else:
                # Clear all
                self.cache.clear()
                self.access_order.clear()
                self.access_frequency.clear()
                self.total_size_bytes = 0
                logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "total_entries": len(self.cache),
                "total_size_mb": self.total_size_bytes / (1024 * 1024),
                "max_size_mb": self.config.max_size_mb,
                "max_entries": self.config.max_entries,
                "hit_rate": self._calculate_hit_rate(),
                "entries_by_type": self._get_entries_by_type(),
                "oldest_entry": self._get_oldest_entry(),
                "newest_entry": self._get_newest_entry()
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_accesses = sum(self.access_frequency.values())
        if total_accesses == 0:
            return 0.0
        
        # This is a simplified calculation
        # In a real implementation, you'd track hits vs misses
        return 0.8  # Placeholder
    
    def _get_entries_by_type(self) -> Dict[str, int]:
        """Get count of entries by type."""
        type_counts = {}
        for entry in self.cache.values():
            type_name = entry.cache_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts
    
    def _get_oldest_entry(self) -> Optional[Dict[str, Any]]:
        """Get oldest cache entry."""
        if not self.cache:
            return None
        
        oldest_entry = min(self.cache.values(), key=lambda x: x.created_at)
        return {
            "key": oldest_entry.key,
            "created_at": oldest_entry.created_at,
            "cache_type": oldest_entry.cache_type.value
        }
    
    def _get_newest_entry(self) -> Optional[Dict[str, Any]]:
        """Get newest cache entry."""
        if not self.cache:
            return None
        
        newest_entry = max(self.cache.values(), key=lambda x: x.created_at)
        return {
            "key": newest_entry.key,
            "created_at": newest_entry.created_at,
            "cache_type": newest_entry.cache_type.value
        }
    
    def cache_function(self, cache_type: CacheType = CacheType.PROCESSING_RESULTS,
                      ttl: Optional[float] = None, tags: List[str] = None):
        """Decorator to cache function results."""
        def decorator(func: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, cache_type, ttl, tags)
                
                return result
            
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                key = self._generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(key, result, cache_type, ttl, tags)
                
                return result
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def cleanup(self):
        """Cleanup cache resources."""
        self.stop_cleanup_thread()
        self.clear()
        logger.info("Intelligent cache cleaned up")

# Global cache instance
_global_cache: Optional[IntelligentCache] = None

def get_cache() -> IntelligentCache:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache

def cache_result(cache_type: CacheType = CacheType.PROCESSING_RESULTS,
                ttl: Optional[float] = None, tags: List[str] = None):
    """Decorator to cache function results."""
    cache = get_cache()
    return cache.cache_function(cache_type, ttl, tags)


