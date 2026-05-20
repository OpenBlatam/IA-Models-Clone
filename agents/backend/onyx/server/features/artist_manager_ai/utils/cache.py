"""
Advanced Cache Manager
======================

Advanced cache manager with LRU eviction, statistics, and persistence.
Refactored with best practices.
"""

import logging
import threading
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Advanced cache manager with:
    - TTL (Time To Live) support
    - LRU (Least Recently Used) eviction
    - Thread-safe operations
    - Statistics tracking
    - Persistence support
    - Memory limits
    """
    
    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        max_size: Optional[int] = None,
        enable_lru: bool = True,
        thread_safe: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            default_ttl_seconds: Default TTL in seconds
            max_size: Maximum number of entries (None = unlimited)
            enable_lru: Enable LRU eviction
            thread_safe: Enable thread safety
        """
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size
        self.enable_lru = enable_lru
        self.thread_safe = thread_safe
        
        # Use OrderedDict for LRU support
        if enable_lru:
            self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        else:
            self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Thread lock for thread safety
        self._lock = threading.RLock() if thread_safe else None
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        
        self._logger = logger
    
    def _acquire_lock(self):
        """Acquire lock if thread-safe."""
        if self._lock:
            self._lock.acquire()
    
    def _release_lock(self):
        """Release lock if thread-safe."""
        if self._lock:
            self._lock.release()
    
    def _generate_key(
        self,
        prefix: str,
        *args,
        **kwargs
    ) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Cache key string
        """
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        if self.max_size is None:
            return
        
        if len(self.cache) >= self.max_size:
            if self.enable_lru:
                # Remove least recently used (first item in OrderedDict)
                self.cache.popitem(last=False)
            else:
                # Remove oldest entry
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].get("created_at", datetime.min)
                )
                del self.cache[oldest_key]
            
            self.stats["evictions"] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        self._acquire_lock()
        try:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.get("expires_at") and entry["expires_at"] < datetime.now():
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            # Update LRU order
            if self.enable_lru:
                self.cache.move_to_end(key)
            
            self.stats["hits"] += 1
            return entry["value"]
        finally:
            self._release_lock()
    
    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (None = use default)
        """
        self._acquire_lock()
        try:
            # Evict if needed
            if key not in self.cache:
                self._evict_if_needed()
            
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            entry = {
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now(),
                "access_count": self.cache.get(key, {}).get("access_count", 0) + 1
            }
            
            self.cache[key] = entry
            
            # Update LRU order
            if self.enable_lru:
                self.cache.move_to_end(key)
            
            self.stats["sets"] += 1
        finally:
            self._release_lock()
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        self._acquire_lock()
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
            return False
        finally:
            self._release_lock()
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            prefix: Optional prefix to clear only matching keys
        
        Returns:
            Number of keys deleted
        """
        self._acquire_lock()
        try:
            if prefix:
                keys_to_delete = [
                    key for key in list(self.cache.keys())
                    if key.startswith(prefix)
                ]
                for key in keys_to_delete:
                    del self.cache[key]
                self.stats["deletes"] += len(keys_to_delete)
                return len(keys_to_delete)
            else:
                count = len(self.cache)
                self.cache.clear()
                self.stats["deletes"] += count
                return count
        finally:
            self._release_lock()
    
    def get_or_set(
        self,
        key: str,
        callable_func: Callable[[], Any],
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and set.
        
        Args:
            key: Cache key
            callable_func: Function to compute value if not cached
            ttl_seconds: TTL in seconds
        
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = callable_func()
        self.set(key, value, ttl_seconds)
        return value
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired entries.
        
        Returns:
            Number of entries removed
        """
        self._acquire_lock()
        try:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.get("expires_at") and entry["expires_at"] < now
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
        finally:
            self._release_lock()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        self._acquire_lock()
        try:
            now = datetime.now()
            total_entries = len(self.cache)
            expired_entries = sum(
                1 for entry in self.cache.values()
                if entry.get("expires_at") and entry["expires_at"] < now
            )
            
            hits = self.stats["hits"]
            misses = self.stats["misses"]
            total_requests = hits + misses
            hit_rate = hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_entries,
                "expired_entries": expired_entries,
                "max_size": self.max_size,
                "default_ttl_seconds": self.default_ttl,
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "sets": self.stats["sets"],
                "deletes": self.stats["deletes"],
                "evictions": self.stats["evictions"]
            }
        finally:
            self._release_lock()
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save cache to file.
        
        Args:
            filepath: Path to save cache
        """
        self._acquire_lock()
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Only save non-expired entries
            now = datetime.now()
            cache_data = {
                key: entry for key, entry in self.cache.items()
                if not entry.get("expires_at") or entry["expires_at"] > now
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(cache_data, f)
            
            self._logger.info(f"Cache saved to {filepath}")
        finally:
            self._release_lock()
    
    def load_from_file(self, filepath: Union[str, Path]) -> int:
        """
        Load cache from file.
        
        Args:
            filepath: Path to load cache from
        
        Returns:
            Number of entries loaded
        """
        self._acquire_lock()
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                self._logger.warning(f"Cache file not found: {filepath}")
                return 0
            
            with open(filepath, "rb") as f:
                cache_data = pickle.load(f)
            
            # Filter expired entries
            now = datetime.now()
            valid_entries = {
                key: entry for key, entry in cache_data.items()
                if not entry.get("expires_at") or entry["expires_at"] > now
            }
            
            self.cache.update(valid_entries)
            self._logger.info(f"Loaded {len(valid_entries)} entries from {filepath}")
            return len(valid_entries)
        finally:
            self._release_lock()
