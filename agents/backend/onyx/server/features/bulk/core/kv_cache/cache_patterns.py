"""
Cache patterns implementation for KV cache.

This module provides common cache patterns like cache-aside,
write-through, write-back, and refresh-ahead.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class CachePattern(Enum):
    """Cache patterns."""
    CACHE_ASIDE = "cache_aside"  # Application manages cache
    WRITE_THROUGH = "write_through"  # Write to cache and storage simultaneously
    WRITE_BACK = "write_back"  # Write to cache first, flush to storage later
    REFRESH_AHEAD = "refresh_ahead"  # Refresh before expiration
    READ_THROUGH = "read_through"  # Cache automatically loads from storage


@dataclass
class PatternConfig:
    """Configuration for cache pattern."""
    pattern: CachePattern
    storage_backend: Optional[Callable] = None
    write_delay: float = 0.0  # For write-back pattern
    refresh_ahead_ratio: float = 0.8  # Refresh when 80% of TTL has passed


class CacheAside:
    """Cache-aside pattern implementation."""
    
    def __init__(self, cache: Any, storage_loader: Callable[[str], Any]):
        self.cache = cache
        self.storage_loader = storage_loader
        
    def get(self, key: str) -> Any:
        """Get from cache, load from storage if miss."""
        value = self.cache.get(key)
        if value is None:
            # Cache miss - load from storage
            value = self.storage_loader(key)
            if value is not None:
                self.cache.put(key, value)
        return value
        
    def put(self, key: str, value: Any) -> bool:
        """Put in cache only (application manages storage)."""
        return self.cache.put(key, value)
        
    def delete(self, key: str) -> bool:
        """Delete from cache."""
        return self.cache.delete(key)


class WriteThrough:
    """Write-through pattern implementation."""
    
    def __init__(
        self,
        cache: Any,
        storage_writer: Callable[[str, Any], bool],
        storage_loader: Optional[Callable[[str], Any]] = None
    ):
        self.cache = cache
        self.storage_writer = storage_writer
        self.storage_loader = storage_loader
        
    def get(self, key: str) -> Any:
        """Get from cache, load from storage if miss."""
        value = self.cache.get(key)
        if value is None and self.storage_loader:
            value = self.storage_loader(key)
            if value is not None:
                self.cache.put(key, value)
        return value
        
    def put(self, key: str, value: Any) -> bool:
        """Write to both cache and storage."""
        # Write to storage first
        storage_result = self.storage_writer(key, value)
        if storage_result:
            # Then write to cache
            return self.cache.put(key, value)
        return False
        
    def delete(self, key: str) -> bool:
        """Delete from both cache and storage."""
        # Delete from storage first
        storage_result = self.storage_writer(key, None)  # None indicates delete
        if storage_result:
            return self.cache.delete(key)
        return False


class WriteBack:
    """Write-back pattern implementation."""
    
    def __init__(
        self,
        cache: Any,
        storage_writer: Callable[[str, Any], bool],
        flush_interval: float = 60.0
    ):
        self.cache = cache
        self.storage_writer = storage_writer
        self.flush_interval = flush_interval
        self._dirty_keys: set = set()
        self._lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        self.start_flush_thread()
        
    def start_flush_thread(self) -> None:
        """Start background flush thread."""
        if self._running:
            return
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
    def stop_flush_thread(self) -> None:
        """Stop background flush thread."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            
    def _flush_loop(self) -> None:
        """Flush dirty keys to storage."""
        while self._running:
            try:
                self.flush_all()
                time.sleep(self.flush_interval)
            except Exception as e:
                print(f"Error in flush loop: {e}")
                
    def get(self, key: str) -> Any:
        """Get from cache."""
        return self.cache.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Write to cache only, mark as dirty."""
        result = self.cache.put(key, value)
        if result:
            with self._lock:
                self._dirty_keys.add(key)
        return result
        
    def delete(self, key: str) -> bool:
        """Delete from cache, mark as dirty."""
        result = self.cache.delete(key)
        if result:
            with self._lock:
                self._dirty_keys.add(key)
        return result
        
    def flush(self, key: str) -> bool:
        """Flush a specific key to storage."""
        value = self.cache.get(key)
        if value is None:
            return False
            
        result = self.storage_writer(key, value)
        if result:
            with self._lock:
                self._dirty_keys.discard(key)
        return result
        
    def flush_all(self) -> int:
        """Flush all dirty keys to storage."""
        flushed_count = 0
        
        with self._lock:
            keys_to_flush = list(self._dirty_keys)
            
        for key in keys_to_flush:
            if self.flush(key):
                flushed_count += 1
                
        return flushed_count
        
    def get_dirty_keys(self) -> List[str]:
        """Get list of dirty keys."""
        with self._lock:
            return list(self._dirty_keys)


class RefreshAhead:
    """Refresh-ahead pattern implementation."""
    
    def __init__(
        self,
        cache: Any,
        loader: Callable[[str], Any],
        refresh_ratio: float = 0.8
    ):
        self.cache = cache
        self.loader = loader
        self.refresh_ratio = refresh_ratio
        self._entry_timestamps: Dict[str, float] = {}
        self._ttls: Dict[str, float] = {}
        self._lock = threading.Lock()
        
    def get(self, key: str, ttl: float = 3600.0) -> Any:
        """Get from cache, refresh if needed."""
        value = self.cache.get(key)
        
        if value is not None:
            with self._lock:
                if key in self._entry_timestamps:
                    age = time.time() - self._entry_timestamps[key]
                    key_ttl = self._ttls.get(key, ttl)
                    
                    # Refresh if past refresh threshold
                    if age > (key_ttl * self.refresh_ratio):
                        # Refresh in background
                        self._refresh_in_background(key, key_ttl)
                        
            return value
        else:
            # Cache miss - load and cache
            value = self.loader(key)
            if value is not None:
                self.cache.put(key, value)
                with self._lock:
                    self._entry_timestamps[key] = time.time()
                    self._ttls[key] = ttl
            return value
            
    def put(self, key: str, value: Any, ttl: float = 3600.0) -> bool:
        """Put in cache."""
        result = self.cache.put(key, value)
        if result:
            with self._lock:
                self._entry_timestamps[key] = time.time()
                self._ttls[key] = ttl
        return result
        
    def _refresh_in_background(self, key: str, ttl: float) -> None:
        """Refresh key in background."""
        def refresh():
            new_value = self.loader(key)
            if new_value is not None:
                self.cache.put(key, new_value)
                with self._lock:
                    self._entry_timestamps[key] = time.time()
                    
        threading.Thread(target=refresh, daemon=True).start()


class PatternCacheFactory:
    """Factory for creating cache pattern implementations."""
    
    @staticmethod
    def create(
        cache: Any,
        pattern: CachePattern,
        config: PatternConfig
    ) -> Any:
        """Create cache pattern implementation."""
        if pattern == CachePattern.CACHE_ASIDE:
            if not config.storage_backend:
                raise ValueError("storage_backend required for cache-aside")
            return CacheAside(cache, config.storage_backend)
            
        elif pattern == CachePattern.WRITE_THROUGH:
            if not config.storage_backend:
                raise ValueError("storage_backend required for write-through")
            return WriteThrough(cache, config.storage_backend, config.storage_backend)
            
        elif pattern == CachePattern.WRITE_BACK:
            if not config.storage_backend:
                raise ValueError("storage_backend required for write-back")
            return WriteBack(cache, config.storage_backend, config.write_delay)
            
        elif pattern == CachePattern.REFRESH_AHEAD:
            if not config.storage_backend:
                raise ValueError("storage_backend required for refresh-ahead")
            return RefreshAhead(cache, config.storage_backend, config.refresh_ahead_ratio)
            
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")














