#!/usr/bin/env python3
"""
Smart Cache System - Advanced AI Document Processor
=================================================

Intelligent multi-level caching system with predictive prefetching and adaptive optimization.
"""

import asyncio
import time
import json
import hashlib
import pickle
import weakref
import threading
import gc
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import orjson
import msgpack
import lz4.frame
import zstandard as zstd
from collections import OrderedDict, defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    priority: int = 0  # Higher number = higher priority
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    memory_usage_mb: float = 0.0

class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size_mb: float = 1024, max_entries: int = 10000):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self.compression_enabled = True
        self.compression_threshold = 1024  # Compress entries larger than 1KB
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value with compression if beneficial."""
        try:
            # Try orjson first (fastest)
            serialized = orjson.dumps(value)
        except (TypeError, ValueError):
            try:
                # Fallback to pickle
                serialized = pickle.dumps(value)
            except Exception:
                # Last resort: string conversion
                serialized = str(value).encode('utf-8')
        
        # Compress if beneficial
        if self.compression_enabled and len(serialized) > self.compression_threshold:
            try:
                compressed = lz4.frame.compress(serialized)
                if len(compressed) < len(serialized):
                    return compressed
            except Exception:
                pass
        
        return serialized
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value with decompression if needed."""
        try:
            # Try to decompress first
            try:
                decompressed = lz4.frame.decompress(data)
                data = decompressed
            except Exception:
                pass  # Not compressed or compression failed
            
            # Try orjson first
            try:
                return orjson.loads(data)
            except (TypeError, ValueError):
                # Fallback to pickle
                return pickle.loads(data)
        except Exception:
            # Last resort: string conversion
            return data.decode('utf-8')
    
    def _calculate_size(self, entry: CacheEntry) -> int:
        """Calculate the size of a cache entry."""
        try:
            serialized = self._serialize_value(entry.value)
            return len(serialized) + len(entry.key.encode('utf-8'))
        except Exception:
            return 1024  # Default size estimate
    
    def _evict_entries(self, needed_bytes: int):
        """Evict entries to make room."""
        evicted_bytes = 0
        evicted_count = 0
        
        # Evict by priority and access time
        while (evicted_bytes < needed_bytes and 
               len(self.entries) > 0 and 
               (self.current_size_bytes > self.max_size_bytes or 
                len(self.entries) >= self.max_entries)):
            
            # Find entry to evict (lowest priority, oldest access)
            evict_key = None
            lowest_priority = float('inf')
            oldest_access = datetime.utcnow()
            
            for key, entry in self.entries.items():
                if (entry.priority < lowest_priority or 
                    (entry.priority == lowest_priority and entry.last_accessed < oldest_access)):
                    evict_key = key
                    lowest_priority = entry.priority
                    oldest_access = entry.last_accessed
            
            if evict_key:
                entry = self.entries.pop(evict_key)
                evicted_bytes += entry.size_bytes
                evicted_count += 1
                self.stats.evictions += 1
                logger.debug(f"Evicted cache entry: {evict_key}")
        
        self.current_size_bytes -= evicted_bytes
        logger.info(f"Evicted {evicted_count} entries, freed {evicted_bytes} bytes")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.entries:
                self.stats.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check TTL
            if entry.ttl_seconds:
                if datetime.utcnow() - entry.created_at > timedelta(seconds=entry.ttl_seconds):
                    del self.entries[key]
                    self.current_size_bytes -= entry.size_bytes
                    self.stats.misses += 1
                    return None
            
            # Update access info
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self.entries.move_to_end(key)
            
            self.stats.hits += 1
            self._update_hit_rate()
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            priority: int = 0, tags: List[str] = None, metadata: Dict[str, Any] = None):
        """Set value in cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self.entries:
                old_entry = self.entries[key]
                self.current_size_bytes -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl_seconds,
                priority=priority,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Calculate size
            entry.size_bytes = self._calculate_size(entry)
            
            # Check if we need to evict
            if (self.current_size_bytes + entry.size_bytes > self.max_size_bytes or 
                len(self.entries) >= self.max_entries):
                self._evict_entries(entry.size_bytes)
            
            # Add entry
            self.entries[key] = entry
            self.current_size_bytes += entry.size_bytes
            self.stats.entry_count = len(self.entries)
            self.stats.total_size_bytes = self.current_size_bytes
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self.entries:
                entry = self.entries.pop(key)
                self.current_size_bytes -= entry.size_bytes
                self.stats.entry_count = len(self.entries)
                self.stats.total_size_bytes = self.current_size_bytes
                return True
            return False
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self.entries.clear()
            self.current_size_bytes = 0
            self.stats.entry_count = 0
            self.stats.total_size_bytes = 0
    
    def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Get entries by tags."""
        with self._lock:
            result = {}
            for key, entry in self.entries.items():
                if any(tag in entry.tags for tag in tags):
                    result[key] = entry.value
            return result
    
    def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries by tags."""
        with self._lock:
            deleted_count = 0
            keys_to_delete = []
            
            for key, entry in self.entries.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                if self.delete(key):
                    deleted_count += 1
            
            return deleted_count
    
    def _update_hit_rate(self):
        """Update hit rate statistic."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            # Update memory usage
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size_bytes=self.stats.total_size_bytes,
                entry_count=self.stats.entry_count,
                hit_rate=self.stats.hit_rate,
                avg_access_time=0.0,  # Would need timing data
                memory_usage_mb=self.stats.memory_usage_mb
            )

class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: str = "./cache", max_size_mb: float = 10240):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._load_metadata()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create hash of key for filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_metadata(self):
        """Load cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.current_size_bytes = metadata.get('current_size_bytes', 0)
                    self.stats = CacheStats(**metadata.get('stats', {}))
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_metadata(self):
        """Save cache metadata."""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            metadata = {
                'current_size_bytes': self.current_size_bytes,
                'stats': {
                    'hits': self.stats.hits,
                    'misses': self.stats.misses,
                    'evictions': self.stats.evictions,
                    'total_size_bytes': self.stats.total_size_bytes,
                    'entry_count': self.stats.entry_count,
                    'hit_rate': self.stats.hit_rate,
                    'avg_access_time': self.stats.avg_access_time,
                    'memory_usage_mb': self.stats.memory_usage_mb
                }
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                self.stats.misses += 1
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                try:
                    data = lz4.frame.decompress(data)
                except Exception:
                    pass  # Not compressed
                
                # Deserialize
                try:
                    value = orjson.loads(data)
                except Exception:
                    value = pickle.loads(data)
                
                self.stats.hits += 1
                self._update_hit_rate()
                return value
                
            except Exception as e:
                logger.error(f"Failed to read cache file {file_path}: {e}")
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Set value in disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            try:
                # Serialize value
                try:
                    data = orjson.dumps(value)
                except Exception:
                    data = pickle.dumps(value)
                
                # Compress
                try:
                    data = lz4.frame.compress(data)
                except Exception:
                    pass
                
                # Write to file
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update size
                file_size = file_path.stat().st_size
                self.current_size_bytes += file_size
                self.stats.entry_count += 1
                self.stats.total_size_bytes = self.current_size_bytes
                
                # Save metadata
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Failed to write cache file {file_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.current_size_bytes -= file_size
                    self.stats.entry_count -= 1
                    self.stats.total_size_bytes = self.current_size_bytes
                    self._save_metadata()
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete cache file {file_path}: {e}")
            
            return False
    
    def clear(self):
        """Clear all disk cache entries."""
        with self._lock:
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                
                self.current_size_bytes = 0
                self.stats.entry_count = 0
                self.stats.total_size_bytes = 0
                self._save_metadata()
                
            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")
    
    def _update_hit_rate(self):
        """Update hit rate statistic."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def get_stats(self) -> CacheStats:
        """Get disk cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size_bytes=self.stats.total_size_bytes,
                entry_count=self.stats.entry_count,
                hit_rate=self.stats.hit_rate,
                avg_access_time=0.0,
                memory_usage_mb=0.0
            )

class SmartCache:
    """Intelligent multi-level cache system."""
    
    def __init__(self, memory_size_mb: float = 1024, disk_size_mb: float = 10240, 
                 cache_dir: str = "./cache"):
        self.memory_cache = MemoryCache(max_size_mb=memory_size_mb)
        self.disk_cache = DiskCache(cache_dir=cache_dir, max_size_mb=disk_size_mb)
        self.prefetch_enabled = True
        self.adaptive_ttl = True
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Start background tasks
        self._cleanup_task = None
        self._prefetch_task = None
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
        
        if self._prefetch_task is None and self.prefetch_enabled:
            self._prefetch_task = asyncio.create_task(self._prefetch_popular_entries())
    
    async def _cleanup_expired_entries(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                self._cleanup_memory_cache()
                self._cleanup_disk_cache()
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _prefetch_popular_entries(self):
        """Background task to prefetch popular entries."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                self._analyze_access_patterns()
            except Exception as e:
                logger.error(f"Cache prefetch error: {e}")
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries in memory cache."""
        with self._lock:
            current_time = datetime.utcnow()
            expired_keys = []
            
            for key, entry in self.memory_cache.entries.items():
                if entry.ttl_seconds:
                    if current_time - entry.created_at > timedelta(seconds=entry.ttl_seconds):
                        expired_keys.append(key)
            
            for key in expired_keys:
                self.memory_cache.delete(key)
    
    def _cleanup_disk_cache(self):
        """Clean up expired entries in disk cache."""
        # Implementation would check file modification times
        # and remove old entries
        pass
    
    def _analyze_access_patterns(self):
        """Analyze access patterns for prefetching."""
        # Implementation would analyze access patterns
        # and prefetch likely-to-be-accessed entries
        pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value, priority=1)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            priority: int = 0, tags: List[str] = None, metadata: Dict[str, Any] = None):
        """Set value in cache (both memory and disk)."""
        # Set in memory cache
        self.memory_cache.set(key, value, ttl_seconds, priority, tags, metadata)
        
        # Set in disk cache (for persistence)
        self.disk_cache.set(key, value, ttl_seconds)
        
        # Record access pattern
        self.access_patterns[key].append(datetime.utcnow())
    
    def delete(self, key: str) -> bool:
        """Delete entry from both caches."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key)
        return memory_deleted or disk_deleted
    
    def clear(self):
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_by_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Get entries by tags."""
        return self.memory_cache.get_by_tags(tags)
    
    def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries by tags."""
        memory_deleted = self.memory_cache.delete_by_tags(tags)
        # Note: Disk cache doesn't support tag-based operations in this implementation
        return memory_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = self.disk_cache.get_stats()
        
        return {
            'memory': {
                'hits': memory_stats.hits,
                'misses': memory_stats.misses,
                'hit_rate': memory_stats.hit_rate,
                'entry_count': memory_stats.entry_count,
                'size_mb': memory_stats.total_size_bytes / 1024 / 1024,
                'memory_usage_mb': memory_stats.memory_usage_mb
            },
            'disk': {
                'hits': disk_stats.hits,
                'misses': disk_stats.misses,
                'hit_rate': disk_stats.hit_rate,
                'entry_count': disk_stats.entry_count,
                'size_mb': disk_stats.total_size_bytes / 1024 / 1024
            },
            'combined': {
                'total_hits': memory_stats.hits + disk_stats.hits,
                'total_misses': memory_stats.misses + disk_stats.misses,
                'overall_hit_rate': (memory_stats.hits + disk_stats.hits) / 
                                  (memory_stats.hits + memory_stats.misses + 
                                   disk_stats.hits + disk_stats.misses) if 
                                  (memory_stats.hits + memory_stats.misses + 
                                   disk_stats.hits + disk_stats.misses) > 0 else 0.0
            }
        }
    
    def display_cache_dashboard(self):
        """Display cache performance dashboard."""
        stats = self.get_stats()
        
        # Memory cache table
        memory_table = Table(title="Memory Cache Statistics")
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="green")
        
        memory = stats['memory']
        memory_table.add_row("Hits", str(memory['hits']))
        memory_table.add_row("Misses", str(memory['misses']))
        memory_table.add_row("Hit Rate", f"{memory['hit_rate']:.1%}")
        memory_table.add_row("Entries", str(memory['entry_count']))
        memory_table.add_row("Size", f"{memory['size_mb']:.1f} MB")
        memory_table.add_row("Memory Usage", f"{memory['memory_usage_mb']:.1f} MB")
        
        console.print(memory_table)
        
        # Disk cache table
        disk_table = Table(title="Disk Cache Statistics")
        disk_table.add_column("Metric", style="cyan")
        disk_table.add_column("Value", style="green")
        
        disk = stats['disk']
        disk_table.add_row("Hits", str(disk['hits']))
        disk_table.add_row("Misses", str(disk['misses']))
        disk_table.add_row("Hit Rate", f"{disk['hit_rate']:.1%}")
        disk_table.add_row("Entries", str(disk['entry_count']))
        disk_table.add_row("Size", f"{disk['size_mb']:.1f} MB")
        
        console.print(disk_table)
        
        # Combined statistics
        combined_table = Table(title="Combined Cache Statistics")
        combined_table.add_column("Metric", style="cyan")
        combined_table.add_column("Value", style="green")
        
        combined = stats['combined']
        combined_table.add_row("Total Hits", str(combined['total_hits']))
        combined_table.add_row("Total Misses", str(combined['total_misses']))
        combined_table.add_row("Overall Hit Rate", f"{combined['overall_hit_rate']:.1%}")
        
        console.print(combined_table)
    
    def cleanup(self):
        """Cleanup cache resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._prefetch_task:
            self._prefetch_task.cancel()
        
        self.memory_cache.clear()
        self.disk_cache.clear()
        
        logger.info("Smart cache cleanup completed")

# Global smart cache instance
smart_cache = SmartCache()

# Utility functions
def get_cache(key: str) -> Optional[Any]:
    """Get value from global smart cache."""
    return smart_cache.get(key)

def set_cache(key: str, value: Any, ttl_seconds: Optional[int] = None, 
              priority: int = 0, tags: List[str] = None, metadata: Dict[str, Any] = None):
    """Set value in global smart cache."""
    smart_cache.set(key, value, ttl_seconds, priority, tags, metadata)

def delete_cache(key: str) -> bool:
    """Delete entry from global smart cache."""
    return smart_cache.delete(key)

def clear_cache():
    """Clear global smart cache."""
    smart_cache.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get global smart cache statistics."""
    return smart_cache.get_stats()

def display_cache_dashboard():
    """Display global smart cache dashboard."""
    smart_cache.display_cache_dashboard()

def cleanup_smart_cache():
    """Cleanup global smart cache."""
    smart_cache.cleanup()

# Cache decorators
def cached(ttl_seconds: Optional[int] = None, priority: int = 0, 
           tags: List[str] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = get_cache(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            set_cache(cache_key, result, ttl_seconds, priority, tags)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = get_cache(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            set_cache(cache_key, result, ttl_seconds, priority, tags)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

if __name__ == "__main__":
    # Example usage
    async def main():
        # Test cache operations
        set_cache("test_key", {"message": "Hello, World!"}, ttl_seconds=300)
        
        result = get_cache("test_key")
        print(f"Cached result: {result}")
        
        # Display dashboard
        display_cache_dashboard()
        
        # Cleanup
        cleanup_smart_cache()
    
    asyncio.run(main())
















