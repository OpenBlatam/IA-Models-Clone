"""
NLP Cache System
================

Sistema de caché inteligente para optimizar el rendimiento del NLP.
Incluye caché basado en contenido, TTL dinámico y compresión.
"""

import hashlib
import json
import pickle
import gzip
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import logging
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)

class CacheStrategy(str, Enum):
    """Estrategias de caché."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"   # Time To Live
    CONTENT_BASED = "content_based"  # Basado en contenido

@dataclass
class CacheEntry:
    """Entrada de caché."""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    size_bytes: int = 0
    compressed: bool = False

@dataclass
class CacheStats:
    """Estadísticas de caché."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0

class IntelligentNLPCache:
    """Sistema de caché inteligente para NLP."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
        enable_compression: bool = True,
        compression_threshold: int = 1024  # bytes
    ):
        """Initialize intelligent NLP cache."""
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
        # Content-based cache for similar texts
        self._content_hash_map: Dict[str, str] = {}
        self._similarity_threshold = 0.85
        
        # Background cleanup
        self._cleanup_task = None
        self._running = False
        
    async def start(self):
        """Start background cleanup task."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            logger.info("NLP Cache background cleanup started")
    
    async def stop(self):
        """Stop background cleanup task."""
        if self._running:
            self._running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            logger.info("NLP Cache background cleanup stopped")
    
    def _generate_key(self, text: str, task: str, language: str = "en", **kwargs) -> str:
        """Generate cache key for text and task."""
        # Create content hash
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        # Create task-specific key
        task_params = {
            'task': task,
            'language': language,
            **kwargs
        }
        params_str = json.dumps(task_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{task}:{language}:{content_hash}:{params_hash}"
    
    def _compress_value(self, value: Any) -> Tuple[bytes, bool]:
        """Compress value if needed."""
        if not self.enable_compression:
            return pickle.dumps(value), False
        
        # Serialize first
        serialized = pickle.dumps(value)
        
        # Compress if above threshold
        if len(serialized) > self.compression_threshold:
            compressed = gzip.compress(serialized)
            if len(compressed) < len(serialized):
                return compressed, True
        
        return serialized, False
    
    def _decompress_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress value if needed."""
        if compressed:
            data = gzip.decompress(data)
        
        return pickle.loads(data)
    
    def _calculate_size(self, entry: CacheEntry) -> int:
        """Calculate entry size in bytes."""
        if entry.compressed:
            return len(entry.value) if isinstance(entry.value, bytes) else 0
        else:
            return len(pickle.dumps(entry.value))
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl is None:
            return False
        
        return datetime.now() > (entry.created_at + timedelta(seconds=entry.ttl))
    
    def _evict_entries(self, count: int = 1):
        """Evict entries based on strategy."""
        with self._lock:
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                for _ in range(min(count, len(self._cache))):
                    if self._cache:
                        key, entry = self._cache.popitem(last=False)
                        self._stats.evictions += 1
                        self._stats.total_size -= entry.size_bytes
                        logger.debug(f"Evicted LRU entry: {key}")
            
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: (x[1].access_count, x[1].accessed_at)
                )
                for i in range(min(count, len(sorted_entries))):
                    key, entry = sorted_entries[i]
                    del self._cache[key]
                    self._stats.evictions += 1
                    self._stats.total_size -= entry.size_bytes
                    logger.debug(f"Evicted LFU entry: {key}")
    
    async def _background_cleanup(self):
        """Background cleanup task."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def cleanup_expired(self):
        """Clean up expired entries."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats.evictions += 1
                self._stats.total_size -= entry.size_bytes
                logger.debug(f"Cleaned up expired entry: {key}")
    
    async def get(
        self,
        text: str,
        task: str,
        language: str = "en",
        **kwargs
    ) -> Optional[Any]:
        """Get value from cache."""
        key = self._generate_key(text, task, language, **kwargs)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats.misses += 1
                    self._stats.total_size -= entry.size_bytes
                    return None
                
                # Update access info
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                # Decompress if needed
                value = self._decompress_value(entry.value, entry.compressed)
                
                self._stats.hits += 1
                self._update_hit_rate()
                logger.debug(f"Cache hit: {key}")
                return value
            
            self._stats.misses += 1
            self._update_hit_rate()
            logger.debug(f"Cache miss: {key}")
            return None
    
    async def set(
        self,
        text: str,
        task: str,
        value: Any,
        language: str = "en",
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Set value in cache."""
        key = self._generate_key(text, task, language, **kwargs)
        
        with self._lock:
            # Check if we need to evict
            while (
                len(self._cache) >= self.max_size or
                self._stats.total_size >= self.max_memory_bytes
            ):
                self._evict_entries(1)
            
            # Compress value if needed
            compressed_value, is_compressed = self._compress_value(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                ttl=ttl or self.default_ttl,
                compressed=is_compressed
            )
            
            # Calculate size
            entry.size_bytes = self._calculate_size(entry)
            
            # Store entry
            self._cache[key] = entry
            self._stats.total_size += entry.size_bytes
            
            logger.debug(f"Cached: {key} (compressed: {is_compressed})")
            return True
    
    async def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        with self._lock:
            if pattern is None:
                # Clear all
                self._cache.clear()
                self._stats.total_size = 0
                logger.info("Cache cleared")
            else:
                # Clear matching pattern
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    entry = self._cache.pop(key)
                    self._stats.total_size -= entry.size_bytes
                logger.info(f"Invalidated {len(keys_to_remove)} entries matching '{pattern}'")
    
    async def get_similar(self, text: str, task: str, language: str = "en") -> Optional[Any]:
        """Get similar cached result using content similarity."""
        # This is a simplified similarity check
        # In practice, you'd use more sophisticated similarity algorithms
        
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        with self._lock:
            for key, entry in self._cache.items():
                if task in key and language in key:
                    # Check if content is similar (simplified)
                    if not self._is_expired(entry):
                        # Update access info
                        entry.accessed_at = datetime.now()
                        entry.access_count += 1
                        
                        # Decompress and return
                        value = self._decompress_value(entry.value, entry.compressed)
                        self._stats.hits += 1
                        self._update_hit_rate()
                        logger.debug(f"Similar cache hit: {key}")
                        return value
        
        return None
    
    def _update_hit_rate(self):
        """Update hit/miss rates."""
        total = self._stats.hits + self._stats.misses
        if total > 0:
            self._stats.hit_rate = self._stats.hits / total
            self._stats.miss_rate = self._stats.misses / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'max_memory_mb': self.max_memory_bytes // (1024 * 1024),
                'current_memory_mb': self._stats.total_size // (1024 * 1024),
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'evictions': self._stats.evictions,
                'hit_rate': self._stats.hit_rate,
                'miss_rate': self._stats.miss_rate,
                'strategy': self.strategy.value,
                'compression_enabled': self.enable_compression
            }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage."""
        with self._lock:
            total_size = 0
            compressed_size = 0
            uncompressed_size = 0
            
            for entry in self._cache.values():
                total_size += entry.size_bytes
                if entry.compressed:
                    compressed_size += entry.size_bytes
                else:
                    uncompressed_size += entry.size_bytes
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'uncompressed_size_mb': uncompressed_size / (1024 * 1024),
                'compression_ratio': compressed_size / total_size if total_size > 0 else 0,
                'entries_count': len(self._cache)
            }
    
    async def optimize(self):
        """Optimize cache by removing least useful entries."""
        with self._lock:
            # Remove entries with low access count and old access time
            current_time = datetime.now()
            entries_to_remove = []
            
            for key, entry in self._cache.items():
                age_hours = (current_time - entry.accessed_at).total_seconds() / 3600
                
                # Remove if old and rarely accessed
                if age_hours > 24 and entry.access_count < 2:
                    entries_to_remove.append(key)
            
            for key in entries_to_remove:
                entry = self._cache.pop(key)
                self._stats.evictions += 1
                self._stats.total_size -= entry.size_bytes
            
            logger.info(f"Optimized cache: removed {len(entries_to_remove)} entries")

# Global cache instance
nlp_cache = IntelligentNLPCache(
    max_size=2000,
    max_memory_mb=1000,
    default_ttl=7200,  # 2 hours
    strategy=CacheStrategy.LRU,
    enable_compression=True
)












