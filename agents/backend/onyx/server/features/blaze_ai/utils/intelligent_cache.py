"""
Blaze AI Intelligent Cache System v7.1.0

Advanced caching system with multiple strategies, automatic eviction,
performance optimization, and intelligent memory management.
"""

import asyncio
import hashlib
import json
import logging
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import pickle
import zlib
import lz4.frame
import snappy

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = auto()           # Least Recently Used
    LFU = auto()           # Least Frequently Used
    FIFO = auto()          # First In, First Out
    LIFO = auto()          # Last In, First Out
    RANDOM = auto()        # Random eviction
    TTL = auto()           # Time To Live based
    SIZE = auto()          # Size based eviction
    HYBRID = auto()        # Combination of strategies

class CompressionType(Enum):
    """Data compression types."""
    NONE = auto()
    ZLIB = auto()
    LZ4 = auto()
    SNAPPY = auto()
    PICKLE = auto()

class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1 = auto()    # Fastest, smallest (memory)
    L2 = auto()    # Medium speed, medium size (SSD)
    L3 = auto()    # Slower, largest (HDD/Network)

# Default constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL = 3600  # 1 hour
DEFAULT_COMPRESSION = CompressionType.LZ4
DEFAULT_STRATEGY = CacheStrategy.LRU

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    compression_type: CompressionType = CompressionType.NONE
    ttl: Optional[float] = None
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """Get time since last access."""
        return time.time() - self.accessed_at

@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    compression_ratio: float = 1.0
    avg_access_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / max(total_requests, 1)
    
    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hit_count + self.miss_count

# ============================================================================
# COMPRESSION UTILITIES
# ============================================================================

class CompressionManager:
    """Manages data compression and decompression."""
    
    @staticmethod
    def compress(data: Any, compression_type: CompressionType) -> Tuple[bytes, CompressionType]:
        """Compress data using specified method."""
        try:
            if compression_type == CompressionType.NONE:
                return pickle.dumps(data), CompressionType.NONE
            
            serialized = pickle.dumps(data)
            
            if compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(serialized, level=9)
            elif compression_type == CompressionType.LZ4:
                compressed = lz4.frame.compress(serialized)
            elif compression_type == CompressionType.SNAPPY:
                compressed = snappy.compress(serialized)
            else:
                return serialized, CompressionType.NONE
            
            # Only use compression if it actually saves space
            if len(compressed) < len(serialized):
                return compressed, compression_type
            else:
                return serialized, CompressionType.NONE
                
        except Exception as e:
            logger.warning(f"Compression failed, falling back to pickle: {e}")
            return pickle.dumps(data), CompressionType.NONE
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> Any:
        """Decompress data using specified method."""
        try:
            if compression_type == CompressionType.NONE:
                return pickle.loads(data)
            
            if compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(data)
            elif compression_type == CompressionType.LZ4:
                decompressed = lz4.frame.decompress(data)
            elif compression_type == CompressionType.SNAPPY:
                decompressed = snappy.decompress(data)
            else:
                return pickle.loads(data)
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

# ============================================================================
# EVICTION STRATEGIES
# ============================================================================

class EvictionStrategy(ABC):
    """Abstract base class for cache eviction strategies."""
    
    @abstractmethod
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select keys for eviction."""
        pass

class LRUEvictionStrategy(EvictionStrategy):
    """Least Recently Used eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select least recently used entries."""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].accessed_at
        )
        return [key for key, _ in sorted_entries[:count]]

class LFUEvictionStrategy(EvictionStrategy):
    """Least Frequently Used eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select least frequently used entries."""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].access_count
        )
        return [key for key, _ in sorted_entries[:count]]

class FIFOEvictionStrategy(EvictionStrategy):
    """First In, First Out eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select oldest entries."""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].created_at
        )
        return [key for key, _ in sorted_entries[:count]]

class TTLBasedEvictionStrategy(EvictionStrategy):
    """Time To Live based eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select expired or oldest entries."""
        # First, remove expired entries
        expired_keys = [key for key, entry in entries.items() if entry.is_expired()]
        
        if len(expired_keys) >= count:
            return expired_keys[:count]
        
        # If not enough expired entries, add oldest ones
        remaining_count = count - len(expired_keys)
        sorted_entries = sorted(
            [(k, v) for k, v in entries.items() if k not in expired_keys],
            key=lambda x: x[1].created_at
        )
        
        return expired_keys + [key for key, _ in sorted_entries[:remaining_count]]

class SizeBasedEvictionStrategy(EvictionStrategy):
    """Size based eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select largest entries."""
        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].size_bytes,
            reverse=True
        )
        return [key for key, _ in sorted_entries[:count]]

class HybridEvictionStrategy(EvictionStrategy):
    """Hybrid eviction strategy combining multiple approaches."""
    
    def __init__(self, strategies: List[Tuple[EvictionStrategy, float]]):
        """
        Initialize with strategies and their weights.
        
        Args:
            strategies: List of (strategy, weight) tuples
        """
        self.strategies = strategies
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry], count: int = 1) -> List[str]:
        """Select entries using weighted combination of strategies."""
        if not self.strategies:
            return []
        
        # Get eviction candidates from each strategy
        all_candidates = []
        for strategy, weight in self.strategies:
            candidates = strategy.select_for_eviction(entries, count * 2)  # Get more candidates
            all_candidates.extend([(key, weight) for key in candidates])
        
        # Count occurrences and apply weights
        key_scores = {}
        for key, weight in all_candidates:
            if key not in key_scores:
                key_scores[key] = 0
            key_scores[key] += weight
        
        # Sort by score and return top candidates
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        return [key for key, _ in sorted_keys[:count]]

# ============================================================================
# MAIN CACHE CLASS
# ============================================================================

class IntelligentCache:
    """Advanced intelligent cache system."""
    
    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        strategy: CacheStrategy = DEFAULT_STRATEGY,
        compression: CompressionType = DEFAULT_COMPRESSION,
        ttl: Optional[float] = DEFAULT_TTL,
        enable_stats: bool = True,
        auto_cleanup: bool = True,
        cleanup_interval: float = 60.0
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.compression = compression
        self.ttl = ttl
        self.enable_stats = enable_stats
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # Cache storage
        self._entries: Dict[str, CacheEntry] = {}
        self._tags: Dict[str, Set[str]] = {}  # tag -> set of keys
        
        # Statistics
        self._stats = CacheStats()
        self._access_times: List[float] = []
        
        # Eviction strategy
        self._eviction_strategy = self._create_eviction_strategy()
        
        # Compression manager
        self._compression_manager = CompressionManager()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Start background cleanup if enabled
        if self.auto_cleanup:
            self._start_cleanup_task()
    
    def _create_eviction_strategy(self) -> EvictionStrategy:
        """Create eviction strategy based on configuration."""
        if self.strategy == CacheStrategy.LRU:
            return LRUEvictionStrategy()
        elif self.strategy == CacheStrategy.LFU:
            return LFUEvictionStrategy()
        elif self.strategy == CacheStrategy.FIFO:
            return FIFOEvictionStrategy()
        elif self.strategy == CacheStrategy.TTL:
            return TTLBasedEvictionStrategy()
        elif self.strategy == CacheStrategy.SIZE:
            return SizeBasedEvictionStrategy()
        elif self.strategy == CacheStrategy.HYBRID:
            # Create hybrid strategy with LRU and TTL
            return HybridEvictionStrategy([
                (LRUEvictionStrategy(), 0.6),
                (TTLBasedEvictionStrategy(), 0.4)
            ])
        else:
            return LRUEvictionStrategy()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_expired_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    await asyncio.sleep(5.0)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                await self._remove_entry(key, reason="expired")
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        start_time = time.perf_counter()
        
        async with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self._record_miss()
                return default
            
            if entry.is_expired():
                await self._remove_entry(key, reason="expired")
                self._record_miss()
                return default
            
            # Update access statistics
            entry.update_access()
            self._record_hit(start_time)
            
            # Decompress if needed
            if entry.compression_type != CompressionType.NONE:
                try:
                    return self._compression_manager.decompress(entry.value, entry.compression_type)
                except Exception as e:
                    logger.error(f"Decompression failed for key {key}: {e}")
                    await self._remove_entry(key, reason="decompression_error")
                    return default
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        compress: Optional[bool] = None
    ) -> bool:
        """Set value in cache."""
        async with self._lock:
            try:
                # Determine compression
                should_compress = compress if compress is not None else (self.compression != CompressionType.NONE)
                
                # Compress data if needed
                if should_compress:
                    compressed_data, actual_compression = self._compression_manager.compress(value, self.compression)
                else:
                    compressed_data, actual_compression = value, CompressionType.NONE
                
                # Calculate size
                size_bytes = len(compressed_data) if isinstance(compressed_data, bytes) else len(str(compressed_data))
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_data,
                    ttl=ttl or self.ttl,
                    compression_type=actual_compression,
                    size_bytes=size_bytes,
                    tags=tags or set()
                )
                
                # Check if we need to evict entries
                if len(self._entries) >= self.max_size:
                    await self._evict_entries(1)
                
                # Store entry
                self._entries[key] = entry
                
                # Update tags index
                for tag in entry.tags:
                    if tag not in self._tags:
                        self._tags[tag] = set()
                    self._tags[tag].add(key)
                
                # Update statistics
                self._stats.total_entries = len(self._entries)
                self._stats.total_size_bytes += size_bytes
                
                # Update compression ratio
                if actual_compression != CompressionType.NONE:
                    original_size = len(pickle.dumps(value))
                    self._stats.compression_ratio = (
                        (self._stats.compression_ratio * (self._stats.total_entries - 1) + 
                         original_size / max(size_bytes, 1)) / self._stats.total_entries
                    )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache entry {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return await self._remove_entry(key, reason="manual_delete")
    
    async def clear(self, tags: Optional[Set[str]] = None):
        """Clear cache entries, optionally by tags."""
        async with self._lock:
            if tags:
                # Clear entries with specific tags
                keys_to_remove = set()
                for tag in tags:
                    if tag in self._tags:
                        keys_to_remove.update(self._tags[tag])
                        del self._tags[tag]
                
                for key in keys_to_remove:
                    await self._remove_entry(key, reason="tag_clear")
            else:
                # Clear all entries
                for key in list(self._entries.keys()):
                    await self._remove_entry(key, reason="clear_all")
    
    async def get_or_set(
        self,
        key: str,
        default_factory: Callable,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> Any:
        """Get value from cache or set using factory function."""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Create default value
        if asyncio.iscoroutinefunction(default_factory):
            value = await default_factory()
        else:
            value = default_factory()
        
        # Store in cache
        await self.set(key, value, ttl=ttl, tags=tags)
        return value
    
    async def _remove_entry(self, key: str, reason: str = "unknown") -> bool:
        """Remove entry from cache."""
        if key not in self._entries:
            return False
        
        entry = self._entries[key]
        
        # Remove from tags index
        for tag in entry.tags:
            if tag in self._tags:
                self._tags[tag].discard(key)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        # Update statistics
        self._stats.total_size_bytes -= entry.size_bytes
        self._stats.eviction_count += 1
        
        # Remove entry
        del self._entries[key]
        self._stats.total_entries = len(self._entries)
        
        logger.debug(f"Removed cache entry {key} (reason: {reason})")
        return True
    
    async def _evict_entries(self, count: int):
        """Evict entries using the configured strategy."""
        if not self._entries:
            return
        
        keys_to_evict = self._eviction_strategy.select_for_eviction(self._entries, count)
        
        for key in keys_to_evict:
            await self._remove_entry(key, reason="eviction")
    
    def _record_hit(self, start_time: float):
        """Record cache hit."""
        self._stats.hit_count += 1
        access_time = time.perf_counter() - start_time
        self._access_times.append(access_time)
        
        # Keep only recent access times for average calculation
        if len(self._access_times) > 1000:
            self._access_times = self._access_times[-1000:]
        
        self._stats.avg_access_time = sum(self._access_times) / len(self._access_times)
    
    def _record_miss(self):
        """Record cache miss."""
        self._stats.miss_count += 1
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Update memory usage
        try:
            import psutil
            process = psutil.Process()
            self._stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return self._stats
    
    def get_keys_by_tag(self, tag: str) -> Set[str]:
        """Get all keys with a specific tag."""
        return self._tags.get(tag, set()).copy()
    
    def get_tags(self) -> Set[str]:
        """Get all tags in use."""
        return set(self._tags.keys())
    
    async def shutdown(self):
        """Shutdown the cache system."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear all entries
        await self.clear()

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_cache(
    max_size: int = DEFAULT_CACHE_SIZE,
    strategy: CacheStrategy = DEFAULT_STRATEGY,
    compression: CompressionType = DEFAULT_COMPRESSION,
    ttl: Optional[float] = DEFAULT_TTL
) -> IntelligentCache:
    """Create a new cache instance."""
    return IntelligentCache(
        max_size=max_size,
        strategy=strategy,
        compression=compression,
        ttl=ttl
    )

def create_memory_optimized_cache() -> IntelligentCache:
    """Create a memory-optimized cache."""
    return IntelligentCache(
        max_size=500,
        strategy=CacheStrategy.LRU,
        compression=CompressionType.LZ4,
        ttl=1800,  # 30 minutes
        auto_cleanup=True,
        cleanup_interval=30.0
    )

def create_performance_cache() -> IntelligentCache:
    """Create a performance-optimized cache."""
    return IntelligentCache(
        max_size=2000,
        strategy=CacheStrategy.HYBRID,
        compression=CompressionType.SNAPPY,
        ttl=7200,  # 2 hours
        auto_cleanup=True,
        cleanup_interval=60.0
    )

def create_persistent_cache() -> IntelligentCache:
    """Create a persistent cache with long TTL."""
    return IntelligentCache(
        max_size=5000,
        strategy=CacheStrategy.LFU,
        compression=CompressionType.ZLIB,
        ttl=86400,  # 24 hours
        auto_cleanup=True,
        cleanup_interval=300.0
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "CacheStrategy",
    "CompressionType", 
    "CacheLevel",
    "CacheEntry",
    "CacheStats",
    "EvictionStrategy",
    "LRUEvictionStrategy",
    "LFUEvictionStrategy",
    "FIFOEvictionStrategy",
    "TTLBasedEvictionStrategy",
    "SizeBasedEvictionStrategy",
    "HybridEvictionStrategy",
    "CompressionManager",
    "IntelligentCache",
    "create_cache",
    "create_memory_optimized_cache",
    "create_performance_cache",
    "create_persistent_cache"
]
