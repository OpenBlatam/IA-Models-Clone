"""
Advanced caching system for Blaze AI with Redis support, compression, encryption, and intelligent eviction.

This module provides high-performance caching with multiple backends, compression algorithms,
encryption, and sophisticated eviction strategies for optimal memory usage and performance.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from collections import OrderedDict
import threading
import weakref

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import cryptography.fernet as fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..core.interfaces import CoreConfig
from .logging import get_logger

# =============================================================================
# Core Types and Enums
# =============================================================================

class CacheBackend(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    TTL = "ttl"
    HYBRID = "hybrid"

@dataclass
class CacheConfig:
    """Cache configuration."""
    backend: CacheBackend = CacheBackend.HYBRID
    max_size: int = 10000
    ttl: float = 3600.0  # 1 hour default
    compression: CompressionType = CompressionType.GZIP
    encryption: bool = False
    encryption_key: Optional[str] = None
    eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    enable_stats: bool = True
    compression_threshold: int = 1024  # Only compress items larger than this
    batch_size: int = 100
    enable_persistence: bool = False
    persistence_file: str = "cache_backup.pkl"

# =============================================================================
# Cache Entry and Statistics
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size: int = 0
    compressed: bool = False
    encrypted: bool = False
    ttl: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl <= 0:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    compression_ratio: float = 1.0
    memory_usage: int = 0
    redis_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def total_operations(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes

# =============================================================================
# Compression and Encryption Utilities
# =============================================================================

class CompressionManager:
    """Manages data compression and decompression."""
    
    def __init__(self, compression_type: CompressionType):
        self.compression_type = compression_type
        self.logger = get_logger("compression_manager")
    
    def compress(self, data: bytes) -> tuple[bytes, bool]:
        """Compress data using specified algorithm."""
        if self.compression_type == CompressionType.NONE:
            return data, False
        
        try:
            if self.compression_type == CompressionType.GZIP:
                compressed = zlib.compress(data, level=6)
            elif self.compression_type == CompressionType.LZ4:
                try:
                    import lz4.frame
                    compressed = lz4.frame.compress(data)
                except ImportError:
                    self.logger.warning("LZ4 not available, falling back to GZIP")
                    compressed = zlib.compress(data, level=6)
            elif self.compression_type == CompressionType.ZSTD:
                try:
                    import zstandard as zstd
                    compressor = zstd.ZstdCompressor(level=3)
                    compressed = compressor.compress(data)
                except ImportError:
                    self.logger.warning("ZSTD not available, falling back to GZIP")
                    compressed = zlib.compress(data, level=6)
            else:
                return data, False
            
            # Only use compression if it actually saves space
            if len(compressed) < len(data):
                return compressed, True
            return data, False
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return data, False
    
    def decompress(self, data: bytes, was_compressed: bool) -> bytes:
        """Decompress data."""
        if not was_compressed:
            return data
        
        try:
            if self.compression_type == CompressionType.GZIP:
                return zlib.decompress(data)
            elif self.compression_type == CompressionType.LZ4:
                try:
                    import lz4.frame
                    return lz4.frame.decompress(data)
                except ImportError:
                    return zlib.decompress(data)
            elif self.compression_type == CompressionType.ZSTD:
                try:
                    import zstandard as zstd
                    decompressor = zstd.ZstdDecompressor()
                    return decompressor.decompress(data)
                except ImportError:
                    return zlib.decompress(data)
            else:
                return data
                
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return data

class EncryptionManager:
    """Manages data encryption and decryption."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key
        self.fernet = None
        self.logger = get_logger("encryption_manager")
        
        if encryption_key and CRYPTO_AVAILABLE:
            self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption with the provided key."""
        try:
            # Derive a key from the provided encryption key
            salt = b'blaze_ai_cache_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(self.encryption_key.encode())
            self.fernet = fernet.Fernet(fernet.Fernet.generate_key())
        except Exception as e:
            self.logger.error(f"Failed to setup encryption: {e}")
            self.fernet = None
    
    def encrypt(self, data: bytes) -> tuple[bytes, bool]:
        """Encrypt data."""
        if not self.fernet:
            return data, False
        
        try:
            encrypted = self.fernet.encrypt(data)
            return encrypted, True
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data, False
    
    def decrypt(self, data: bytes, was_encrypted: bool) -> bytes:
        """Decrypt data."""
        if not was_encrypted or not self.fernet:
            return data
        
        try:
            return self.fernet.decrypt(data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return data

# =============================================================================
# Eviction Strategies
# =============================================================================

class EvictionStrategy(ABC):
    """Abstract base class for eviction strategies."""
    
    @abstractmethod
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select a key for eviction."""
        pass

class LRUEvictionStrategy(EvictionStrategy):
    """Least Recently Used eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select least recently used entry."""
        if not entries:
            return None
        
        oldest_key = min(entries.keys(), key=lambda k: entries[k].accessed_at)
        return oldest_key

class LFUEvictionStrategy(EvictionStrategy):
    """Least Frequently Used eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select least frequently used entry."""
        if not entries:
            return None
        
        least_frequent_key = min(entries.keys(), key=lambda k: entries[k].access_count)
        return least_frequent_key

class FIFOEvictionStrategy(EvictionStrategy):
    """First In First Out eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select oldest entry by creation time."""
        if not entries:
            return None
        
        oldest_key = min(entries.keys(), key=lambda k: entries[k].created_at)
        return oldest_key

class RandomEvictionStrategy(EvictionStrategy):
    """Random eviction strategy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select random entry."""
        if not entries:
            return None
        
        import random
        return random.choice(list(entries.keys()))

class HybridEvictionStrategy(EvictionStrategy):
    """Hybrid eviction strategy combining multiple approaches."""
    
    def __init__(self, lru_weight: float = 0.4, lfu_weight: float = 0.4, ttl_weight: float = 0.2):
        self.lru_weight = lru_weight
        self.lfu_weight = lfu_weight
        self.ttl_weight = ttl_weight
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select entry using weighted combination of strategies."""
        if not entries:
            return None
        
        current_time = time.time()
        scores = {}
        
        for key, entry in entries.items():
            # LRU score (lower is worse)
            lru_score = (current_time - entry.accessed_at) / 3600  # Normalize to hours
            
            # LFU score (lower is worse)
            lfu_score = 1.0 / max(entry.access_count, 1)
            
            # TTL score (closer to expiration is worse)
            if entry.ttl > 0:
                age = current_time - entry.created_at
                ttl_score = age / entry.ttl
            else:
                ttl_score = 0.5  # Neutral score for no TTL
            
            # Combined weighted score (higher is worse)
            combined_score = (
                lru_score * self.lru_weight +
                lfu_score * self.lfu_weight +
                ttl_score * self.ttl_weight
            )
            
            scores[key] = combined_score
        
        # Return key with highest (worst) score
        return max(scores.keys(), key=lambda k: scores[k])

# =============================================================================
# Cache Backends
# =============================================================================

class CacheBackendInterface(ABC):
    """Abstract interface for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get cache size."""
        pass

class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.entries: Dict[str, CacheEntry] = {}
        self.eviction_strategy = self._create_eviction_strategy()
        self._lock = asyncio.Lock()
    
    def _create_eviction_strategy(self) -> EvictionStrategy:
        """Create eviction strategy based on config."""
        if self.config.eviction_policy == EvictionPolicy.LRU:
            return LRUEvictionStrategy()
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            return LFUEvictionStrategy()
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            return FIFOEvictionStrategy()
        elif self.config.eviction_policy == EvictionPolicy.RANDOM:
            return RandomEvictionStrategy()
        else:
            return HybridEvictionStrategy()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                if entry.is_expired():
                    del self.entries[key]
                    return None
                
                entry.touch()
                return entry.value
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Check if we need to evict entries
            if len(self.entries) >= self.config.max_size:
                await self._evict_entries(1)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl or self.config.ttl,
                size=self._estimate_size(value)
            )
            
            self.entries[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self.entries:
                del self.entries[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                if entry.is_expired():
                    del self.entries[key]
                    return False
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        async with self._lock:
            self.entries.clear()
            return True
    
    async def size(self) -> int:
        """Get cache size."""
        async with self._lock:
            # Clean expired entries first
            expired_keys = [k for k, v in self.entries.items() if v.is_expired()]
            for key in expired_keys:
                del self.entries[key]
            
            return len(self.entries)
    
    async def _evict_entries(self, count: int):
        """Evict entries using the eviction strategy."""
        for _ in range(min(count, len(self.entries))):
            key_to_evict = self.eviction_strategy.select_for_eviction(self.entries)
            if key_to_evict:
                del self.entries[key_to_evict]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate

class RedisCacheBackend(CacheBackendInterface):
    """Redis cache backend."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.logger = get_logger("redis_cache")
        self._connection_lock = asyncio.Lock()
    
    async def _ensure_connection(self):
        """Ensure Redis connection is established."""
        if self.redis_client is None:
            async with self._connection_lock:
                if self.redis_client is None:
                    try:
                        self.redis_client = redis.from_url(
                            self.config.redis_url,
                            db=self.config.redis_db,
                            password=self.config.redis_password,
                            decode_responses=False
                        )
                        await self.redis_client.ping()
                        self.logger.info("Connected to Redis")
                    except Exception as e:
                        self.logger.error(f"Failed to connect to Redis: {e}")
                        raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            await self._ensure_connection()
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cache."""
        try:
            await self._ensure_connection()
            data = pickle.dumps(value)
            ttl_seconds = int(ttl or self.config.ttl)
            
            if ttl_seconds > 0:
                await self.redis_client.setex(key, ttl_seconds, data)
            else:
                await self.redis_client.set(key, data)
            
            return True
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            await self._ensure_connection()
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            await self._ensure_connection()
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear Redis cache."""
        try:
            await self._ensure_connection()
            await self.redis_client.flushdb()
            return True
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            return False
    
    async def size(self) -> int:
        """Get Redis cache size."""
        try:
            await self._ensure_connection()
            return await self.redis_client.dbsize()
        except Exception as e:
            self.logger.error(f"Redis size error: {e}")
            return 0

# =============================================================================
# Main Advanced Cache Class
# =============================================================================

class AdvancedCache:
    """Advanced caching system with multiple backends, compression, and encryption."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.logger = get_logger("advanced_cache")
        self.stats = CacheStats()
        
        # Initialize managers
        self.compression_manager = CompressionManager(self.config.compression)
        self.encryption_manager = EncryptionManager(self.config.encryption_key)
        
        # Initialize backends
        self.memory_backend = MemoryCacheBackend(self.config)
        self.redis_backend = RedisCacheBackend(self.config) if REDIS_AVAILABLE else None
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background cleanup and stats collection tasks."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.enable_stats and (self._stats_task is None or self._stats_task.done()):
            self._stats_task = asyncio.create_task(self._stats_collection_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Try memory cache first
        value = await self.memory_backend.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try Redis cache if available
        if self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            value = await self.redis_backend.get(key)
            if value is not None:
                # Store in memory for faster access
                await self.memory_backend.set(key, value)
                self.stats.hits += 1
                return value
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        try:
            # Serialize and compress value
            serialized = pickle.dumps(value)
            
            # Compress if above threshold
            compressed, was_compressed = self.compression_manager.compress(serialized)
            
            # Encrypt if enabled
            encrypted, was_encrypted = self.encryption_manager.encrypt(compressed)
            
            # Store in memory cache
            success = await self.memory_backend.set(key, value, ttl)
            
            # Store in Redis if available
            if success and self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
                await self.redis_backend.set(key, value, ttl)
            
            if success:
                self.stats.sets += 1
                if was_compressed:
                    self.stats.compression_ratio = len(compressed) / len(serialized)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        success = await self.memory_backend.delete(key)
        
        if self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            await self.redis_backend.delete(key)
        
        if success:
            self.stats.deletes += 1
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        exists = await self.memory_backend.exists(key)
        
        if not exists and self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            exists = await self.redis_backend.exists(key)
        
        return exists
    
    async def clear(self) -> bool:
        """Clear all caches."""
        success = await self.memory_backend.clear()
        
        if self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            await self.redis_backend.clear()
        
        return success
    
    async def size(self) -> int:
        """Get total cache size."""
        memory_size = await self.memory_backend.size()
        redis_size = 0
        
        if self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            redis_size = await self.redis_backend.size()
        
        return memory_size + redis_size
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Update memory usage
        self.stats.memory_usage = await self.memory_backend.size()
        
        if self.redis_backend and self.config.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            self.stats.redis_usage = await self.redis_backend.size()
        
        return self.stats
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._shutdown_event.is_set():
            try:
                # Clean expired entries
                await self.memory_backend.size()  # This triggers cleanup
                
                # Update stats
                await self.get_stats()
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(600)
    
    async def _stats_collection_loop(self):
        """Background stats collection loop."""
        while not self._shutdown_event.is_set():
            try:
                await self.get_stats()
                await asyncio.sleep(60)  # Update stats every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats collection error: {e}")
                await asyncio.sleep(120)
    
    async def shutdown(self):
        """Shutdown the cache system."""
        self.logger.info("Shutting down advanced cache...")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._stats_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Advanced cache shutdown complete")

# =============================================================================
# Global Cache Instance
# =============================================================================

_default_cache: Optional[AdvancedCache] = None

def get_advanced_cache(config: Optional[CacheConfig] = None) -> AdvancedCache:
    """Get the global advanced cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = AdvancedCache(config)
    return _default_cache

async def shutdown_advanced_cache():
    """Shutdown the global advanced cache."""
    global _default_cache
    if _default_cache:
        await _default_cache.shutdown()
        _default_cache = None

# Export main classes
__all__ = [
    "AdvancedCache",
    "CacheConfig",
    "CacheBackend",
    "CompressionType",
    "EvictionPolicy",
    "CacheStats",
    "get_advanced_cache",
    "shutdown_advanced_cache"
]


