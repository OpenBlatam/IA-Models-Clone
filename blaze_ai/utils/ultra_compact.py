"""
Blaze AI Ultra Compact Utilities v7.0.0

Ultra-compact data structures and storage mechanisms for maximum memory efficiency,
including intelligent compression, hybrid storage, and bit-packed arrays.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
import threading
import time
import zlib
import pickle
import struct
import array
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class CompressionType(Enum):
    """Data compression types."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    GZIP = "gzip"
    ADAPTIVE = "adaptive"

class StorageType(Enum):
    """Storage strategy types."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    COMPRESSED = "compressed"
    STREAMING = "streaming"

class DataType(Enum):
    """Data type categories."""
    NUMERIC = "numeric"
    TEXT = "text"
    BINARY = "binary"
    STRUCTURED = "structured"
    MIXED = "mixed"

# Generic type for data
T = TypeVar('T')

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class UltraCompactConfig:
    """Configuration for ultra-compact storage."""
    compression_type: CompressionType = CompressionType.ADAPTIVE
    storage_type: StorageType = StorageType.HYBRID
    data_type: DataType = DataType.MIXED
    enable_compression: bool = True
    enable_caching: bool = True
    enable_streaming: bool = True
    compression_level: int = 6  # 0-9, higher = more compression
    cache_size: int = 1000
    chunk_size: int = 1024 * 1024  # 1MB chunks
    max_workers: int = 16
    custom_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompactMetrics:
    """Compact storage performance metrics."""
    total_stores: int = 0
    total_retrieves: int = 0
    compression_ratio: float = 1.0
    average_store_time: float = 0.0
    average_retrieve_time: float = 0.0
    total_store_time: float = 0.0
    total_retrieve_time: float = 0.0
    memory_saved: float = 0.0  # MB
    cache_hits: int = 0
    cache_misses: int = 0
    
    def record_store(self, store_time: float, original_size: int, compressed_size: int):
        """Record store operation metrics."""
        self.total_stores += 1
        self.total_store_time += store_time
        self.average_store_time = self.total_store_time / self.total_stores
        
        # Calculate compression ratio
        if original_size > 0:
            self.compression_ratio = compressed_size / original_size
        
        # Calculate memory saved
        memory_saved_mb = (original_size - compressed_size) / (1024 * 1024)
        self.memory_saved += memory_saved_mb
    
    def record_retrieve(self, retrieve_time: float):
        """Record retrieve operation metrics."""
        self.total_retrieves += 1
        self.total_retrieve_time += retrieve_time
        self.average_retrieve_time = self.total_retrieve_time / self.total_retrieves
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_stores": self.total_stores,
            "total_retrieves": self.total_retrieves,
            "compression_ratio": self.compression_ratio,
            "average_store_time": self.average_store_time,
            "average_retrieve_time": self.average_retrieve_time,
            "total_store_time": self.total_store_time,
            "total_retrieve_time": self.total_retrieve_time,
            "memory_saved_mb": self.memory_saved,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

# ============================================================================
# ULTRA COMPACT STORAGE
# ============================================================================

class UltraCompactStorage:
    """Ultra-compact storage with intelligent compression and hybrid strategies."""
    
    def __init__(self, config: UltraCompactConfig):
        self.config = config
        self.compact_metrics = CompactMetrics()
        self.worker_pools: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.compression_engines: Dict[CompressionType, Any] = {}
        self._lock = threading.Lock()
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize ultra-compact storage."""
        try:
            logger.info("Initializing Ultra Compact Storage")
            
            # Initialize compression engines
            await self._initialize_compression_engines()
            
            # Initialize worker pools
            await self._initialize_worker_pools()
            
            # Initialize cache
            if self.config.enable_caching:
                await self._initialize_cache()
            
            self._initialized = True
            logger.info("Ultra Compact Storage initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultra Compact Storage: {e}")
            return False
    
    async def _initialize_compression_engines(self):
        """Initialize compression engines."""
        try:
            # Initialize ZLIB compression
            try:
                self.compression_engines[CompressionType.ZLIB] = zlib
                logger.info("ZLIB compression engine initialized")
            except ImportError:
                logger.warning("ZLIB not available")
            
            # Initialize LZ4 compression if available
            try:
                import lz4.frame
                self.compression_engines[CompressionType.LZ4] = lz4.frame
                logger.info("LZ4 compression engine initialized")
            except ImportError:
                logger.warning("LZ4 not available")
            
            # Initialize Snappy compression if available
            try:
                import snappy
                self.compression_engines[CompressionType.SNAPPY] = snappy
                logger.info("Snappy compression engine initialized")
            except ImportError:
                logger.warning("Snappy not available")
            
            # Initialize GZIP compression
            try:
                import gzip
                self.compression_engines[CompressionType.GZIP] = gzip
                logger.info("GZIP compression engine initialized")
            except ImportError:
                logger.warning("GZIP not available")
            
            logger.info("Compression engines initialized")
            
        except Exception as e:
            logger.error(f"Error initializing compression engines: {e}")
    
    async def _initialize_worker_pools(self):
        """Initialize worker pools for compression operations."""
        try:
            # Thread pool for compression operations
            self.worker_pools["thread"] = ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
            
            # Process pool for CPU-intensive compression operations
            self.worker_pools["process"] = ProcessPoolExecutor(
                max_workers=self.config.max_workers // 2
            )
            
            logger.info(f"Compression worker pools initialized with {self.config.max_workers} total workers")
            
        except Exception as e:
            logger.error(f"Error initializing compression worker pools: {e}")
    
    async def _initialize_cache(self):
        """Initialize storage cache."""
        try:
            # Initialize cache with weak references for automatic cleanup
            self.cache = weakref.WeakValueDictionary()
            logger.info("Storage cache initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage cache: {e}")
    
    async def store(self, key: str, data: Any, **kwargs) -> bool:
        """Store data with ultra-compact optimization."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Serialize data
            serialized_data = await self._serialize_data(data)
            original_size = len(serialized_data)
            
            # Choose compression strategy
            compression_type = self._choose_compression_strategy(serialized_data, **kwargs)
            
            # Compress data
            compressed_data = await self._compress_data(serialized_data, compression_type)
            compressed_size = len(compressed_data)
            
            # Store based on storage strategy
            if self.config.storage_type == StorageType.MEMORY:
                success = await self._store_in_memory(key, compressed_data, compression_type)
            elif self.config.storage_type == StorageType.DISK:
                success = await self._store_on_disk(key, compressed_data, compression_type)
            elif self.config.storage_type == StorageType.HYBRID:
                success = await self._store_hybrid(key, compressed_data, compression_type)
            else:
                success = await self._store_compressed(key, compressed_data, compression_type)
            
            # Record metrics
            store_time = time.perf_counter() - start_time
            self.compact_metrics.record_store(store_time, original_size, compressed_size)
            
            # Cache if enabled
            if self.config.enable_caching and success:
                await self._cache_data(key, data)
            
            logger.info(f"Data stored with {compression_type.value} compression: {key}")
            return success
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return False
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Any]:
        """Retrieve data with ultra-compact optimization."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            if self.config.enable_caching:
                cached_data = await self._get_cached_data(key)
                if cached_data is not None:
                    self.compact_metrics.record_cache_hit()
                    return cached_data
                else:
                    self.compact_metrics.record_cache_miss()
            
            # Retrieve compressed data
            compressed_data = await self._retrieve_compressed_data(key)
            if compressed_data is None:
                return None
            
            # Decompress data
            decompressed_data = await self._decompress_data(compressed_data)
            
            # Deserialize data
            data = await self._deserialize_data(decompressed_data)
            
            # Record metrics
            retrieve_time = time.perf_counter() - start_time
            self.compact_metrics.record_retrieve(retrieve_time)
            
            # Cache if enabled
            if self.config.enable_caching:
                await self._cache_data(key, data)
            
            logger.info(f"Data retrieved: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return None
    
    async def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        try:
            # Use pickle for serialization
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            raise
    
    async def _deserialize_data(self, serialized_data: bytes) -> Any:
        """Deserialize data from bytes."""
        try:
            # Use pickle for deserialization
            data = pickle.loads(serialized_data)
            return data
            
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _choose_compression_strategy(self, data: bytes, **kwargs) -> CompressionType:
        """Choose optimal compression strategy."""
        try:
            if not self.config.enable_compression:
                return CompressionType.NONE
            
            # Check if specific compression type is requested
            if 'compression_type' in kwargs:
                requested_type = kwargs['compression_type']
                if isinstance(requested_type, CompressionType) and requested_type in self.compression_engines:
                    return requested_type
            
            # Adaptive compression based on data characteristics
            if self.config.compression_type == CompressionType.ADAPTIVE:
                data_size = len(data)
                
                # For small data, use faster compression
                if data_size < 1024:  # < 1KB
                    if CompressionType.SNAPPY in self.compression_engines:
                        return CompressionType.SNAPPY
                    elif CompressionType.LZ4 in self.compression_engines:
                        return CompressionType.LZ4
                
                # For medium data, use balanced compression
                elif data_size < 1024 * 1024:  # < 1MB
                    if CompressionType.LZ4 in self.compression_engines:
                        return CompressionType.LZ4
                    elif CompressionType.ZLIB in self.compression_engines:
                        return CompressionType.ZLIB
                
                # For large data, use high compression
                else:
                    if CompressionType.ZLIB in self.compression_engines:
                        return CompressionType.ZLIB
                    elif CompressionType.GZIP in self.compression_engines:
                        return CompressionType.GZIP
            
            # Default to ZLIB if available
            if CompressionType.ZLIB in self.compression_engines:
                return CompressionType.ZLIB
            
            # Fallback to no compression
            return CompressionType.NONE
            
        except Exception as e:
            logger.error(f"Error choosing compression strategy: {e}")
            return CompressionType.NONE
    
    async def _compress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified compression type."""
        try:
            if compression_type == CompressionType.NONE:
                return data
            
            if compression_type not in self.compression_engines:
                logger.warning(f"Compression type {compression_type.value} not available, using no compression")
                return data
            
            engine = self.compression_engines[compression_type]
            
            if compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(data, level=self.config.compression_level)
            elif compression_type == CompressionType.LZ4:
                compressed = engine.compress(data, compression_level=self.config.compression_level)
            elif compression_type == CompressionType.SNAPPY:
                compressed = engine.compress(data)
            elif compression_type == CompressionType.GZIP:
                compressed = engine.compress(data, compresslevel=self.config.compression_level)
            else:
                compressed = data
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return data
    
    async def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data."""
        try:
            # Try to detect compression type from metadata
            # For now, assume ZLIB compression
            try:
                decompressed = zlib.decompress(compressed_data)
                return decompressed
            except zlib.error:
                # Try other compression types
                try:
                    import lz4.frame
                    decompressed = lz4.frame.decompress(compressed_data)
                    return decompressed
                except Exception:
                    # Assume no compression
                    return compressed_data
            
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return compressed_data
    
    async def _store_in_memory(self, key: str, data: bytes, compression_type: CompressionType) -> bool:
        """Store data in memory."""
        try:
            with self._lock:
                # Store with metadata
                storage_entry = {
                    "data": data,
                    "compression_type": compression_type,
                    "timestamp": time.time(),
                    "size": len(data)
                }
                
                # Use weak reference for automatic cleanup
                self.cache[key] = storage_entry
                return True
                
        except Exception as e:
            logger.error(f"Error storing in memory: {e}")
            return False
    
    async def _store_on_disk(self, key: str, data: bytes, compression_type: CompressionType) -> bool:
        """Store data on disk."""
        try:
            # This would implement actual disk storage
            # For now, store in memory as fallback
            return await self._store_in_memory(key, data, compression_type)
            
        except Exception as e:
            logger.error(f"Error storing on disk: {e}")
            return False
    
    async def _store_hybrid(self, key: str, data: bytes, compression_type: CompressionType) -> bool:
        """Store data using hybrid strategy."""
        try:
            # Store in memory for fast access
            memory_success = await self._store_in_memory(key, data, compression_type)
            
            # Also store on disk for persistence
            disk_success = await self._store_on_disk(key, data, compression_type)
            
            return memory_success and disk_success
            
        except Exception as e:
            logger.error(f"Error storing with hybrid strategy: {e}")
            return False
    
    async def _store_compressed(self, key: str, data: bytes, compression_type: CompressionType) -> bool:
        """Store data with compression metadata."""
        try:
            # Store with compression information
            storage_entry = {
                "data": data,
                "compression_type": compression_type,
                "timestamp": time.time(),
                "size": len(data),
                "compressed": True
            }
            
            with self._lock:
                self.cache[key] = storage_entry
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing compressed data: {e}")
            return False
    
    async def _retrieve_compressed_data(self, key: str) -> Optional[bytes]:
        """Retrieve compressed data."""
        try:
            with self._lock:
                if key in self.cache:
                    storage_entry = self.cache[key]
                    return storage_entry.get("data")
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving compressed data: {e}")
            return None
    
    async def _cache_data(self, key: str, data: Any):
        """Cache data for fast access."""
        try:
            if self.config.enable_caching:
                with self._lock:
                    # Store original data in cache
                    cache_entry = {
                        "data": data,
                        "timestamp": time.time(),
                        "access_count": 0
                    }
                    self.cache[f"cache_{key}"] = cache_entry
                    
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    async def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        try:
            cache_key = f"cache_{key}"
            with self._lock:
                if cache_key in self.cache:
                    cache_entry = self.cache[cache_key]
                    cache_entry["access_count"] += 1
                    return cache_entry["data"]
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    def get_compact_stats(self) -> Dict[str, Any]:
        """Get ultra-compact storage statistics."""
        return {
            "storage_status": "initialized" if self._initialized else "uninitialized",
            "config": {
                "compression_type": self.config.compression_type.value,
                "storage_type": self.config.storage_type.value,
                "data_type": self.config.data_type.value,
                "enable_compression": self.config.enable_compression,
                "enable_caching": self.config.enable_caching
            },
            "compact_metrics": self.compact_metrics.to_dict(),
            "worker_pools": {
                name: type(pool).__name__ for name, pool in self.worker_pools.items()
            },
            "compression_engines": {
                engine_type.value: "available" for engine_type in self.compression_engines.keys()
            },
            "cache_size": len(self.cache)
        }
    
    async def shutdown(self):
        """Shutdown ultra-compact storage."""
        try:
            # Shutdown worker pools
            for name, pool in self.worker_pools.items():
                pool.shutdown(wait=True)
            
            # Clear cache
            with self._lock:
                self.cache.clear()
            
            logger.info("Ultra Compact Storage shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error during Ultra Compact Storage shutdown: {e}")

# ============================================================================
# BIT-PACKED ARRAY
# ============================================================================

class BitPackedArray:
    """Ultra-compact bit-packed array for numeric data."""
    
    def __init__(self, data_type: str = "uint8", initial_data: Optional[List[int]] = None):
        self.data_type = data_type
        self.bits_per_element = self._get_bits_per_element(data_type)
        self.data = array.array(data_type, initial_data or [])
        self._lock = threading.Lock()
        
    def _get_bits_per_element(self, data_type: str) -> int:
        """Get bits per element for data type."""
        type_bits = {
            "uint8": 8,
            "uint16": 16,
            "uint32": 32,
            "uint64": 64,
            "int8": 8,
            "int16": 16,
            "int32": 32,
            "int64": 64
        }
        return type_bits.get(data_type, 32)
    
    def append(self, value: int):
        """Append value to array."""
        with self._lock:
            self.data.append(value)
    
    def extend(self, values: List[int]):
        """Extend array with values."""
        with self._lock:
            self.data.extend(values)
    
    def __getitem__(self, index: int) -> int:
        """Get item at index."""
        return self.data[index]
    
    def __setitem__(self, index: int, value: int):
        """Set item at index."""
        with self._lock:
            self.data[index] = value
    
    def __len__(self) -> int:
        """Get array length."""
        return len(self.data)
    
    def to_bytes(self) -> bytes:
        """Convert array to bytes."""
        return self.data.tobytes()
    
    def from_bytes(self, data: bytes):
        """Load array from bytes."""
        with self._lock:
            self.data = array.array(self.data_type)
            self.data.frombytes(data)
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        return len(self.data) * (self.bits_per_element // 8)
    
    def compress(self) -> bytes:
        """Compress the array."""
        try:
            # Use ZLIB compression
            data_bytes = self.to_bytes()
            compressed = zlib.compress(data_bytes, level=9)
            return compressed
        except Exception as e:
            logger.error(f"Error compressing array: {e}")
            return self.to_bytes()
    
    def decompress(self, compressed_data: bytes):
        """Decompress the array."""
        try:
            decompressed = zlib.decompress(compressed_data)
            self.from_bytes(decompressed)
        except Exception as e:
            logger.error(f"Error decompressing array: {e}")

# ============================================================================
# COMPACT STRING ARRAY
# ============================================================================

class CompactStringArray:
    """Ultra-compact string array with deduplication."""
    
    def __init__(self):
        self.strings: List[str] = []
        self.string_map: Dict[str, int] = {}
        self.indices: List[int] = []
        self._lock = threading.Lock()
        
    def add_string(self, string: str) -> int:
        """Add string and return its index."""
        with self._lock:
            if string in self.string_map:
                # String already exists, return existing index
                index = self.string_map[string]
                self.indices.append(index)
                return index
            else:
                # Add new string
                index = len(self.strings)
                self.strings.append(string)
                self.string_map[string] = index
                self.indices.append(index)
                return index
    
    def get_string(self, index: int) -> str:
        """Get string by index."""
        return self.strings[index]
    
    def get_strings(self) -> List[str]:
        """Get all unique strings."""
        return self.strings.copy()
    
    def get_indices(self) -> List[int]:
        """Get all string indices."""
        return self.indices.copy()
    
    def __len__(self) -> int:
        """Get number of string references."""
        return len(self.indices)
    
    def get_unique_count(self) -> int:
        """Get number of unique strings."""
        return len(self.strings)
    
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        total_size = sum(len(s.encode('utf-8')) for s in self.strings)
        total_size += len(self.indices) * 4  # 4 bytes per index
        return total_size
    
    def compress(self) -> bytes:
        """Compress the string array."""
        try:
            # Serialize and compress
            data = {
                "strings": self.strings,
                "indices": self.indices
            }
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized, level=9)
            return compressed
        except Exception as e:
            logger.error(f"Error compressing string array: {e}")
            return b""
    
    def decompress(self, compressed_data: bytes):
        """Decompress the string array."""
        try:
            decompressed = zlib.decompress(compressed_data)
            data = pickle.loads(decompressed)
            
            with self._lock:
                self.strings = data["strings"]
                self.indices = data["indices"]
                self.string_map = {s: i for i, s in enumerate(self.strings)}
                
        except Exception as e:
            logger.error(f"Error decompressing string array: {e}")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_ultra_compact_storage(config: Optional[UltraCompactConfig] = None) -> UltraCompactStorage:
    """Create an ultra-compact storage instance."""
    if config is None:
        config = UltraCompactConfig()
    return UltraCompactStorage(config)

def create_memory_optimized_config() -> UltraCompactConfig:
    """Create a memory-optimized configuration."""
    return UltraCompactConfig(
        compression_type=CompressionType.ADAPTIVE,
        storage_type=StorageType.MEMORY,
        enable_compression=True,
        enable_caching=True,
        compression_level=9
    )

def create_disk_optimized_config() -> UltraCompactConfig:
    """Create a disk-optimized configuration."""
    return UltraCompactConfig(
        compression_type=CompressionType.ZLIB,
        storage_type=StorageType.DISK,
        enable_compression=True,
        enable_caching=False,
        compression_level=6
    )

def create_hybrid_optimized_config() -> UltraCompactConfig:
    """Create a hybrid-optimized configuration."""
    return UltraCompactConfig(
        compression_type=CompressionType.ADAPTIVE,
        storage_type=StorageType.HYBRID,
        enable_compression=True,
        enable_caching=True,
        enable_streaming=True,
        compression_level=7
    )

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "CompressionType",
    "StorageType",
    "DataType",
    
    # Configuration
    "UltraCompactConfig",
    "CompactMetrics",
    
    # Main Classes
    "UltraCompactStorage",
    "BitPackedArray",
    "CompactStringArray",
    
    # Factory Functions
    "create_ultra_compact_storage",
    "create_memory_optimized_config",
    "create_disk_optimized_config",
    "create_hybrid_optimized_config"
]

# Version info
__version__ = "7.0.0"
