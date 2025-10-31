from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import pickle
import gzip
import base64
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Generic, Callable
from functools import wraps, lru_cache
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict
import statistics
from pydantic import BaseModel, ValidationError, ConfigDict
from pydantic.json import pydantic_encoder
import numpy as np
import torch
from PIL import Image
import io
            import lz4.frame
            import lz4.frame
            import zstandard as zstd
            import zstandard as zstd
            import psutil
    from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
"""
🚀 PYDANTIC SERIALIZATION OPTIMIZATION - AI VIDEO SYSTEM
=======================================================

Advanced data serialization and deserialization optimization using Pydantic
with features like:
- Optimized serialization/deserialization
- Caching of serialized data
- Compression for large objects
- Performance monitoring
- Custom serializers for complex types
- Batch processing optimization
- Memory-efficient serialization
"""



logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T', bound=BaseModel)
K = TypeVar('K')

# ============================================================================
# 1. SERIALIZATION CONFIGURATION
# ============================================================================

class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MESSAGE_PACK = "msgpack"
    PROTOBUF = "protobuf"
    CUSTOM = "custom"

class CompressionType(str, Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

@dataclass
class SerializationConfig:
    """Configuration for serialization optimization."""
    format: SerializationFormat = SerializationFormat.JSON
    compression: CompressionType = CompressionType.GZIP
    compression_threshold: int = 1024  # Compress if larger than this
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    max_cache_size: int = 1000
    enable_compression: bool = True
    enable_stats: bool = True
    pretty_print: bool = False
    exclude_none: bool = True
    exclude_defaults: bool = False
    use_enum_values: bool = True
    serialize_as_string: bool = False

# ============================================================================
# 2. SERIALIZATION STATISTICS
# ============================================================================

@dataclass
class SerializationStats:
    """Statistics for serialization performance."""
    total_serializations: int = 0
    total_deserializations: int = 0
    serialization_time: float = 0.0
    deserialization_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_ratio: float = 1.0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    @property
    def avg_serialization_time(self) -> float:
        """Average serialization time."""
        return self.serialization_time / self.total_serializations if self.total_serializations > 0 else 0.0
    
    @property
    def avg_deserialization_time(self) -> float:
        """Average deserialization time."""
        return self.deserialization_time / self.total_deserializations if self.total_deserializations > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    @property
    def compression_efficiency(self) -> float:
        """Compression efficiency (1.0 = no compression, 0.0 = maximum compression)."""
        if self.total_size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.total_size_bytes
    
    def reset(self) -> Any:
        """Reset statistics."""
        self.total_serializations = 0
        self.total_deserializations = 0
        self.serialization_time = 0.0
        self.deserialization_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_ratio = 1.0
        self.total_size_bytes = 0
        self.compressed_size_bytes = 0

# ============================================================================
# 3. CUSTOM SERIALIZERS
# ============================================================================

class CustomSerializers:
    """Custom serializers for complex data types."""
    
    @staticmethod
    def serialize_numpy_array(arr: np.ndarray) -> Dict[str, Any]:
        """Serialize numpy array."""
        return {
            "__type__": "numpy_array",
            "data": base64.b64encode(arr.tobytes()).decode('utf-8'),
            "dtype": str(arr.dtype),
            "shape": arr.shape
        }
    
    @staticmethod
    def deserialize_numpy_array(data: Dict[str, Any]) -> np.ndarray:
        """Deserialize numpy array."""
        if data.get("__type__") == "numpy_array":
            arr_bytes = base64.b64decode(data["data"])
            arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
            return arr.reshape(data["shape"])
        raise ValueError("Invalid numpy array data")
    
    @staticmethod
    def serialize_torch_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
        """Serialize PyTorch tensor."""
        return {
            "__type__": "torch_tensor",
            "data": base64.b64encode(tensor.numpy().tobytes()).decode('utf-8'),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(tensor.device)
        }
    
    @staticmethod
    def deserialize_torch_tensor(data: Dict[str, Any]) -> torch.Tensor:
        """Deserialize PyTorch tensor."""
        if data.get("__type__") == "torch_tensor":
            arr_bytes = base64.b64decode(data["data"])
            arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
            arr = arr.reshape(data["shape"])
            tensor = torch.from_numpy(arr)
            if data["device"] != "cpu":
                tensor = tensor.to(data["device"])
            return tensor
        raise ValueError("Invalid torch tensor data")
    
    @staticmethod
    def serialize_pil_image(image: Image.Image) -> Dict[str, Any]:
        """Serialize PIL Image."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return {
            "__type__": "pil_image",
            "data": base64.b64encode(buffer.getvalue()).decode('utf-8'),
            "mode": image.mode,
            "size": image.size
        }
    
    @staticmethod
    def deserialize_pil_image(data: Dict[str, Any]) -> Image.Image:
        """Deserialize PIL Image."""
        if data.get("__type__") == "pil_image":
            image_bytes = base64.b64decode(data["data"])
            buffer = io.BytesIO(image_bytes)
            return Image.open(buffer)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError("Invalid PIL image data")
    
    @staticmethod
    def serialize_video_frame(frame: np.ndarray) -> Dict[str, Any]:
        """Serialize video frame (numpy array with metadata)."""
        return {
            "__type__": "video_frame",
            "data": base64.b64encode(frame.tobytes()).decode('utf-8'),
            "dtype": str(frame.dtype),
            "shape": frame.shape,
            "timestamp": time.time()
        }
    
    @staticmethod
    def deserialize_video_frame(data: Dict[str, Any]) -> np.ndarray:
        """Deserialize video frame."""
        if data.get("__type__") == "video_frame":
            frame_bytes = base64.b64decode(data["data"])
            frame = np.frombuffer(frame_bytes, dtype=np.dtype(data["dtype"]))
            return frame.reshape(data["shape"])
        raise ValueError("Invalid video frame data")

# ============================================================================
# 4. SERIALIZATION CACHE
# ============================================================================

class SerializationCache:
    """Cache for serialized data to improve performance."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        
    """__init__ function."""
self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, data: Any, config: SerializationConfig) -> str:
        """Generate cache key for data."""
        # Create a hash of the data and config
        data_str = str(data)
        config_str = f"{config.format}_{config.compression}_{config.pretty_print}"
        content = f"{data_str}:{config_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, data: Any, config: SerializationConfig) -> Optional[bytes]:
        """Get serialized data from cache."""
        key = self._generate_key(data, config)
        
        async with self._lock:
            if key in self.cache:
                cache_entry = self.cache[key]
                
                # Check TTL
                if time.time() - cache_entry["timestamp"] > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                return cache_entry["data"]
        
        return None
    
    async def set(self, data: Any, config: SerializationConfig, serialized_data: bytes):
        """Store serialized data in cache."""
        key = self._generate_key(data, config)
        
        async with self._lock:
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            
            self.cache[key] = {
                "data": serialized_data,
                "timestamp": time.time(),
                "size": len(serialized_data)
            }
            self.access_times[key] = time.time()
    
    async def _evict_oldest(self) -> Any:
        """Evict oldest cache entries."""
        if not self.access_times:
            return
        
        # Find oldest entry
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    async def clear(self) -> Any:
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry["size"] for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "ttl": self.ttl
        }

# ============================================================================
# 5. COMPRESSION UTILITIES
# ============================================================================

class CompressionUtils:
    """Utilities for data compression."""
    
    @staticmethod
    def should_compress(data: bytes, threshold: int) -> bool:
        """Check if data should be compressed."""
        return len(data) > threshold
    
    @staticmethod
    def compress_gzip(data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    @staticmethod
    def decompress_gzip(data: bytes) -> bytes:
        """Decompress data using gzip."""
        return gzip.decompress(data)
    
    @staticmethod
    def compress_lz4(data: bytes) -> bytes:
        """Compress data using LZ4."""
        try:
            return lz4.frame.compress(data)
        except ImportError:
            logger.warning("LZ4 not available, falling back to gzip")
            return CompressionUtils.compress_gzip(data)
    
    @staticmethod
    def decompress_lz4(data: bytes) -> bytes:
        """Decompress data using LZ4."""
        try:
            return lz4.frame.decompress(data)
        except ImportError:
            logger.warning("LZ4 not available, falling back to gzip")
            return CompressionUtils.decompress_gzip(data)
    
    @staticmethod
    def compress_zstd(data: bytes) -> bytes:
        """Compress data using Zstandard."""
        try:
            compressor = zstd.ZstdCompressor(level=3)
            return compressor.compress(data)
        except ImportError:
            logger.warning("Zstandard not available, falling back to gzip")
            return CompressionUtils.compress_gzip(data)
    
    @staticmethod
    def decompress_zstd(data: bytes) -> bytes:
        """Decompress data using Zstandard."""
        try:
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        except ImportError:
            logger.warning("Zstandard not available, falling back to gzip")
            return CompressionUtils.decompress_gzip(data)
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified compression type."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return CompressionUtils.compress_gzip(data)
        elif compression_type == CompressionType.LZ4:
            return CompressionUtils.compress_lz4(data)
        elif compression_type == CompressionType.ZSTD:
            return CompressionUtils.compress_zstd(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified compression type."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return CompressionUtils.decompress_gzip(data)
        elif compression_type == CompressionType.LZ4:
            return CompressionUtils.decompress_lz4(data)
        elif compression_type == CompressionType.ZSTD:
            return CompressionUtils.decompress_zstd(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")

# ============================================================================
# 6. OPTIMIZED SERIALIZER
# ============================================================================

class OptimizedSerializer:
    """Optimized serializer with caching and compression."""
    
    def __init__(self, config: SerializationConfig = None):
        
    """__init__ function."""
self.config = config or SerializationConfig()
        self.cache = SerializationCache(
            max_size=self.config.max_cache_size,
            ttl=self.config.cache_ttl
        ) if self.config.enable_caching else None
        self.stats = SerializationStats() if self.config.enable_stats else None
        self.custom_serializers = CustomSerializers()
    
    async def serialize(self, data: Any, model_class: Optional[Type[T]] = None) -> bytes:
        """Serialize data with optimization."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache and self.config.enable_caching:
                cached_result = await self.cache.get(data, self.config)
                if cached_result is not None:
                    if self.stats:
                        self.stats.cache_hits += 1
                    return cached_result
                
                if self.stats:
                    self.stats.cache_misses += 1
            
            # Serialize data
            if model_class and isinstance(data, model_class):
                serialized = await self._serialize_pydantic_model(data)
            else:
                serialized = await self._serialize_generic_data(data)
            
            # Compress if needed
            if self.config.enable_compression and self.config.compression != CompressionType.NONE:
                if CompressionUtils.should_compress(serialized, self.config.compression_threshold):
                    compressed = CompressionUtils.compress(serialized, self.config.compression)
                    
                    # Add compression header
                    header = {
                        "compressed": True,
                        "compression_type": self.config.compression.value,
                        "original_size": len(serialized),
                        "compressed_size": len(compressed)
                    }
                    
                    final_data = json.dumps(header).encode() + b"\n" + compressed
                    
                    if self.stats:
                        self.stats.compressed_size_bytes += len(final_data)
                        self.stats.total_size_bytes += len(serialized)
                        self.stats.compression_ratio = self.stats.compressed_size_bytes / self.stats.total_size_bytes
                    
                    serialized = final_data
                else:
                    # Add uncompressed header
                    header = {
                        "compressed": False,
                        "compression_type": CompressionType.NONE.value,
                        "original_size": len(serialized),
                        "compressed_size": len(serialized)
                    }
                    
                    final_data = json.dumps(header).encode() + b"\n" + serialized
                    serialized = final_data
            
            # Cache result
            if self.cache and self.config.enable_caching:
                await self.cache.set(data, self.config, serialized)
            
            # Update statistics
            if self.stats:
                self.stats.total_serializations += 1
                self.stats.serialization_time += time.time() - start_time
                if not self.config.enable_compression:
                    self.stats.total_size_bytes += len(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise
    
    async def deserialize(self, data: bytes, model_class: Optional[Type[T]] = None) -> Any:
        """Deserialize data with optimization."""
        start_time = time.time()
        
        try:
            # Check for compression header
            if data.startswith(b"{"):
                header_end = data.find(b"\n")
                if header_end != -1:
                    header_data = data[:header_end]
                    compressed_data = data[header_end + 1:]
                    
                    header = json.loads(header_data.decode())
                    
                    if header.get("compressed", False):
                        # Decompress data
                        decompressed = CompressionUtils.decompress(
                            compressed_data,
                            CompressionType(header["compression_type"])
                        )
                        data = decompressed
            
            # Deserialize data
            if model_class:
                result = await self._deserialize_pydantic_model(data, model_class)
            else:
                result = await self._deserialize_generic_data(data)
            
            # Update statistics
            if self.stats:
                self.stats.total_deserializations += 1
                self.stats.deserialization_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    async def _serialize_pydantic_model(self, model: BaseModel) -> bytes:
        """Serialize Pydantic model."""
        if self.config.format == SerializationFormat.JSON:
            return model.model_dump_json(
                exclude_none=self.config.exclude_none,
                exclude_defaults=self.config.exclude_defaults,
                use_enum_values=self.config.use_enum_values,
                indent=2 if self.config.pretty_print else None
            ).encode('utf-8')
        
        elif self.config.format == SerializationFormat.PICKLE:
            return pickle.dumps(model)
        
        else:
            raise ValueError(f"Unsupported format for Pydantic models: {self.config.format}")
    
    async def _deserialize_pydantic_model(self, data: bytes, model_class: Type[T]) -> T:
        """Deserialize Pydantic model."""
        if self.config.format == SerializationFormat.JSON:
            json_data = data.decode('utf-8')
            return model_class.model_validate_json(json_data)
        
        elif self.config.format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        
        else:
            raise ValueError(f"Unsupported format for Pydantic models: {self.config.format}")
    
    async def _serialize_generic_data(self, data: Any) -> bytes:
        """Serialize generic data."""
        if self.config.format == SerializationFormat.JSON:
            # Handle custom types
            if isinstance(data, np.ndarray):
                data = self.custom_serializers.serialize_numpy_array(data)
            elif isinstance(data, torch.Tensor):
                data = self.custom_serializers.serialize_torch_tensor(data)
            elif isinstance(data, Image.Image):
                data = self.custom_serializers.serialize_pil_image(data)
            
            return json.dumps(
                data,
                default=pydantic_encoder,
                indent=2 if self.config.pretty_print else None,
                ensure_ascii=False
            ).encode('utf-8')
        
        elif self.config.format == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")
    
    async def _deserialize_generic_data(self, data: bytes) -> Any:
        """Deserialize generic data."""
        if self.config.format == SerializationFormat.JSON:
            json_data = data.decode('utf-8')
            result = json.loads(json_data)
            
            # Handle custom types
            if isinstance(result, dict) and "__type__" in result:
                if result["__type__"] == "numpy_array":
                    return self.custom_serializers.deserialize_numpy_array(result)
                elif result["__type__"] == "torch_tensor":
                    return self.custom_serializers.deserialize_torch_tensor(result)
                elif result["__type__"] == "pil_image":
                    return self.custom_serializers.deserialize_pil_image(result)
            
            return result
        
        elif self.config.format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        
        else:
            raise ValueError(f"Unsupported format: {self.config.format}")
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get serialization statistics."""
        if not self.stats:
            return None
        
        stats = {
            "total_serializations": self.stats.total_serializations,
            "total_deserializations": self.stats.total_deserializations,
            "avg_serialization_time": self.stats.avg_serialization_time,
            "avg_deserialization_time": self.stats.avg_deserialization_time,
            "total_size_bytes": self.stats.total_size_bytes,
            "compressed_size_bytes": self.stats.compressed_size_bytes,
            "compression_efficiency": self.stats.compression_efficiency
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
            stats["cache_hit_rate"] = self.stats.cache_hit_rate
        
        return stats
    
    async def clear_cache(self) -> Any:
        """Clear serialization cache."""
        if self.cache:
            await self.cache.clear()

# ============================================================================
# 7. BATCH SERIALIZATION OPTIMIZER
# ============================================================================

class BatchSerializationOptimizer:
    """Optimizer for batch serialization operations."""
    
    def __init__(self, serializer: OptimizedSerializer):
        
    """__init__ function."""
self.serializer = serializer
        self.batch_stats = defaultdict(int)
    
    async def serialize_batch(self, items: List[Any], model_class: Optional[Type[T]] = None) -> List[bytes]:
        """Serialize a batch of items efficiently."""
        start_time = time.time()
        
        # Use asyncio.gather for concurrent serialization
        tasks = [
            self.serializer.serialize(item, model_class) 
            for item in items
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to serialize item {i}: {result}")
            else:
                successful_results.append(result)
        
        # Update batch statistics
        self.batch_stats["total_batches"] += 1
        self.batch_stats["total_items"] += len(items)
        self.batch_stats["successful_items"] += len(successful_results)
        self.batch_stats["batch_time"] += time.time() - start_time
        
        return successful_results
    
    async def deserialize_batch(self, data_list: List[bytes], model_class: Optional[Type[T]] = None) -> List[Any]:
        """Deserialize a batch of data efficiently."""
        start_time = time.time()
        
        # Use asyncio.gather for concurrent deserialization
        tasks = [
            self.serializer.deserialize(data, model_class) 
            for data in data_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to deserialize item {i}: {result}")
            else:
                successful_results.append(result)
        
        # Update batch statistics
        self.batch_stats["total_deserialize_batches"] += 1
        self.batch_stats["total_deserialize_items"] += len(data_list)
        self.batch_stats["successful_deserialize_items"] += len(successful_results)
        self.batch_stats["batch_deserialize_time"] += time.time() - start_time
        
        return successful_results
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch serialization statistics."""
        return {
            "total_batches": self.batch_stats["total_batches"],
            "total_items": self.batch_stats["total_items"],
            "successful_items": self.batch_stats["successful_items"],
            "success_rate": (
                self.batch_stats["successful_items"] / self.batch_stats["total_items"]
                if self.batch_stats["total_items"] > 0 else 0.0
            ),
            "avg_batch_time": (
                self.batch_stats["batch_time"] / self.batch_stats["total_batches"]
                if self.batch_stats["total_batches"] > 0 else 0.0
            ),
            "total_deserialize_batches": self.batch_stats["total_deserialize_batches"],
            "total_deserialize_items": self.batch_stats["total_deserialize_items"],
            "successful_deserialize_items": self.batch_stats["successful_deserialize_items"],
            "deserialize_success_rate": (
                self.batch_stats["successful_deserialize_items"] / self.batch_stats["total_deserialize_items"]
                if self.batch_stats["total_deserialize_items"] > 0 else 0.0
            ),
            "avg_batch_deserialize_time": (
                self.batch_stats["batch_deserialize_time"] / self.batch_stats["total_deserialize_batches"]
                if self.batch_stats["total_deserialize_batches"] > 0 else 0.0
            )
        }

# ============================================================================
# 8. PERFORMANCE MONITORING
# ============================================================================

class SerializationPerformanceMonitor:
    """Monitor serialization performance."""
    
    def __init__(self) -> Any:
        self.performance_data = defaultdict(list)
        self.alerts = []
    
    @asynccontextmanager
    async def monitor_serialization(self, operation: str, data_size: int = 0):
        """Monitor serialization operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.performance_data[operation].append({
                "duration": duration,
                "memory_delta": memory_delta,
                "data_size": data_size,
                "timestamp": time.time()
            })
            
            # Keep only recent data
            if len(self.performance_data[operation]) > 1000:
                self.performance_data[operation] = self.performance_data[operation][-1000:]
            
            # Check for performance alerts
            if duration > 1.0:  # More than 1 second
                self.alerts.append({
                    "type": "slow_serialization",
                    "operation": operation,
                    "duration": duration,
                    "timestamp": time.time()
                })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "operations": {},
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": time.time()
        }
        
        for operation, data in self.performance_data.items():
            if data:
                durations = [item["duration"] for item in data]
                memory_deltas = [item["memory_delta"] for item in data]
                data_sizes = [item["data_size"] for item in data]
                
                report["operations"][operation] = {
                    "total_operations": len(data),
                    "avg_duration": statistics.mean(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations),
                    "avg_memory_delta": statistics.mean(memory_deltas),
                    "avg_data_size": statistics.mean(data_sizes) if data_sizes else 0,
                    "throughput": len(data) / sum(durations) if sum(durations) > 0 else 0
                }
        
        return report
    
    def clear_alerts(self) -> Any:
        """Clear performance alerts."""
        self.alerts.clear()

# ============================================================================
# 9. DECORATORS AND UTILITIES
# ============================================================================

def optimized_serialization(config: SerializationConfig = None):
    """Decorator for optimized serialization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            serializer = OptimizedSerializer(config)
            
            # Extract data to serialize from function result
            result = await func(*args, **kwargs)
            
            # Serialize result
            serialized = await serializer.serialize(result)
            
            return serialized
        
        return wrapper
    return decorator

def optimized_deserialization(model_class: Type[T], config: SerializationConfig = None):
    """Decorator for optimized deserialization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            serializer = OptimizedSerializer(config)
            
            # Extract serialized data from args/kwargs
            serialized_data = kwargs.get('data') or args[0]
            
            # Deserialize data
            deserialized = await serializer.deserialize(serialized_data, model_class)
            
            # Call original function with deserialized data
            kwargs['data'] = deserialized
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# ============================================================================
# 10. USAGE EXAMPLES
# ============================================================================

async def example_serialization_optimization():
    """Example of using the serialization optimization system."""
    
    # Create serializer with optimized configuration
    config = SerializationConfig(
        format=SerializationFormat.JSON,
        compression=CompressionType.GZIP,
        compression_threshold=1024,
        enable_caching=True,
        cache_ttl=3600,
        enable_stats=True
    )
    
    serializer = OptimizedSerializer(config)
    
    # Example Pydantic model
    
    class VideoData(BaseModel):
        video_id: str = Field(..., description="Video identifier")
        title: str = Field(..., description="Video title")
        duration: float = Field(..., description="Video duration")
        metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Create sample data
    video_data = VideoData(
        video_id="video_123",
        title="Sample Video",
        duration=120.5,
        metadata={"quality": "high", "format": "mp4"}
    )
    
    # Serialize data
    serialized = await serializer.serialize(video_data, VideoData)
    print(f"Serialized size: {len(serialized)} bytes")
    
    # Deserialize data
    deserialized = await serializer.deserialize(serialized, VideoData)
    print(f"Deserialized data: {deserialized}")
    
    # Get statistics
    stats = serializer.get_stats()
    print(f"Serialization stats: {stats}")
    
    # Batch serialization
    batch_optimizer = BatchSerializationOptimizer(serializer)
    
    video_list = [video_data] * 10  # 10 copies
    serialized_batch = await batch_optimizer.serialize_batch(video_list, VideoData)
    print(f"Batch serialized {len(serialized_batch)} items")
    
    # Performance monitoring
    monitor = SerializationPerformanceMonitor()
    
    async with monitor.monitor_serialization("video_serialization", len(serialized)):
        await serializer.serialize(video_data, VideoData)
    
    performance_report = monitor.get_performance_report()
    print(f"Performance report: {performance_report}")

match __name__:
    case "__main__":
    asyncio.run(example_serialization_optimization()) 