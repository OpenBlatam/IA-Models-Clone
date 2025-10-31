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
import json
import time
import hashlib
import pickle
import gzip
from typing import (
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from enum import Enum
import logging
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from pydantic.json import pydantic_encoder
import structlog
    from caching_system import CacheManager, CacheConfig, create_cache_manager
    import msgpack
    import orjson
from typing import Any, List, Dict, Optional
"""
🚀 OPTIMIZED PYDANTIC SERIALIZATION SYSTEM
==========================================

Production-ready Pydantic serialization system with:
- High-performance serialization/deserialization
- Caching for serialized data
- Compression for large objects
- Validation and error handling
- Performance monitoring
- Integration with existing caching system

Features:
- Optimized Pydantic model serialization
- Cached serialization results
- Automatic compression for large objects
- Validation caching
- Performance metrics
- Error recovery
- Type safety
"""

    Any, Optional, Dict, List, Union, Callable, Awaitable,
    Tuple, Set, TypeVar, Generic, Type, get_type_hints
)


# Import caching system
try:
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class SerializationFormat(str, Enum):
    """Supported serialization formats."""
    JSON = "json"           # Standard JSON
    PICKLE = "pickle"       # Python pickle
    MSGPACK = "msgpack"     # MessagePack (if available)
    ORJSON = "orjson"       # Fast JSON (if available)

class CompressionLevel(str, Enum):
    """Compression levels."""
    NONE = "none"           # No compression
    FAST = "fast"           # Fast compression
    BALANCED = "balanced"   # Balanced compression
    MAX = "max"             # Maximum compression

class SerializationConfig(BaseModel):
    """Configuration for serialization system."""
    
    # Serialization format
    default_format: SerializationFormat = Field(default=SerializationFormat.JSON, description="Default serialization format")
    fallback_format: SerializationFormat = Field(default=SerializationFormat.PICKLE, description="Fallback format")
    
    # Compression settings
    enable_compression: bool = Field(default=True, description="Enable compression")
    compression_level: CompressionLevel = Field(default=CompressionLevel.BALANCED, description="Compression level")
    compression_threshold: int = Field(default=1024, description="Minimum size for compression")
    
    # Caching settings
    enable_caching: bool = Field(default=True, description="Enable serialization caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Maximum cache size")
    
    # Validation settings
    enable_validation: bool = Field(default=True, description="Enable validation")
    cache_validation: bool = Field(default=True, description="Cache validation results")
    strict_validation: bool = Field(default=False, description="Strict validation mode")
    
    # Performance settings
    enable_profiling: bool = Field(default=True, description="Enable performance profiling")
    profile_threshold: float = Field(default=0.1, description="Profiling threshold in seconds")
    
    # Error handling
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=0.1, description="Retry delay in seconds")
    
    # Pydantic settings
    pydantic_config: ConfigDict = Field(default_factory=lambda: ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds()
        }
    ), description="Pydantic configuration")
    
    class Config:
        validate_assignment = True

# ============================================================================
# SERIALIZATION UTILITIES
# ============================================================================

class SerializationUtils:
    """Utility functions for serialization."""
    
    @staticmethod
    def get_compression_level(level: CompressionLevel) -> int:
        """Get compression level for gzip."""
        levels = {
            CompressionLevel.NONE: 0,
            CompressionLevel.FAST: 1,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.MAX: 9
        }
        return levels.get(level, 6)
    
    @staticmethod
    def should_compress(data_size: int, config: SerializationConfig) -> bool:
        """Determine if data should be compressed."""
        return (
            config.enable_compression and 
            data_size > config.compression_threshold and
            config.compression_level != CompressionLevel.NONE
        )
    
    @staticmethod
    def compress_data(data: bytes, config: SerializationConfig) -> bytes:
        """Compress data using gzip."""
        if not SerializationUtils.should_compress(len(data), config):
            return data
        
        level = SerializationUtils.get_compression_level(config.compression_level)
        compressed = gzip.compress(data, level)
        
        # Add compression marker
        return b"gzip:" + compressed
    
    @staticmethod
    def decompress_data(data: bytes) -> bytes:
        """Decompress data."""
        if data.startswith(b"gzip:"):
            compressed_data = data[5:]  # Remove "gzip:" prefix
            return gzip.decompress(compressed_data)
        return data
    
    @staticmethod
    def generate_hash(data: Any) -> str:
        """Generate hash for data."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, bytes):
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()

# ============================================================================
# SERIALIZERS
# ============================================================================

class BaseSerializer:
    """Base serializer class."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
        self.stats = {
            "serializations": 0,
            "deserializations": 0,
            "errors": 0,
            "total_time": 0.0
        }
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serializer statistics."""
        total_operations = self.stats["serializations"] + self.stats["deserializations"]
        avg_time = self.stats["total_time"] / total_operations if total_operations > 0 else 0
        
        return {
            "serializations": self.stats["serializations"],
            "deserializations": self.stats["deserializations"],
            "errors": self.stats["errors"],
            "total_time": self.stats["total_time"],
            "avg_time": avg_time,
            "error_rate": self.stats["errors"] / total_operations if total_operations > 0 else 0
        }

class JSONSerializer(BaseSerializer):
    """JSON-based serializer with Pydantic support."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data to JSON."""
        start_time = time.time()
        
        try:
            # Use Pydantic encoder for better compatibility
            json_str = json.dumps(data, default=pydantic_encoder, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')
            
            # Compress if needed
            compressed_bytes = SerializationUtils.compress_data(json_bytes, self.config)
            
            self.stats["serializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return compressed_bytes
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("JSON serialization error", error=str(e))
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data from JSON."""
        start_time = time.time()
        
        try:
            # Decompress if needed
            decompressed_bytes = SerializationUtils.decompress_data(data)
            
            # Parse JSON
            json_str = decompressed_bytes.decode('utf-8')
            result = json.loads(json_str)
            
            self.stats["deserializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("JSON deserialization error", error=str(e))
            raise

class PickleSerializer(BaseSerializer):
    """Pickle-based serializer."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize data using pickle."""
        start_time = time.time()
        
        try:
            pickle_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if needed
            compressed_bytes = SerializationUtils.compress_data(pickle_bytes, self.config)
            
            self.stats["serializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return compressed_bytes
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Pickle serialization error", error=str(e))
            raise
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data using pickle."""
        start_time = time.time()
        
        try:
            # Decompress if needed
            decompressed_bytes = SerializationUtils.decompress_data(data)
            
            # Unpickle
            result = pickle.loads(decompressed_bytes)
            
            self.stats["deserializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Pickle deserialization error", error=str(e))
            raise

# Try to import optional serializers
try:
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

if MSGPACK_AVAILABLE:
    class MessagePackSerializer(BaseSerializer):
        """MessagePack-based serializer."""
        
        def serialize(self, data: Any) -> bytes:
            """Serialize data using MessagePack."""
            start_time = time.time()
            
            try:
                msgpack_bytes = msgpack.packb(data, use_bin_type=True)
                
                # Compress if needed
                compressed_bytes = SerializationUtils.compress_data(msgpack_bytes, self.config)
                
                self.stats["serializations"] += 1
                self.stats["total_time"] += time.time() - start_time
                
                return compressed_bytes
                
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("MessagePack serialization error", error=str(e))
                raise
        
        def deserialize(self, data: bytes) -> Any:
            """Deserialize data using MessagePack."""
            start_time = time.time()
            
            try:
                # Decompress if needed
                decompressed_bytes = SerializationUtils.decompress_data(data)
                
                # Unpack
                result = msgpack.unpackb(decompressed_bytes, raw=False)
                
                self.stats["deserializations"] += 1
                self.stats["total_time"] += time.time() - start_time
                
                return result
                
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("MessagePack deserialization error", error=str(e))
                raise

if ORJSON_AVAILABLE:
    class OrJSONSerializer(BaseSerializer):
        """Fast JSON serializer using orjson."""
        
        def serialize(self, data: Any) -> bytes:
            """Serialize data using orjson."""
            start_time = time.time()
            
            try:
                json_bytes = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
                
                # Compress if needed
                compressed_bytes = SerializationUtils.compress_data(json_bytes, self.config)
                
                self.stats["serializations"] += 1
                self.stats["total_time"] += time.time() - start_time
                
                return compressed_bytes
                
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("orjson serialization error", error=str(e))
                raise
        
        def deserialize(self, data: bytes) -> Any:
            """Deserialize data using orjson."""
            start_time = time.time()
            
            try:
                # Decompress if needed
                decompressed_bytes = SerializationUtils.decompress_data(data)
                
                # Parse
                result = orjson.loads(decompressed_bytes)
                
                self.stats["deserializations"] += 1
                self.stats["total_time"] += time.time() - start_time
                
                return result
                
            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error("orjson deserialization error", error=str(e))
                raise

# ============================================================================
# PYDANTIC MODEL SERIALIZER
# ============================================================================

class PydanticModelSerializer:
    """Specialized serializer for Pydantic models."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger("PydanticModelSerializer")
        self.stats = {
            "model_serializations": 0,
            "model_deserializations": 0,
            "validations": 0,
            "validation_cache_hits": 0,
            "errors": 0,
            "total_time": 0.0
        }
        
        # Initialize serializers
        self.serializers = self._init_serializers()
        self.validation_cache = {} if config.cache_validation else None
    
    def _init_serializers(self) -> Dict[SerializationFormat, BaseSerializer]:
        """Initialize available serializers."""
        serializers = {
            SerializationFormat.JSON: JSONSerializer(self.config),
            SerializationFormat.PICKLE: PickleSerializer(self.config)
        }
        
        if MSGPACK_AVAILABLE:
            serializers[SerializationFormat.MSGPACK] = MessagePackSerializer(self.config)
        
        if ORJSON_AVAILABLE:
            serializers[SerializationFormat.ORJSON] = OrJSONSerializer(self.config)
        
        return serializers
    
    def serialize_model(self, model: BaseModel, format: Optional[SerializationFormat] = None) -> bytes:
        """Serialize Pydantic model."""
        start_time = time.time()
        format = format or self.config.default_format
        
        try:
            # Convert model to dict
            model_dict = model.model_dump()
            
            # Serialize using selected format
            serializer = self.serializers.get(format)
            if not serializer:
                # Fallback to default format
                serializer = self.serializers[self.config.fallback_format]
            
            serialized = serializer.serialize(model_dict)
            
            self.stats["model_serializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return serialized
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Model serialization error", error=str(e), model_type=type(model).__name__)
            raise
    
    def deserialize_model(self, data: bytes, model_class: Type[BaseModel], format: Optional[SerializationFormat] = None) -> BaseModel:
        """Deserialize data to Pydantic model."""
        start_time = time.time()
        format = format or self.config.default_format
        
        try:
            # Deserialize using selected format
            serializer = self.serializers.get(format)
            if not serializer:
                # Fallback to default format
                serializer = self.serializers[self.config.fallback_format]
            
            model_dict = serializer.deserialize(data)
            
            # Validate and create model
            model = self._validate_and_create_model(model_class, model_dict)
            
            self.stats["model_deserializations"] += 1
            self.stats["total_time"] += time.time() - start_time
            
            return model
            
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Model deserialization error", error=str(e), model_class=model_class.__name__)
            raise
    
    def _validate_and_create_model(self, model_class: Type[BaseModel], data: Dict[str, Any]) -> BaseModel:
        """Validate data and create model instance."""
        self.stats["validations"] += 1
        
        # Check validation cache
        if self.validation_cache is not None:
            cache_key = f"{model_class.__name__}:{SerializationUtils.generate_hash(data)}"
            if cache_key in self.validation_cache:
                self.stats["validation_cache_hits"] += 1
                return self.validation_cache[cache_key]
        
        try:
            # Create model instance
            model = model_class(**data)
            
            # Cache validation result
            if self.validation_cache is not None:
                cache_key = f"{model_class.__name__}:{SerializationUtils.generate_hash(data)}"
                self.validation_cache[cache_key] = model
            
            return model
            
        except ValidationError as e:
            self.logger.error("Validation error", errors=e.errors(), model_class=model_class.__name__)
            raise
        except Exception as e:
            self.logger.error("Model creation error", error=str(e), model_class=model_class.__name__)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serializer statistics."""
        total_operations = self.stats["model_serializations"] + self.stats["model_deserializations"]
        avg_time = self.stats["total_time"] / total_operations if total_operations > 0 else 0
        
        return {
            "model_serializations": self.stats["model_serializations"],
            "model_deserializations": self.stats["model_deserializations"],
            "validations": self.stats["validations"],
            "validation_cache_hits": self.stats["validation_cache_hits"],
            "errors": self.stats["errors"],
            "total_time": self.stats["total_time"],
            "avg_time": avg_time,
            "validation_cache_hit_rate": (
                self.stats["validation_cache_hits"] / self.stats["validations"] 
                if self.stats["validations"] > 0 else 0
            )
        }

# ============================================================================
# CACHED SERIALIZATION
# ============================================================================

class CachedSerializationManager:
    """Manager for cached serialization operations."""
    
    def __init__(self, config: SerializationConfig, cache_manager: Optional[CacheManager] = None):
        
    """__init__ function."""
self.config = config
        self.cache_manager = cache_manager
        self.model_serializer = PydanticModelSerializer(config)
        self.logger = structlog.get_logger("CachedSerializationManager")
        
        # Initialize cache if not provided
        if self.cache_manager is None and CACHING_AVAILABLE:
            cache_config = CacheConfig(
                memory_cache_size=config.cache_max_size,
                memory_cache_ttl=config.cache_ttl,
                enable_multi_tier=False
            )
            self.cache_manager = create_cache_manager(cache_config)
    
    async def start(self) -> Any:
        """Start the serialization manager."""
        if self.cache_manager:
            await self.cache_manager.start()
        self.logger.info("Cached serialization manager started")
    
    async def stop(self) -> Any:
        """Stop the serialization manager."""
        if self.cache_manager:
            await self.cache_manager.stop()
        self.logger.info("Cached serialization manager stopped")
    
    async def serialize_model_cached(
        self, 
        model: BaseModel, 
        format: Optional[SerializationFormat] = None,
        cache_key: Optional[str] = None
    ) -> bytes:
        """Serialize model with caching."""
        if not self.config.enable_caching or not self.cache_manager:
            return self.model_serializer.serialize_model(model, format)
        
        # Generate cache key
        if cache_key is None:
            model_hash = SerializationUtils.generate_hash(model.model_dump())
            format_str = (format or self.config.default_format).value
            cache_key = f"serialize:{type(model).__name__}:{model_hash}:{format_str}"
        
        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Serialize and cache
        serialized = self.model_serializer.serialize_model(model, format)
        await self.cache_manager.set(cache_key, serialized, ttl=self.config.cache_ttl)
        
        return serialized
    
    async def deserialize_model_cached(
        self, 
        data: bytes, 
        model_class: Type[BaseModel], 
        format: Optional[SerializationFormat] = None,
        cache_key: Optional[str] = None
    ) -> BaseModel:
        """Deserialize model with caching."""
        if not self.config.enable_caching or not self.cache_manager:
            return self.model_serializer.deserialize_model(data, model_class, format)
        
        # Generate cache key
        if cache_key is None:
            data_hash = SerializationUtils.generate_hash(data)
            format_str = (format or self.config.default_format).value
            cache_key = f"deserialize:{model_class.__name__}:{data_hash}:{format_str}"
        
        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Deserialize and cache
        model = self.model_serializer.deserialize_model(data, model_class, format)
        await self.cache_manager.set(cache_key, model, ttl=self.config.cache_ttl)
        
        return model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        model_stats = self.model_serializer.get_stats()
        cache_stats = {}
        
        if self.cache_manager:
            cache_stats = asyncio.run(self.cache_manager.get_stats())
        
        return {
            "model_serializer": model_stats,
            "cache": cache_stats,
            "config": {
                "enable_caching": self.config.enable_caching,
                "enable_compression": self.config.enable_compression,
                "enable_validation": self.config.enable_validation,
                "cache_validation": self.config.cache_validation
            }
        }

# ============================================================================
# PERFORMANCE PROFILING
# ============================================================================

class SerializationProfiler:
    """Performance profiler for serialization operations."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger("SerializationProfiler")
        self.profiles = {}
    
    def profile_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Profile a serialization operation."""
        if not self.config.enable_profiling:
            return func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record profile
            if operation_name not in self.profiles:
                self.profiles[operation_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0,
                    "avg_time": 0.0
                }
            
            profile = self.profiles[operation_name]
            profile["count"] += 1
            profile["total_time"] += duration
            profile["min_time"] = min(profile["min_time"], duration)
            profile["max_time"] = max(profile["max_time"], duration)
            profile["avg_time"] = profile["total_time"] / profile["count"]
            
            # Log slow operations
            if duration > self.config.profile_threshold:
                self.logger.warning(
                    "Slow serialization operation",
                    operation=operation_name,
                    duration=duration,
                    threshold=self.config.profile_threshold
                )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Serialization operation failed",
                operation=operation_name,
                duration=duration,
                error=str(e)
            )
            raise
    
    async def profile_async_operation(self, operation_name: str, func: Callable, *args, **kwargs):
        """Profile an async serialization operation."""
        if not self.config.enable_profiling:
            return await func(*args, **kwargs)
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Record profile
            if operation_name not in self.profiles:
                self.profiles[operation_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0,
                    "avg_time": 0.0
                }
            
            profile = self.profiles[operation_name]
            profile["count"] += 1
            profile["total_time"] += duration
            profile["min_time"] = min(profile["min_time"], duration)
            profile["max_time"] = max(profile["max_time"], duration)
            profile["avg_time"] = profile["total_time"] / profile["count"]
            
            # Log slow operations
            if duration > self.config.profile_threshold:
                self.logger.warning(
                    "Slow async serialization operation",
                    operation=operation_name,
                    duration=duration,
                    threshold=self.config.profile_threshold
                )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Async serialization operation failed",
                operation=operation_name,
                duration=duration,
                error=str(e)
            )
            raise
    
    def get_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance profiles."""
        return self.profiles.copy()
    
    def get_profile(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get profile for specific operation."""
        return self.profiles.get(operation_name)
    
    def clear_profiles(self) -> Any:
        """Clear all profiles."""
        self.profiles.clear()

# ============================================================================
# DECORATORS
# ============================================================================

def serialized(
    format: Optional[SerializationFormat] = None,
    cache_key: Optional[str] = None,
    manager: Optional[CachedSerializationManager] = None
):
    """Decorator for serializing function results."""
    def decorator(func: Callable[..., Awaitable[BaseModel]]) -> Callable[..., Awaitable[bytes]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if manager is None:
                # Create temporary manager
                config = SerializationConfig()
                temp_manager = CachedSerializationManager(config)
                await temp_manager.start()
                
                try:
                    # Execute function
                    model = await func(*args, **kwargs)
                    
                    # Serialize result
                    result = await temp_manager.serialize_model_cached(model, format, cache_key)
                    return result
                    
                finally:
                    await temp_manager.stop()
            else:
                # Execute function
                model = await func(*args, **kwargs)
                
                # Serialize result
                result = await manager.serialize_model_cached(model, format, cache_key)
                return result
        
        return wrapper
    return decorator

def deserialized(
    model_class: Type[BaseModel],
    format: Optional[SerializationFormat] = None,
    cache_key: Optional[str] = None,
    manager: Optional[CachedSerializationManager] = None
):
    """Decorator for deserializing function inputs."""
    def decorator(func: Callable[[bytes], Awaitable[Any]]) -> Callable[[bytes], Awaitable[Any]]:
        @wraps(func)
        async def wrapper(data: bytes):
            
    """wrapper function."""
if manager is None:
                # Create temporary manager
                config = SerializationConfig()
                temp_manager = CachedSerializationManager(config)
                await temp_manager.start()
                
                try:
                    # Deserialize input
                    model = await temp_manager.deserialize_model_cached(data, model_class, format, cache_key)
                    
                    # Execute function
                    result = await func(model)
                    return result
                    
                finally:
                    await temp_manager.stop()
            else:
                # Deserialize input
                model = await manager.deserialize_model_cached(data, model_class, format, cache_key)
                
                # Execute function
                result = await func(model)
                return result
        
        return wrapper
    return decorator

# ============================================================================
# MAIN SERIALIZATION MANAGER
# ============================================================================

class OptimizedSerializationManager:
    """Main manager for optimized Pydantic serialization."""
    
    def __init__(self, config: Optional[SerializationConfig] = None, cache_manager: Optional[CacheManager] = None):
        
    """__init__ function."""
self.config = config or SerializationConfig()
        self.cache_manager = cache_manager
        self.cached_manager = CachedSerializationManager(self.config, cache_manager)
        self.profiler = SerializationProfiler(self.config)
        self.logger = structlog.get_logger("OptimizedSerializationManager")
    
    async def start(self) -> Any:
        """Start the serialization manager."""
        await self.cached_manager.start()
        self.logger.info("Optimized serialization manager started")
    
    async def stop(self) -> Any:
        """Stop the serialization manager."""
        await self.cached_manager.stop()
        self.logger.info("Optimized serialization manager stopped")
    
    async def serialize_model(
        self, 
        model: BaseModel, 
        format: Optional[SerializationFormat] = None,
        cache_key: Optional[str] = None
    ) -> bytes:
        """Serialize Pydantic model with profiling."""
        return await self.profiler.profile_async_operation(
            "serialize_model",
            self.cached_manager.serialize_model_cached,
            model, format, cache_key
        )
    
    async def deserialize_model(
        self, 
        data: bytes, 
        model_class: Type[BaseModel], 
        format: Optional[SerializationFormat] = None,
        cache_key: Optional[str] = None
    ) -> BaseModel:
        """Deserialize data to Pydantic model with profiling."""
        return await self.profiler.profile_async_operation(
            "deserialize_model",
            self.cached_manager.deserialize_model_cached,
            data, model_class, format, cache_key
        )
    
    def serialize_sync(self, model: BaseModel, format: Optional[SerializationFormat] = None) -> bytes:
        """Synchronous model serialization."""
        return self.profiler.profile_operation(
            "serialize_sync",
            self.cached_manager.model_serializer.serialize_model,
            model, format
        )
    
    def deserialize_sync(self, data: bytes, model_class: Type[BaseModel], format: Optional[SerializationFormat] = None) -> BaseModel:
        """Synchronous model deserialization."""
        return self.profiler.profile_operation(
            "deserialize_sync",
            self.cached_manager.model_serializer.deserialize_model,
            data, model_class, format
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "cached_manager": self.cached_manager.get_stats(),
            "profiler": self.profiler.get_profiles(),
            "config": self.config.dict()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        stats = self.get_stats()
        profiles = stats["profiler"]
        
        report = {
            "summary": {
                "total_operations": sum(p["count"] for p in profiles.values()),
                "total_time": sum(p["total_time"] for p in profiles.values()),
                "avg_time": sum(p["avg_time"] * p["count"] for p in profiles.values()) / sum(p["count"] for p in profiles.values()) if any(p["count"] > 0 for p in profiles.values()) else 0
            },
            "operations": profiles,
            "recommendations": self._generate_recommendations(stats)
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check cache hit rates
        cache_stats = stats["cached_manager"]["cache"]
        if "l1_cache" in cache_stats:
            l1_hit_rate = cache_stats["l1_cache"]["hit_rate"]
            if l1_hit_rate < 0.5:
                recommendations.append("Consider increasing cache size or TTL for better hit rates")
        
        # Check serialization performance
        profiles = stats["profiler"]
        for operation, profile in profiles.items():
            if profile["avg_time"] > 0.1:  # 100ms threshold
                recommendations.append(f"Consider optimizing {operation} - average time: {profile['avg_time']:.3f}s")
        
        # Check compression effectiveness
        if self.config.enable_compression:
            recommendations.append("Compression is enabled - monitor memory usage vs CPU overhead")
        
        return recommendations

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_serialization_config(**kwargs) -> SerializationConfig:
    """Create serialization configuration with defaults."""
    return SerializationConfig(**kwargs)

def create_serialization_manager(
    config: Optional[SerializationConfig] = None,
    cache_manager: Optional[CacheManager] = None
) -> OptimizedSerializationManager:
    """Create optimized serialization manager."""
    return OptimizedSerializationManager(config, cache_manager)

def benchmark_serialization(
    models: List[BaseModel],
    config: Optional[SerializationConfig] = None
) -> Dict[str, Any]:
    """Benchmark serialization performance."""
    config = config or SerializationConfig()
    manager = OptimizedSerializationManager(config)
    
    results = {
        "formats": {},
        "compression_levels": {},
        "model_sizes": {}
    }
    
    # Test different formats
    for format in SerializationFormat:
        if format == SerializationFormat.MSGPACK and not MSGPACK_AVAILABLE:
            continue
        if format == SerializationFormat.ORJSON and not ORJSON_AVAILABLE:
            continue
        
        start_time = time.time()
        total_size = 0
        
        for model in models:
            serialized = manager.serialize_sync(model, format)
            total_size += len(serialized)
        
        duration = time.time() - start_time
        
        results["formats"][format.value] = {
            "duration": duration,
            "total_size": total_size,
            "avg_size": total_size / len(models),
            "throughput": len(models) / duration
        }
    
    # Test compression levels
    for level in CompressionLevel:
        config.compression_level = level
        manager = OptimizedSerializationManager(config)
        
        start_time = time.time()
        total_size = 0
        
        for model in models:
            serialized = manager.serialize_sync(model)
            total_size += len(serialized)
        
        duration = time.time() - start_time
        
        results["compression_levels"][level.value] = {
            "duration": duration,
            "total_size": total_size,
            "avg_size": total_size / len(models),
            "throughput": len(models) / duration
        }
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of how to use the optimized serialization system."""
    
    # Create sample Pydantic models
    class User(BaseModel):
        id: int
        name: str
        email: str
        created_at: datetime = Field(default_factory=datetime.now)
    
    class Post(BaseModel):
        id: int
        title: str
        content: str
        author_id: int
        created_at: datetime = Field(default_factory=datetime.now)
    
    # Create serialization manager
    config = SerializationConfig(
        enable_caching=True,
        enable_compression=True,
        enable_profiling=True
    )
    
    manager = create_serialization_manager(config)
    await manager.start()
    
    # Create sample data
    user = User(id=1, name="John Doe", email="john@example.com")
    post = Post(id=1, title="Hello World", content="This is a test post", author_id=1)
    
    # Serialize models
    user_serialized = await manager.serialize_model(user, SerializationFormat.JSON)
    post_serialized = await manager.serialize_model(post, SerializationFormat.ORJSON)
    
    # Deserialize models
    user_deserialized = await manager.deserialize_model(user_serialized, User, SerializationFormat.JSON)
    post_deserialized = await manager.deserialize_model(post_serialized, Post, SerializationFormat.ORJSON)
    
    # Get statistics
    stats = manager.get_stats()
    report = manager.get_performance_report()
    
    print(f"User serialized size: {len(user_serialized)} bytes")
    print(f"Post serialized size: {len(post_serialized)} bytes")
    print(f"Performance report: {report}")
    
    await manager.stop()

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 