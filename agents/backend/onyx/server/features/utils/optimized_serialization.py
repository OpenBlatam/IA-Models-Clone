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
import logging
import hashlib
import pickle
import gzip
import json
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Tuple, Type, TypeVar
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import weakref
import functools
import orjson
from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator, model_validator
from pydantic.json import pydantic_encoder
import structlog
from typing import Any, List, Dict, Optional
"""
⚡ Optimized Serialization System
=================================

Advanced data serialization and deserialization optimization with Pydantic:
- Custom serializers and deserializers
- Compression and optimization
- Serialization caching
- Performance monitoring
- Schema validation
- Type conversion
- Memory optimization
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class SerializationFormat(Enum):
    """Serialization formats"""
    JSON = "json"
    ORJSON = "orjson"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_ORJSON = "compressed_orjson"

class CompressionLevel(Enum):
    """Compression levels"""
    NONE = 0
    FAST = 1
    BALANCED = 6
    BEST = 9

@dataclass
class SerializationConfig:
    """Serialization configuration"""
    # Format settings
    default_format: SerializationFormat = SerializationFormat.ORJSON
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600
    
    # Validation settings
    validate_on_serialize: bool = True
    validate_on_deserialize: bool = True
    strict_validation: bool = False
    
    # Memory settings
    enable_memory_optimization: bool = True
    max_memory_usage_mb: int = 512
    
    # Monitoring settings
    enable_metrics: bool = True
    log_slow_operations: bool = True
    slow_operation_threshold: float = 1.0

@dataclass
class SerializationMetrics:
    """Serialization performance metrics"""
    total_operations: int = 0
    serialize_operations: int = 0
    deserialize_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_operations: int = 0
    validation_operations: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    errors: int = 0

class OptimizedSerializer:
    """
    Advanced serializer with multiple formats, compression, and caching.
    """
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = SerializationMetrics()
        self.serialization_cache = {}
        self._lock = asyncio.Lock()
        
        # Initialize format handlers
        self._format_handlers = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.ORJSON: self._serialize_orjson,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.COMPRESSED_JSON: self._serialize_compressed_json,
            SerializationFormat.COMPRESSED_ORJSON: self._serialize_compressed_orjson
        }
        
        self._deserialization_handlers = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.ORJSON: self._deserialize_orjson,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.COMPRESSED_JSON: self._deserialize_compressed_json,
            SerializationFormat.COMPRESSED_ORJSON: self._deserialize_compressed_orjson
        }
    
    async def serialize(
        self, 
        data: Any, 
        format: SerializationFormat = None,
        model_class: Type[T] = None,
        validate: bool = None
    ) -> bytes:
        """Serialize data with optimization."""
        start_time = time.time()
        format = format or self.config.default_format
        validate = validate if validate is not None else self.config.validate_on_serialize
        
        try:
            # Generate cache key
            cache_key = None
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(data, format, model_class, validate)
                
                # Check cache
                async with self._lock:
                    if cache_key in self.serialization_cache:
                        cached_result = self.serialization_cache[cache_key]
                        if time.time() - cached_result['timestamp'] < self.config.cache_ttl:
                            self.metrics.cache_hits += 1
                            return cached_result['data']
                        else:
                            del self.serialization_cache[cache_key]
            
            # Validate if needed
            if validate and model_class:
                if isinstance(data, dict):
                    data = model_class(**data)
                elif not isinstance(data, model_class):
                    raise ValueError(f"Data must be instance of {model_class}")
            
            # Serialize
            handler = self._format_handlers.get(format)
            if not handler:
                raise ValueError(f"Unsupported format: {format}")
            
            result = await handler(data, model_class)
            
            # Cache result
            if cache_key and self.config.enable_caching:
                async with self._lock:
                    if len(self.serialization_cache) >= self.config.cache_size:
                        # Remove oldest entry
                        oldest_key = min(
                            self.serialization_cache.keys(),
                            key=lambda k: self.serialization_cache[k]['timestamp']
                        )
                        del self.serialization_cache[oldest_key]
                    
                    self.serialization_cache[cache_key] = {
                        'data': result,
                        'timestamp': time.time()
                    }
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics('serialize', execution_time)
            
            # Log slow operations
            if self.config.log_slow_operations and execution_time > self.config.slow_operation_threshold:
                logger.warning(f"Slow serialization: {execution_time:.3f}s for format {format}")
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Serialization error: {e}")
            raise
    
    async def deserialize(
        self, 
        data: bytes, 
        format: SerializationFormat = None,
        model_class: Type[T] = None,
        validate: bool = None
    ) -> Any:
        """Deserialize data with optimization."""
        start_time = time.time()
        format = format or self.config.default_format
        validate = validate if validate is not None else self.config.validate_on_deserialize
        
        try:
            # Deserialize
            handler = self._deserialization_handlers.get(format)
            if not handler:
                raise ValueError(f"Unsupported format: {format}")
            
            result = await handler(data, model_class)
            
            # Validate if needed
            if validate and model_class:
                if isinstance(result, dict):
                    result = model_class(**result)
                elif not isinstance(result, model_class):
                    raise ValueError(f"Deserialized data must be instance of {model_class}")
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics('deserialize', execution_time)
            
            # Log slow operations
            if self.config.log_slow_operations and execution_time > self.config.slow_operation_threshold:
                logger.warning(f"Slow deserialization: {execution_time:.3f}s for format {format}")
            
            return result
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def _serialize_json(self, data: Any, model_class: Type[T] = None) -> bytes:
        """Serialize to JSON."""
        if isinstance(data, BaseModel):
            json_str = data.model_dump_json()
        else:
            json_str = json.dumps(data, default=pydantic_encoder)
        
        return json_str.encode('utf-8')
    
    async def _serialize_orjson(self, data: Any, model_class: Type[T] = None) -> bytes:
        """Serialize to ORJSON (fastest)."""
        if isinstance(data, BaseModel):
            return orjson.dumps(data.model_dump())
        else:
            return orjson.dumps(data)
    
    async def _serialize_pickle(self, data: Any, model_class: Type[T] = None) -> bytes:
        """Serialize to Pickle."""
        return pickle.dumps(data)
    
    async def _serialize_compressed_json(self, data: Any, model_class: Type[T] = None) -> bytes:
        """Serialize to compressed JSON."""
        json_data = await self._serialize_json(data, model_class)
        
        if len(json_data) > self.config.compression_threshold:
            compressed = gzip.compress(json_data, compresslevel=self.config.compression_level.value)
            self.metrics.compression_operations += 1
            return b'gzip:' + compressed
        
        return json_data
    
    async def _serialize_compressed_orjson(self, data: Any, model_class: Type[T] = None) -> bytes:
        """Serialize to compressed ORJSON."""
        orjson_data = await self._serialize_orjson(data, model_class)
        
        if len(orjson_data) > self.config.compression_threshold:
            compressed = gzip.compress(orjson_data, compresslevel=self.config.compression_level.value)
            self.metrics.compression_operations += 1
            return b'gzip:' + compressed
        
        return orjson_data
    
    async def _deserialize_json(self, data: bytes, model_class: Type[T] = None) -> Any:
        """Deserialize from JSON."""
        json_str = data.decode('utf-8')
        result = json.loads(json_str)
        
        if model_class:
            result = model_class(**result)
        
        return result
    
    async def _deserialize_orjson(self, data: bytes, model_class: Type[T] = None) -> Any:
        """Deserialize from ORJSON."""
        result = orjson.loads(data)
        
        if model_class:
            result = model_class(**result)
        
        return result
    
    async def _deserialize_pickle(self, data: bytes, model_class: Type[T] = None) -> Any:
        """Deserialize from Pickle."""
        return pickle.loads(data)
    
    async def _deserialize_compressed_json(self, data: bytes, model_class: Type[T] = None) -> Any:
        """Deserialize from compressed JSON."""
        if data.startswith(b'gzip:'):
            compressed_data = data[5:]  # Remove 'gzip:' prefix
            json_data = gzip.decompress(compressed_data)
        else:
            json_data = data
        
        return await self._deserialize_json(json_data, model_class)
    
    async def _deserialize_compressed_orjson(self, data: bytes, model_class: Type[T] = None) -> Any:
        """Deserialize from compressed ORJSON."""
        if data.startswith(b'gzip:'):
            compressed_data = data[5:]  # Remove 'gzip:' prefix
            orjson_data = gzip.decompress(compressed_data)
        else:
            orjson_data = data
        
        return await self._deserialize_orjson(orjson_data, model_class)
    
    def _generate_cache_key(self, data: Any, format: SerializationFormat, model_class: Type[T], validate: bool) -> str:
        """Generate cache key for serialization."""
        key_data = {
            'data_hash': hashlib.md5(str(data).encode()).hexdigest(),
            'format': format.value,
            'model_class': model_class.__name__ if model_class else None,
            'validate': validate
        }
        return hashlib.md5(orjson.dumps(key_data)).hexdigest()
    
    def _update_metrics(self, operation: str, execution_time: float):
        """Update performance metrics."""
        self.metrics.total_operations += 1
        self.metrics.total_time += execution_time
        self.metrics.average_time = self.metrics.total_time / self.metrics.total_operations
        
        if operation == 'serialize':
            self.metrics.serialize_operations += 1
        elif operation == 'deserialize':
            self.metrics.deserialize_operations += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serialization metrics."""
        return {
            "total_operations": self.metrics.total_operations,
            "serialize_operations": self.metrics.serialize_operations,
            "deserialize_operations": self.metrics.deserialize_operations,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.total_operations - self.metrics.cache_hits,
            "cache_hit_rate": self.metrics.cache_hits / self.metrics.total_operations if self.metrics.total_operations > 0 else 0,
            "compression_operations": self.metrics.compression_operations,
            "validation_operations": self.metrics.validation_operations,
            "average_time": self.metrics.average_time,
            "errors": self.metrics.errors,
            "cache_size": len(self.serialization_cache)
        }
    
    def clear_cache(self) -> Any:
        """Clear serialization cache."""
        self.serialization_cache.clear()

class OptimizedPydanticModel(BaseModel):
    """
    Optimized Pydantic model with enhanced serialization capabilities.
    """
    
    model_config = ConfigDict(
        # Use orjson for fastest serialization
        json_loads=orjson.loads,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        
        # Performance optimizations
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        
        # Serialization optimizations
        use_enum_values=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True
    )
    
    def __init__(self, **data) -> Any:
        super().__init__(**data)
    
    def to_bytes(self, format: SerializationFormat = SerializationFormat.ORJSON) -> bytes:
        """Serialize model to bytes."""
        if format == SerializationFormat.ORJSON:
            return orjson.dumps(self.model_dump())
        elif format == SerializationFormat.JSON:
            return self.model_dump_json().encode('utf-8')
        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(self)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def from_bytes(cls: Type[T], data: bytes, format: SerializationFormat = SerializationFormat.ORJSON) -> T:
        """Deserialize model from bytes."""
        if format == SerializationFormat.ORJSON:
            return cls(**orjson.loads(data))
        elif format == SerializationFormat.JSON:
            return cls.model_validate_json(data.decode('utf-8'))
        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def to_dict(self, exclude_none: bool = True, exclude_defaults: bool = False) -> Dict[str, Any]:
        """Convert model to dictionary with optimizations."""
        return self.model_dump(
            exclude_none=exclude_none,
            exclude_defaults=exclude_defaults
        )
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model from dictionary."""
        return cls(**data)
    
    def to_json(self, indent: Optional[int] = None, exclude_none: bool = True) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json(
            indent=indent,
            exclude_none=exclude_none
        )
    
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create model from JSON string."""
        return cls.model_validate_json(json_str)

class SerializationManager:
    """
    Manager for handling serialization across the application.
    """
    
    def __init__(self, config: SerializationConfig = None):
        
    """__init__ function."""
self.config = config or SerializationConfig()
        self.serializer = OptimizedSerializer(self.config)
        self.model_registry = {}
        self.custom_serializers = {}
        self.custom_deserializers = {}
    
    def register_model(self, model_class: Type[T], alias: str = None):
        """Register a model class for optimized serialization."""
        alias = alias or model_class.__name__
        self.model_registry[alias] = model_class
        logger.info(f"Registered model: {alias} -> {model_class.__name__}")
    
    def register_custom_serializer(self, model_class: Type[T], serializer: Callable):
        """Register custom serializer for a model class."""
        self.custom_serializers[model_class] = serializer
        logger.info(f"Registered custom serializer for: {model_class.__name__}")
    
    def register_custom_deserializer(self, model_class: Type[T], deserializer: Callable):
        """Register custom deserializer for a model class."""
        self.custom_deserializers[model_class] = deserializer
        logger.info(f"Registered custom deserializer for: {model_class.__name__}")
    
    async def serialize_model(
        self, 
        model: T, 
        format: SerializationFormat = None,
        validate: bool = None
    ) -> bytes:
        """Serialize a Pydantic model."""
        # Check for custom serializer
        if type(model) in self.custom_serializers:
            return await self.custom_serializers[type(model)](model, format)
        
        return await self.serializer.serialize(model, format, type(model), validate)
    
    async def deserialize_model(
        self, 
        data: bytes, 
        model_class: Type[T],
        format: SerializationFormat = None,
        validate: bool = None
    ) -> T:
        """Deserialize data to a Pydantic model."""
        # Check for custom deserializer
        if model_class in self.custom_deserializers:
            return await self.custom_deserializers[model_class](data, format)
        
        return await self.serializer.deserialize(data, format, model_class, validate)
    
    async def serialize_batch(
        self, 
        models: List[T], 
        format: SerializationFormat = None,
        validate: bool = None
    ) -> List[bytes]:
        """Serialize a batch of models."""
        tasks = [
            self.serialize_model(model, format, validate)
            for model in models
        ]
        return await asyncio.gather(*tasks)
    
    async def deserialize_batch(
        self, 
        data_list: List[bytes], 
        model_class: Type[T],
        format: SerializationFormat = None,
        validate: bool = None
    ) -> List[T]:
        """Deserialize a batch of data to models."""
        tasks = [
            self.deserialize_model(data, model_class, format, validate)
            for data in data_list
        ]
        return await asyncio.gather(*tasks)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serialization metrics."""
        return {
            "serializer": self.serializer.get_metrics(),
            "model_registry": {
                "registered_models": len(self.model_registry),
                "custom_serializers": len(self.custom_serializers),
                "custom_deserializers": len(self.custom_deserializers)
            }
        }
    
    def clear_cache(self) -> Any:
        """Clear all caches."""
        self.serializer.clear_cache()

# Serialization decorators
def optimized_serialize(format: SerializationFormat = SerializationFormat.ORJSON):
    """Decorator for optimized serialization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            # Serialize result if it's a Pydantic model
            if isinstance(result, BaseModel):
                serializer = OptimizedSerializer(SerializationConfig())
                return await serializer.serialize(result, format)
            
            return result
        return wrapper
    return decorator

def optimized_deserialize(model_class: Type[T], format: SerializationFormat = SerializationFormat.ORJSON):
    """Decorator for optimized deserialization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)
            
            # Deserialize result if it's bytes
            if isinstance(result, bytes):
                serializer = OptimizedSerializer(SerializationConfig())
                return await serializer.deserialize(result, format, model_class)
            
            return result
        return wrapper
    return decorator

# Example usage
async def example_serialization_usage():
    """Example usage of optimized serialization."""
    
    # Create configuration
    config = SerializationConfig(
        default_format=SerializationFormat.ORJSON,
        enable_compression=True,
        enable_caching=True,
        enable_metrics=True
    )
    
    # Initialize serialization manager
    manager = SerializationManager(config)
    
    # Define optimized model
    class UserModel(OptimizedPydanticModel):
        id: int = Field(..., description="User ID")
        name: str = Field(..., min_length=1, max_length=100)
        email: str = Field(..., description="User email")
        is_active: bool = Field(default=True)
        created_at: float = Field(default_factory=time.time)
    
    # Register model
    manager.register_model(UserModel)
    
    # Create model instance
    user = UserModel(
        id=123,
        name="John Doe",
        email="john@example.com"
    )
    
    try:
        # Serialize model
        serialized_data = await manager.serialize_model(user)
        logger.info(f"Serialized data size: {len(serialized_data)} bytes")
        
        # Deserialize model
        deserialized_user = await manager.deserialize_model(serialized_data, UserModel)
        logger.info(f"Deserialized user: {deserialized_user}")
        
        # Test different formats
        formats = [
            SerializationFormat.JSON,
            SerializationFormat.ORJSON,
            SerializationFormat.COMPRESSED_JSON,
            SerializationFormat.COMPRESSED_ORJSON
        ]
        
        for format in formats:
            start_time = time.time()
            serialized = await manager.serialize_model(user, format)
            deserialized = await manager.deserialize_model(serialized, UserModel, format)
            execution_time = time.time() - start_time
            
            logger.info(f"Format {format.value}: {len(serialized)} bytes, {execution_time:.4f}s")
        
        # Test batch operations
        users = [
            UserModel(id=i, name=f"User {i}", email=f"user{i}@example.com")
            for i in range(10)
        ]
        
        start_time = time.time()
        serialized_batch = await manager.serialize_batch(users)
        deserialized_batch = await manager.deserialize_batch(serialized_batch, UserModel)
        batch_time = time.time() - start_time
        
        logger.info(f"Batch operation: {len(users)} users, {batch_time:.4f}s")
        
        # Get metrics
        metrics = manager.get_metrics()
        logger.info(f"Serialization metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Serialization error: {e}")

match __name__:
    case "__main__":
    asyncio.run(example_serialization_usage()) 