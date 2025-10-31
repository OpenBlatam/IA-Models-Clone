from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic, Type
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
from pydantic.json import pydantic_encoder, ENCODERS_BY_TYPE
from pydantic.types import Json
import orjson
import ujson
import msgpack
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Pydantic Serialization Optimizer for HeyGen AI API
Optimized data serialization and deserialization with Pydantic.
"""



logger = structlog.get_logger()

# =============================================================================
# Serialization Types
# =============================================================================

class SerializationFormat(Enum):
    """Serialization format enumeration."""
    JSON = "json"
    ORJSON = "orjson"
    UJSON = "ujson"
    MSGPACK = "msgpack"
    PICKLE = "pickle"
    CUSTOM = "custom"

class SerializationStrategy(Enum):
    """Serialization strategy enumeration."""
    FAST = "fast"
    COMPACT = "compact"
    COMPATIBLE = "compatible"
    VALIDATED = "validated"
    CACHED = "cached"

class SerializationPriority(Enum):
    """Serialization priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SerializationConfig:
    """Serialization configuration."""
    format: SerializationFormat = SerializationFormat.ORJSON
    strategy: SerializationStrategy = SerializationStrategy.FAST
    priority: SerializationPriority = SerializationPriority.NORMAL
    enable_caching: bool = True
    enable_compression: bool = False
    enable_validation: bool = True
    enable_optimization: bool = True
    cache_size: int = 1000
    compression_level: int = 6
    max_depth: int = 10
    exclude_none: bool = True
    exclude_unset: bool = True
    exclude_defaults: bool = False
    use_enum_values: bool = True
    use_custom_encoders: bool = True
    enable_type_hints: bool = True
    enable_field_aliases: bool = True
    enable_root_validators: bool = True
    enable_validators: bool = True
    enable_extra_fields: bool = False
    enable_populate_by_name: bool = True

@dataclass
class SerializationStats:
    """Serialization statistics."""
    serializations: int = 0
    deserializations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_errors: int = 0
    total_serialization_time_ms: float = 0.0
    total_deserialization_time_ms: float = 0.0
    average_serialization_time_ms: float = 0.0
    average_deserialization_time_ms: float = 0.0
    last_serialization: Optional[datetime] = None
    last_deserialization: Optional[datetime] = None
    created_at: datetime = None
    
    def __post_init__(self) -> Any:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def update_serialization_stats(self, duration_ms: float, cache_hit: bool = False):
        """Update serialization statistics."""
        self.serializations += 1
        self.total_serialization_time_ms += duration_ms
        self.average_serialization_time_ms = self.total_serialization_time_ms / self.serializations
        self.last_serialization = datetime.now(timezone.utc)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def update_deserialization_stats(self, duration_ms: float, cache_hit: bool = False):
        """Update deserialization statistics."""
        self.deserializations += 1
        self.total_deserialization_time_ms += duration_ms
        self.average_deserialization_time_ms = self.total_deserialization_time_ms / self.deserializations
        self.last_deserialization = datetime.now(timezone.utc)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

# =============================================================================
# Optimized Pydantic Models
# =============================================================================

class OptimizedBaseModel(BaseModel):
    """Optimized base model with performance enhancements."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
        extra='ignore',
        json_encoders={
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        },
        # Serialization optimizations
        json_schema_extra={
            "example": {},
            "examples": []
        },
        # Validation optimizations
        validate_default=True,
        str_strip_whitespace=True,
        str_min_length=0,
        str_max_length=None,
    )
    
    def __init__(self, **data) -> Any:
        super().__init__(**data)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Optimized model dump with caching."""
        return super().model_dump(**kwargs)
    
    def model_dump_json(self, **kwargs) -> str:
        """Optimized model dump to JSON."""
        return super().model_dump_json(**kwargs)
    
    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> "OptimizedBaseModel":
        """Optimized model validation."""
        return super().model_validate(obj, **kwargs)
    
    @classmethod
    def model_validate_json(cls, json_str: str, **kwargs) -> "OptimizedBaseModel":
        """Optimized model validation from JSON."""
        return super().model_validate_json(json_str, **kwargs)

class FastSerializationModel(OptimizedBaseModel):
    """Fast serialization model with minimal validation."""
    
    model_config = ConfigDict(
        validate_assignment=False,
        validate_default=False,
        extra='ignore',
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        },
        # Disable expensive validations
        str_strip_whitespace=False,
        str_min_length=None,
        str_max_length=None,
    )

class CompactSerializationModel(OptimizedBaseModel):
    """Compact serialization model with field optimization."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
        extra='ignore',
        exclude_none=True,
        exclude_unset=True,
        exclude_defaults=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        },
    )

class ValidatedSerializationModel(OptimizedBaseModel):
    """Validated serialization model with full validation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
        extra='forbid',
        validate_default=True,
        str_strip_whitespace=True,
        str_min_length=0,
        str_max_length=None,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        },
    )

# =============================================================================
# Serialization Encoders
# =============================================================================

class OptimizedJSONEncoder:
    """Optimized JSON encoder with performance enhancements."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.encoders = self._build_encoders()
    
    def _build_encoders(self) -> Dict[Type, Callable]:
        """Build optimized encoders."""
        encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
            Enum: lambda v: v.value if self.config.use_enum_values else v.name,
        }
        
        if self.config.use_custom_encoders:
            encoders.update(ENCODERS_BY_TYPE)
        
        return encoders
    
    def encode(self, obj: Any) -> Union[str, bytes]:
        """Encode object based on format."""
        if self.config.format == SerializationFormat.ORJSON:
            return self._encode_orjson(obj)
        elif self.config.format == SerializationFormat.UJSON:
            return self._encode_ujson(obj)
        elif self.config.format == SerializationFormat.MSGPACK:
            return self._encode_msgpack(obj)
        elif self.config.format == SerializationFormat.PICKLE:
            return self._encode_pickle(obj)
        else:
            return self._encode_json(obj)
    
    def _encode_orjson(self, obj: Any) -> bytes:
        """Encode using orjson (fastest)."""
        options = orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY
        
        if self.config.exclude_none:
            options |= orjson.OPT_OMIT_MICROSECONDS
        
        return orjson.dumps(obj, option=options, default=self._default_encoder)
    
    def _encode_ujson(self, obj: Any) -> str:
        """Encode using ujson (fast)."""
        return ujson.dumps(obj, default=self._default_encoder)
    
    def _encode_msgpack(self, obj: Any) -> bytes:
        """Encode using msgpack (compact)."""
        return msgpack.packb(obj, default=self._default_encoder, use_bin_type=True)
    
    def _encode_pickle(self, obj: Any) -> bytes:
        """Encode using pickle (Python-specific)."""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _encode_json(self, obj: Any) -> str:
        """Encode using standard json."""
        return json.dumps(obj, default=self._default_encoder, separators=(',', ':'))
    
    def _default_encoder(self, obj: Any) -> Any:
        """Default encoder for unsupported types."""
        for obj_type, encoder in self.encoders.items():
            if isinstance(obj, obj_type):
                return encoder(obj)
        
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

class OptimizedJSONDecoder:
    """Optimized JSON decoder with performance enhancements."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
    
    def decode(self, data: Union[str, bytes]) -> Any:
        """Decode data based on format."""
        if self.config.format == SerializationFormat.ORJSON:
            return self._decode_orjson(data)
        elif self.config.format == SerializationFormat.UJSON:
            return self._decode_ujson(data)
        elif self.config.format == SerializationFormat.MSGPACK:
            return self._decode_msgpack(data)
        elif self.config.format == SerializationFormat.PICKLE:
            return self._decode_pickle(data)
        else:
            return self._decode_json(data)
    
    def _decode_orjson(self, data: bytes) -> Any:
        """Decode using orjson."""
        return orjson.loads(data)
    
    def _decode_ujson(self, data: str) -> Any:
        """Decode using ujson."""
        return ujson.loads(data)
    
    def _decode_msgpack(self, data: bytes) -> Any:
        """Decode using msgpack."""
        return msgpack.unpackb(data, raw=False)
    
    def _decode_pickle(self, data: bytes) -> Any:
        """Decode using pickle."""
        return pickle.loads(data)
    
    def _decode_json(self, data: str) -> Any:
        """Decode using standard json."""
        return json.loads(data)

# =============================================================================
# Serialization Cache
# =============================================================================

class SerializationCache:
    """Cache for serialized/deserialized data."""
    
    def __init__(self, max_size: int = 1000):
        
    """__init__ function."""
self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    def _generate_key(self, obj: Any, format_type: SerializationFormat) -> str:
        """Generate cache key for object."""
        if hasattr(obj, 'model_dump'):
            # For Pydantic models
            data = obj.model_dump()
        elif isinstance(obj, dict):
            data = obj
        else:
            data = str(obj)
        
        key_data = {
            "data": data,
            "format": format_type.value,
            "type": type(obj).__name__
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def get(self, obj: Any, format_type: SerializationFormat) -> Optional[Any]:
        """Get cached serialization."""
        async with self._lock:
            key = self._generate_key(obj, format_type)
            
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            
            return None
    
    async def set(self, obj: Any, format_type: SerializationFormat, serialized: Any):
        """Set cached serialization."""
        async with self._lock:
            key = self._generate_key(obj, format_type)
            
            # Add to cache
            self._cache[key] = serialized
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Evict if needed
            if len(self._cache) > self.max_size:
                oldest_key = self._access_order[0]
                del self._cache[oldest_key]
                self._access_order.pop(0)
    
    async def clear(self) -> Any:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0
        }

# =============================================================================
# Pydantic Serialization Optimizer
# =============================================================================

class PydanticSerializationOptimizer:
    """Main Pydantic serialization optimizer."""
    
    def __init__(self, config: SerializationConfig):
        
    """__init__ function."""
self.config = config
        self.encoder = OptimizedJSONEncoder(config)
        self.decoder = OptimizedJSONDecoder(config)
        self.cache = SerializationCache(config.cache_size) if config.enable_caching else None
        self.stats = SerializationStats()
        
        # Model registry for optimization
        self._model_registry: Dict[str, Type[OptimizedBaseModel]] = {}
        self._serialization_cache: Dict[str, Callable] = {}
        self._deserialization_cache: Dict[str, Callable] = {}
    
    def register_model(self, model_class: Type[OptimizedBaseModel], alias: Optional[str] = None):
        """Register a model for optimization."""
        model_name = alias or model_class.__name__
        self._model_registry[model_name] = model_class
        
        # Pre-compile serialization methods
        if self.config.enable_optimization:
            self._serialization_cache[model_name] = self._compile_serialization_method(model_class)
            self._deserialization_cache[model_name] = self._compile_deserialization_method(model_class)
    
    def _compile_serialization_method(self, model_class: Type[OptimizedBaseModel]) -> Callable:
        """Compile optimized serialization method."""
        if self.config.strategy == SerializationStrategy.FAST:
            return lambda obj: obj.model_dump(
                exclude_none=self.config.exclude_none,
                exclude_unset=self.config.exclude_unset,
                exclude_defaults=self.config.exclude_defaults,
                use_enum_values=self.config.use_enum_values
            )
        elif self.config.strategy == SerializationStrategy.COMPACT:
            return lambda obj: obj.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                use_enum_values=True
            )
        else:
            return lambda obj: obj.model_dump()
    
    def _compile_deserialization_method(self, model_class: Type[OptimizedBaseModel]) -> Callable:
        """Compile optimized deserialization method."""
        if self.config.enable_validation:
            return lambda data: model_class.model_validate(data)
        else:
            return lambda data: model_class(**data)
    
    async def serialize(self, obj: Any) -> Union[str, bytes]:
        """Serialize object with optimization."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache and self.config.enable_caching:
                cached_result = await self.cache.get(obj, self.config.format)
                if cached_result is not None:
                    self.stats.update_serialization_stats(0, cache_hit=True)
                    return cached_result
            
            # Serialize object
            if isinstance(obj, OptimizedBaseModel):
                serialized = await self._serialize_model(obj)
            elif isinstance(obj, dict):
                serialized = self.encoder.encode(obj)
            elif isinstance(obj, list):
                serialized = self.encoder.encode(obj)
            else:
                serialized = self.encoder.encode(obj)
            
            # Cache result
            if self.cache and self.config.enable_caching:
                await self.cache.set(obj, self.config.format, serialized)
            
            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            self.stats.update_serialization_stats(duration_ms, cache_hit=False)
            
            return serialized
            
        except Exception as e:
            self.stats.validation_errors += 1
            logger.error(f"Serialization error: {e}")
            raise
    
    async def _serialize_model(self, obj: OptimizedBaseModel) -> Union[str, bytes]:
        """Serialize Pydantic model with optimization."""
        model_name = type(obj).__name__
        
        # Use compiled method if available
        if model_name in self._serialization_cache:
            data = self._serialization_cache[model_name](obj)
        else:
            data = obj.model_dump(
                exclude_none=self.config.exclude_none,
                exclude_unset=self.config.exclude_unset,
                exclude_defaults=self.config.exclude_defaults,
                use_enum_values=self.config.use_enum_values
            )
        
        return self.encoder.encode(data)
    
    async def deserialize(self, data: Union[str, bytes], model_class: Optional[Type[OptimizedBaseModel]] = None) -> Any:
        """Deserialize data with optimization."""
        start_time = time.time()
        
        try:
            # Decode data
            decoded_data = self.decoder.decode(data)
            
            # Validate and convert to model if specified
            if model_class:
                if type(model_class).__name__ in self._deserialization_cache:
                    result = self._deserialization_cache[type(model_class).__name__](decoded_data)
                else:
                    if self.config.enable_validation:
                        result = model_class.model_validate(decoded_data)
                    else:
                        result = model_class(**decoded_data)
            else:
                result = decoded_data
            
            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            self.stats.update_deserialization_stats(duration_ms, cache_hit=False)
            
            return result
            
        except Exception as e:
            self.stats.validation_errors += 1
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def batch_serialize(self, objects: List[Any]) -> List[Union[str, bytes]]:
        """Serialize multiple objects efficiently."""
        tasks = [self.serialize(obj) for obj in objects]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_deserialize(self, data_list: List[Union[str, bytes]], model_class: Optional[Type[OptimizedBaseModel]] = None) -> List[Any]:
        """Deserialize multiple objects efficiently."""
        tasks = [self.deserialize(data, model_class) for data in data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics."""
        return {
            "serializations": self.stats.serializations,
            "deserializations": self.stats.deserializations,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "validation_errors": self.stats.validation_errors,
            "average_serialization_time_ms": self.stats.average_serialization_time_ms,
            "average_deserialization_time_ms": self.stats.average_deserialization_time_ms,
            "cache_stats": self.cache.get_stats() if self.cache else None,
            "registered_models": list(self._model_registry.keys()),
            "config": {
                "format": self.config.format.value,
                "strategy": self.config.strategy.value,
                "enable_caching": self.config.enable_caching,
                "enable_validation": self.config.enable_validation
            }
        }
    
    async def clear_cache(self) -> Any:
        """Clear serialization cache."""
        if self.cache:
            await self.cache.clear()
    
    async def optimize_memory(self) -> Any:
        """Optimize memory usage."""
        # Clear caches
        await self.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Rebuild caches if needed
        if self.config.enable_caching:
            self.cache = SerializationCache(self.config.cache_size)

# =============================================================================
# Serialization Decorators
# =============================================================================

def optimized_serialization(
    format_type: SerializationFormat = SerializationFormat.ORJSON,
    strategy: SerializationStrategy = SerializationStrategy.FAST,
    enable_caching: bool = True
):
    """Decorator for optimized serialization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get optimizer from function context or create new one
            optimizer = getattr(wrapper, '_optimizer', None)
            if not optimizer:
                config = SerializationConfig(
                    format=format_type,
                    strategy=strategy,
                    enable_caching=enable_caching
                )
                optimizer = PydanticSerializationOptimizer(config)
                wrapper._optimizer = optimizer
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Serialize result
            serialized_result = await optimizer.serialize(result)
            
            return serialized_result
        
        return wrapper
    return decorator

def optimized_deserialization(
    model_class: Optional[Type[OptimizedBaseModel]] = None,
    format_type: SerializationFormat = SerializationFormat.ORJSON,
    strategy: SerializationStrategy = SerializationStrategy.FAST
):
    """Decorator for optimized deserialization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Get optimizer from function context or create new one
            optimizer = getattr(wrapper, '_optimizer', None)
            if not optimizer:
                config = SerializationConfig(
                    format=format_type,
                    strategy=strategy,
                    enable_caching=True
                )
                optimizer = PydanticSerializationOptimizer(config)
                wrapper._optimizer = optimizer
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Deserialize result
            deserialized_result = await optimizer.deserialize(result, model_class)
            
            return deserialized_result
        
        return wrapper
    return decorator

def cached_serialization(
    ttl: int = 300,
    format_type: SerializationFormat = SerializationFormat.ORJSON
):
    """Decorator for cached serialization."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key_data = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items())),
                "format": format_type.value
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Check cache
            cache = getattr(wrapper, '_cache', {})
            if cache_key in cache:
                cached_time, cached_result = cache[cache_key]
                if time.time() - cached_time < ttl:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = (time.time(), result)
            wrapper._cache = cache
            
            return result
        
        return wrapper
    return decorator

# =============================================================================
# Model Optimization Utilities
# =============================================================================

def optimize_model_fields(model_class: Type[OptimizedBaseModel]) -> Type[OptimizedBaseModel]:
    """Optimize model fields for better performance."""
    
    # Get model fields
    fields = model_class.model_fields
    
    for field_name, field in fields.items():
        # Optimize field configuration
        if field.annotation == str:
            # Optimize string fields
            field.json_schema_extra = {
                "min_length": 0,
                "max_length": None,
                "strip_whitespace": False
            }
        elif field.annotation == int:
            # Optimize integer fields
            field.json_schema_extra = {
                "minimum": None,
                "maximum": None
            }
        elif field.annotation == float:
            # Optimize float fields
            field.json_schema_extra = {
                "minimum": None,
                "maximum": None
            }
    
    return model_class

def create_optimized_model(
    base_class: Type[OptimizedBaseModel] = OptimizedBaseModel,
    **field_definitions
) -> Type[OptimizedBaseModel]:
    """Create an optimized model dynamically."""
    
    # Create field definitions
    fields = {}
    for field_name, field_type in field_definitions.items():
        fields[field_name] = (field_type, Field())
    
    # Create model class
    model_class = type(
        "OptimizedModel",
        (base_class,),
        {
            "__annotations__": fields,
            **fields
        }
    )
    
    # Optimize the model
    return optimize_model_fields(model_class)

# =============================================================================
# Performance Monitoring
# =============================================================================

class SerializationPerformanceMonitor:
    """Monitor serialization performance."""
    
    def __init__(self) -> Any:
        self.metrics: Dict[str, List[float]] = {
            "serialization_times": [],
            "deserialization_times": [],
            "cache_hit_rates": [],
            "memory_usage": []
        }
    
    def record_serialization(self, duration_ms: float):
        """Record serialization time."""
        self.metrics["serialization_times"].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.metrics["serialization_times"]) > 1000:
            self.metrics["serialization_times"] = self.metrics["serialization_times"][-1000:]
    
    def record_deserialization(self, duration_ms: float):
        """Record deserialization time."""
        self.metrics["deserialization_times"].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self.metrics["deserialization_times"]) > 1000:
            self.metrics["deserialization_times"] = self.metrics["deserialization_times"][-1000:]
    
    def record_cache_hit_rate(self, hit_rate: float):
        """Record cache hit rate."""
        self.metrics["cache_hit_rates"].append(hit_rate)
        
        # Keep only last 100 measurements
        if len(self.metrics["cache_hit_rates"]) > 100:
            self.metrics["cache_hit_rates"] = self.metrics["cache_hit_rates"][-100:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        serialization_times = self.metrics["serialization_times"]
        deserialization_times = self.metrics["deserialization_times"]
        cache_hit_rates = self.metrics["cache_hit_rates"]
        
        return {
            "serialization": {
                "count": len(serialization_times),
                "average_ms": sum(serialization_times) / len(serialization_times) if serialization_times else 0,
                "min_ms": min(serialization_times) if serialization_times else 0,
                "max_ms": max(serialization_times) if serialization_times else 0,
                "p95_ms": sorted(serialization_times)[int(len(serialization_times) * 0.95)] if serialization_times else 0
            },
            "deserialization": {
                "count": len(deserialization_times),
                "average_ms": sum(deserialization_times) / len(deserialization_times) if deserialization_times else 0,
                "min_ms": min(deserialization_times) if deserialization_times else 0,
                "max_ms": max(deserialization_times) if deserialization_times else 0,
                "p95_ms": sorted(deserialization_times)[int(len(deserialization_times) * 0.95)] if deserialization_times else 0
            },
            "cache": {
                "average_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
                "min_hit_rate": min(cache_hit_rates) if cache_hit_rates else 0,
                "max_hit_rate": max(cache_hit_rates) if cache_hit_rates else 0
            }
        }

# =============================================================================
# FastAPI Integration
# =============================================================================

def get_serialization_optimizer() -> PydanticSerializationOptimizer:
    """Dependency to get serialization optimizer."""
    config = SerializationConfig(
        format=SerializationFormat.ORJSON,
        strategy=SerializationStrategy.FAST,
        enable_caching=True,
        enable_validation=True
    )
    return PydanticSerializationOptimizer(config)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "SerializationFormat",
    "SerializationStrategy",
    "SerializationPriority",
    "SerializationConfig",
    "SerializationStats",
    "OptimizedBaseModel",
    "FastSerializationModel",
    "CompactSerializationModel",
    "ValidatedSerializationModel",
    "OptimizedJSONEncoder",
    "OptimizedJSONDecoder",
    "SerializationCache",
    "PydanticSerializationOptimizer",
    "optimized_serialization",
    "optimized_deserialization",
    "cached_serialization",
    "optimize_model_fields",
    "create_optimized_model",
    "SerializationPerformanceMonitor",
    "get_serialization_optimizer",
] 