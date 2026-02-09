from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable, Generic
from datetime import datetime
import uuid
import logging
import time
from functools import lru_cache, wraps
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
import orjson
import structlog
import zstandard as zstd
import brotli
from cachetools import TTLCache, LRUCache
from typing import Any, List, Dict, Optional
import asyncio
"""
Optimized Base Model - Pydantic v2 Best Practices
Production-ready base model with performance optimizations, caching, and monitoring.
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T', bound='OptimizedBaseModel')

class PydanticMetrics:
    """Performance metrics collector for Pydantic models."""
    
    def __init__(self) -> Any:
        self.validation_times: List[tuple[str, float]] = []
        self.serialization_times: List[tuple[str, float]] = []
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.model_instances: defaultdict[str, int] = defaultdict(int)
    
    def record_validation_time(self, model_name: str, duration: float) -> None:
        """Record validation time for a model."""
        self.validation_times.append((model_name, duration))
        
        # Keep only last 1000 records to prevent memory bloat
        if len(self.validation_times) > 1000:
            self.validation_times = self.validation_times[-1000:]
    
    def record_serialization_time(self, model_name: str, duration: float) -> None:
        """Record serialization time for a model."""
        self.serialization_times.append((model_name, duration))
        
        if len(self.serialization_times) > 1000:
            self.serialization_times = self.serialization_times[-1000:]
    
    def record_error(self, model_name: str, error_type: str) -> None:
        """Record validation error."""
        self.error_counts[f"{model_name}:{error_type}"] += 1
    
    def record_instance(self, model_name: str) -> None:
        """Record model instance creation."""
        self.model_instances[model_name] += 1
    
    def get_average_validation_time(self, model_name: Optional[str] = None) -> float:
        """Get average validation time."""
        times = [t for name, t in self.validation_times if not model_name or name == model_name]
        return sum(times) / len(times) if times else 0.0
    
    def get_average_serialization_time(self, model_name: Optional[str] = None) -> float:
        """Get average serialization time."""
        times = [t for name, t in self.serialization_times if not model_name or name == model_name]
        return sum(times) / len(times) if times else 0.0
    
    def get_error_rate(self, model_name: Optional[str] = None) -> float:
        """Get error rate for a model."""
        total_instances = sum(self.model_instances.values())
        if total_instances == 0:
            return 0.0
        
        if model_name:
            errors = sum(count for key, count in self.error_counts.items() if key.startswith(model_name))
            instances = self.model_instances.get(model_name, 0)
            return errors / instances if instances > 0 else 0.0
        else:
            total_errors = sum(self.error_counts.values())
            return total_errors / total_instances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_models_created": sum(self.model_instances.values()),
            "average_validation_time_ms": self.get_average_validation_time() * 1000,
            "average_serialization_time_ms": self.get_average_serialization_time() * 1000,
            "overall_error_rate": self.get_error_rate(),
            "model_instances": dict(self.model_instances),
            "error_counts": dict(self.error_counts)
        }

# Global metrics instance
_metrics = PydanticMetrics()

def measure_validation_time(func: Callable) -> Callable:
    """Decorator to measure validation time."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Extract model name from class
            model_name = args[0].__class__.__name__ if args else "Unknown"
            _metrics.record_validation_time(model_name, end_time - start_time)
            
            return result
        except Exception as e:
            end_time = time.perf_counter()
            model_name = args[0].__class__.__name__ if args else "Unknown"
            _metrics.record_error(model_name, type(e).__name__)
            _metrics.record_validation_time(model_name, end_time - start_time)
            raise
    
    return wrapper

class OptimizedBaseModel(BaseModel):
    """
    Optimized base model with Pydantic v2 best practices and ultra-fast cache integration.
    Features:
    - ORJSON integration for fastest serialization
    - Performance monitoring and metrics
    - UltraFastCache (memory + Redis + compression)
    - Caching support for expensive validations
    - Standardized error handling
    - Memory optimization
    - Type safety enhancements
    - Strict field validation
    - Field documentation for OpenAPI
    """
    model_config = ConfigDict(
        json_loads=orjson.loads,
        json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_default=True,
        revalidate_instances="subclass-instances",
        strict=True,
        title="OptimizedBaseModel",
        description="Optimized base model for Onyx with strict validation, ultra-fast cache, and performance metrics."
    )
    _metrics: ClassVar[PydanticMetrics] = _metrics
    _cache: ClassVar[TTLCache] = TTLCache(maxsize=10000, ttl=3600)
    _logger: ClassVar[Any] = logger

    @classmethod
    def cache_set(cls, key: str, value: Any, ttl: int = 3600):
        """Set value in ultra-fast memory cache (TTLCache)."""
        cls._cache[key] = value
        cls._logger.debug("Cache set", key=key)

    @classmethod
    def cache_get(cls, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get value from ultra-fast memory cache (TTLCache)."""
        value = cls._cache.get(key, default)
        cls._logger.debug("Cache get", key=key, hit=value is not None)
        return value

    @classmethod
    def cache_clear(cls) -> Any:
        """Clear the ultra-fast memory cache."""
        cls._cache.clear()
        cls._logger.info("Cache cleared")

    @classmethod
    def compress_data(cls, data: bytes, algorithm: str = "zstd") -> bytes:
        if algorithm == "zstd":
            compressor = zstd.ZstdCompressor(level=3)
            return compressor.compress(data)
        elif algorithm == "brotli":
            return brotli.compress(data, quality=4)
        return data

    @classmethod
    def decompress_data(cls, data: bytes, algorithm: str = "zstd") -> bytes:
        if algorithm == "zstd":
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        elif algorithm == "brotli":
            return brotli.decompress(data)
        return data

    @classmethod
    def get_performance_stats(cls) -> Dict[str, Any]:
        """Get performance statistics for this model and cache."""
        stats = {
            "model_name": cls.__name__,
            "field_count": len(getattr(cls, 'model_fields', {})),
            "average_validation_time_ms": cls._metrics.get_average_validation_time(cls.__name__) * 1000,
            "average_serialization_time_ms": cls._metrics.get_average_serialization_time(cls.__name__) * 1000,
            "cache_size": len(cls._cache),
        }
        stats.update(cls._metrics.get_stats())
        return stats

    def __init__(self, **data: Any) -> None:
        """Initialize model with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            super().__init__(**data)
            
            # Record instance creation
            self._metrics.record_instance(self.__class__.__name__)
            
            # Log successful creation
            logger.debug(
                "Model instantiated successfully",
                model=self.__class__.__name__,
                field_count=len(self.model_fields),
                duration_ms=(time.perf_counter() - start_time) * 1000
            )
            
        except Exception as e:
            # Record error
            self._metrics.record_error(self.__class__.__name__, type(e).__name__)
            
            logger.error(
                "Model instantiation failed",
                model=self.__class__.__name__,
                error=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000
            )
            raise
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook for monitoring."""
        super().model_post_init(__context)
        
        logger.debug(
            "Model post-init completed",
            model=self.__class__.__name__,
            context=__context
        )
    
    @measure_validation_time
    def model_validate(self, obj: Any, *args, **kwargs) -> T:
        """Override validation with performance tracking."""
        return super().model_validate(obj, *args, **kwargs)
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Override serialization with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            result = super().model_dump(*args, **kwargs)
            
            # Record serialization time
            self._metrics.record_serialization_time(
                self.__class__.__name__,
                time.perf_counter() - start_time
            )
            
            return result
            
        except Exception as e:
            # Record error
            self._metrics.record_error(self.__class__.__name__, type(e).__name__)
            raise
    
    def model_dump_json(self, *args, **kwargs) -> str:
        """Override JSON serialization with performance tracking."""
        start_time = time.perf_counter()
        
        try:
            result = super().model_dump_json(*args, **kwargs)
            
            # Record serialization time
            self._metrics.record_serialization_time(
                self.__class__.__name__,
                time.perf_counter() - start_time
            )
            
            return result
            
        except Exception as e:
            # Record error
            self._metrics.record_error(self.__class__.__name__, type(e).__name__)
            raise
    
    @classmethod
    def get_metrics(cls) -> PydanticMetrics:
        """Get performance metrics."""
        return cls._metrics

class CachedOptimizedModel(OptimizedBaseModel):
    """
    Optimized model with built-in caching for expensive operations.
    """
    
    model_config = ConfigDict(
        # Inherit from OptimizedBaseModel
        **OptimizedBaseModel.model_config,
        
        # Additional caching optimizations
        validate_assignment=False,  # Disable for better performance
        extra="ignore"  # More permissive for caching
    )
    
    @classmethod
    @lru_cache(maxsize=128)
    def _cached_validation(cls, value: str) -> str:
        """Cached validation for expensive operations."""
        # Example: expensive validation logic
        return value.strip().lower()
    
    @field_validator("cached_field")
    @classmethod
    def validate_cached_field(cls, v: str) -> str:
        """Use cached validation for expensive operations."""
        return cls._cached_validation(v)

class FrozenOptimizedModel(OptimizedBaseModel):
    """
    Immutable optimized model for configuration and constants.
    """
    
    model_config = ConfigDict(
        # Inherit from OptimizedBaseModel
        **OptimizedBaseModel.model_config,
        
        # Immutability
        frozen=True,
        validate_assignment=False
    )

class TimestampedOptimizedModel(OptimizedBaseModel):
    """
    Optimized model with automatic timestamp management.
    """
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("updated_at", mode="before")
    @classmethod
    def set_updated_at(cls, v: Any) -> datetime:
        """Always set updated_at to current time."""
        return datetime.utcnow()
    
    @computed_field
    @property
    def age_seconds(self) -> float:
        """Compute age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @computed_field
    @property
    def is_recent(self) -> bool:
        """Check if model was created recently (within 1 hour)."""
        return self.age_seconds < 3600

class OptimizedGenericModel(OptimizedBaseModel, Generic[T]):
    """
    Generic optimized model for type-safe collections.
    """
    
    data: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @computed_field
    @property
    def data_type(self) -> str:
        """Get the type name of the data."""
        return type(self.data).__name__
    
    @computed_field
    @property
    def metadata_keys(self) -> List[str]:
        """Get metadata keys."""
        return list(self.metadata.keys())

# Utility functions for model optimization

def create_optimized_model(
    name: str,
    fields: Dict[str, Any],
    base_class: Type[OptimizedBaseModel] = OptimizedBaseModel,
    **kwargs
) -> Type[OptimizedBaseModel]:
    """
    Dynamically create an optimized model.
    
    Args:
        name: Model class name
        fields: Field definitions
        base_class: Base class to inherit from
        **kwargs: Additional model configuration
    
    Returns:
        Optimized model class
    """
    
    # Create field definitions
    field_definitions = {}
    for field_name, field_type in fields.items():
        if isinstance(field_type, tuple):
            field_type, field_config = field_type
            field_definitions[field_name] = (field_type, Field(**field_config))
        else:
            field_definitions[field_name] = (field_type, Field())
    
    # Create model class
    model_class = type(
        name,
        (base_class,),
        {
            "__annotations__": {name: type_ for name, (type_, _) in field_definitions.items()},
            **{name: field for name, (_, field) in field_definitions.items()},
            **kwargs
        }
    )
    
    return model_class

def optimize_existing_model(model_class: Type[BaseModel]) -> Type[OptimizedBaseModel]:
    """
    Convert an existing model to use OptimizedBaseModel.
    
    Args:
        model_class: Existing model class
    
    Returns:
        Optimized version of the model
    """
    
    # Extract fields and annotations
    fields = {}
    annotations = {}
    
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation
        field_config = {
            "default": field_info.default,
            "default_factory": field_info.default_factory,
            "alias": field_info.alias,
            "description": field_info.description,
            "exclude": field_info.exclude,
            "gt": field_info.gt,
            "ge": field_info.ge,
            "lt": field_info.lt,
            "le": field_info.le,
            "multiple_of": field_info.multiple_of,
            "min_length": field_info.min_length,
            "max_length": field_info.max_length,
            "pattern": field_info.pattern,
            "regex": field_info.regex,
        }
        
        # Remove None values
        field_config = {k: v for k, v in field_config.items() if v is not None}
        
        fields[field_name] = (field_type, field_config)
        annotations[field_name] = field_type
    
    # Create optimized model
    optimized_class = type(
        f"Optimized{model_class.__name__}",
        (OptimizedBaseModel,),
        {
            "__annotations__": annotations,
            **{name: Field(**config) for name, (_, config) in fields.items()}
        }
    )
    
    return optimized_class

# Performance monitoring utilities

def get_global_metrics() -> PydanticMetrics:
    """Get global Pydantic metrics."""
    return _metrics

def reset_metrics() -> None:
    """Reset all metrics."""
    global _metrics
    _metrics = PydanticMetrics()

def log_performance_report() -> None:
    """Log comprehensive performance report."""
    stats = _metrics.get_stats()
    
    logger.info(
        "Pydantic Performance Report",
        total_models=stats["total_models_created"],
        avg_validation_ms=stats["average_validation_time_ms"],
        avg_serialization_ms=stats["average_serialization_time_ms"],
        error_rate=stats["overall_error_rate"],
        model_instances=stats["model_instances"],
        error_counts=stats["error_counts"]
    )

# Example usage and best practices

class ExampleOptimizedModel(OptimizedBaseModel):
    """Example of optimized model usage."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and normalize name."""
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        return v.lower().strip()
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Computed display name."""
        return f"{self.name} ({self.id[:8]})"
    
    @computed_field
    @property
    def tag_count(self) -> int:
        """Number of tags."""
        return len(self.tags)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the model."""
        if tag in self.tags:
            self.tags.remove(tag)

# Export main classes and utilities
__all__ = [
    "OptimizedBaseModel",
    "CachedOptimizedModel", 
    "FrozenOptimizedModel",
    "TimestampedOptimizedModel",
    "OptimizedGenericModel",
    "PydanticMetrics",
    "create_optimized_model",
    "optimize_existing_model",
    "get_global_metrics",
    "reset_metrics",
    "log_performance_report",
    "measure_validation_time"
] 