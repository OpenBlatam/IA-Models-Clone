from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Iterator
from datetime import datetime, timezone, date
from decimal import Decimal
import json
import base64
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import asyncio
from dataclasses import asdict, is_dataclass
from pydantic import (
from pydantic_core import core_schema, PydanticCustomError
from pydantic.json_schema import JsonSchemaValue
import structlog
        from urllib.parse import urlunparse
        import time
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Pydantic v2 Serializers for HeyGen AI API
Advanced serialization with custom logic, performance optimizations, and comprehensive features.
"""


    PlainSerializer, PlainValidator, BeforeValidator, AfterValidator,
    WithJsonSchema, GetJsonSchemaHandler, GetCoreSchemaHandler,
    SerializerFunctionWrapHandler, SerializationInfo
)

logger = structlog.get_logger()

T = TypeVar('T')

# =============================================================================
# Custom Serialization Errors
# =============================================================================

class SerializationErrorCode:
    """Serialization error codes for consistent error handling."""
    
    # Data serialization
    INVALID_DATA_TYPE = "INVALID_DATA_TYPE"
    SERIALIZATION_FAILED = "SERIALIZATION_FAILED"
    DESERIALIZATION_FAILED = "DESERIALIZATION_FAILED"
    
    # File serialization
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    
    # URL serialization
    INVALID_URL = "INVALID_URL"
    URL_TOO_LONG = "URL_TOO_LONG"
    
    # Binary serialization
    BINARY_TOO_LARGE = "BINARY_TOO_LARGE"
    INVALID_BINARY_FORMAT = "INVALID_BINARY_FORMAT"


# =============================================================================
# DateTime Serializers
# =============================================================================

def serialize_datetime_iso(v: datetime) -> str:
    """Serialize datetime to ISO format with timezone handling."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def serialize_datetime_unix(v: datetime) -> int:
    """Serialize datetime to Unix timestamp."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return int(v.timestamp())


def serialize_datetime_readable(v: datetime) -> str:
    """Serialize datetime to human-readable format."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.strftime("%Y-%m-%d %H:%M:%S UTC")


def deserialize_datetime_iso(v: str) -> datetime:
    """Deserialize datetime from ISO format."""
    try:
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid datetime format: {e}"
        )


def deserialize_datetime_unix(v: Union[int, float]) -> datetime:
    """Deserialize datetime from Unix timestamp."""
    try:
        return datetime.fromtimestamp(v, tz=timezone.utc)
    except (ValueError, OSError) as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid Unix timestamp: {e}"
        )


# =============================================================================
# Decimal Serializers
# =============================================================================

def serialize_decimal_float(v: Decimal) -> float:
    """Serialize Decimal to float."""
    return float(v)


def serialize_decimal_string(v: Decimal) -> str:
    """Serialize Decimal to string with precision control."""
    return str(v.quantize(Decimal('0.01')))


def serialize_decimal_int(v: Decimal) -> int:
    """Serialize Decimal to integer (rounded)."""
    return int(v.quantize(Decimal('1')))


def deserialize_decimal_float(v: Union[int, float, str]) -> Decimal:
    """Deserialize Decimal from float/int/string."""
    try:
        return Decimal(str(v))
    except (ValueError, TypeError) as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid decimal value: {e}"
        )


# =============================================================================
# File Serializers
# =============================================================================

def serialize_file_path(v: Path) -> str:
    """Serialize Path to string."""
    return str(v)


def serialize_file_base64(v: Path) -> str:
    """Serialize file to base64 string."""
    try:
        if not v.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        
        # Check file size (10MB limit)
        file_size = v.stat().st_size
        if file_size > 10 * 1024 * 1024:
            raise PydanticCustomError(
                SerializationErrorCode.FILE_TOO_LARGE,
                f"File too large: {file_size} bytes"
            )
        
        with open(v, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return base64.b64encode(content).decode('utf-8')
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize file: {e}"
        )


def serialize_file_hash(v: Path) -> str:
    """Serialize file to SHA256 hash."""
    try:
        if not v.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        
        with open(v, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return hashlib.sha256(content).hexdigest()
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to hash file: {e}"
        )


def deserialize_file_path(v: str) -> Path:
    """Deserialize string to Path."""
    try:
        path = Path(v)
        if not path.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        return path
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid file path: {e}"
        )


# =============================================================================
# URL Serializers
# =============================================================================

def serialize_url_normalized(v: str) -> str:
    """Serialize URL with normalization."""
    try:
        parsed = urlparse(v)
        if not parsed.scheme:
            v = f"https://{v}"
            parsed = urlparse(v)
        
        # Normalize URL
        normalized = f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"
        
        return normalized
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_URL,
            f"Invalid URL format: {e}"
        )


def serialize_url_components(v: str) -> Dict[str, str]:
    """Serialize URL to components."""
    try:
        parsed = urlparse(v)
        return {
            'scheme': parsed.scheme,
            'netloc': parsed.netloc,
            'path': parsed.path,
            'query': parsed.query,
            'fragment': parsed.fragment
        }
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_URL,
            f"Invalid URL format: {e}"
        )


def deserialize_url_components(v: Dict[str, str]) -> str:
    """Deserialize URL from components."""
    try:
        return urlunparse((
            v.get('scheme', 'https'),
            v.get('netloc', ''),
            v.get('path', ''),
            '',
            v.get('query', ''),
            v.get('fragment', '')
        ))
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid URL components: {e}"
        )


# =============================================================================
# Binary Data Serializers
# =============================================================================

def serialize_binary_base64(v: bytes) -> str:
    """Serialize binary data to base64 string."""
    try:
        if len(v) > 10 * 1024 * 1024:  # 10MB limit
            raise PydanticCustomError(
                SerializationErrorCode.BINARY_TOO_LARGE,
                f"Binary data too large: {len(v)} bytes"
            )
        return base64.b64encode(v).decode('utf-8')
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize binary data: {e}"
        )


def serialize_binary_hex(v: bytes) -> str:
    """Serialize binary data to hex string."""
    try:
        return v.hex()
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize binary data: {e}"
        )


def deserialize_binary_base64(v: str) -> bytes:
    """Deserialize binary data from base64 string."""
    try:
        return base64.b64decode(v)
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid base64 data: {e}"
        )


def deserialize_binary_hex(v: str) -> bytes:
    """Deserialize binary data from hex string."""
    try:
        return bytes.fromhex(v)
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid hex data: {e}"
        )


# =============================================================================
# JSON Serializers
# =============================================================================

def serialize_json_pretty(v: Any) -> str:
    """Serialize to pretty-printed JSON."""
    try:
        return json.dumps(v, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize to JSON: {e}"
        )


def serialize_json_compact(v: Any) -> str:
    """Serialize to compact JSON."""
    try:
        return json.dumps(v, separators=(',', ':'), ensure_ascii=False, default=str)
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize to JSON: {e}"
        )


def deserialize_json(v: str) -> Any:
    """Deserialize from JSON string."""
    try:
        return json.loads(v)
    except json.JSONDecodeError as e:
        raise PydanticCustomError(
            SerializationErrorCode.DESERIALIZATION_FAILED,
            f"Invalid JSON format: {e}"
        )


# =============================================================================
# Custom Object Serializers
# =============================================================================

def serialize_dataclass(v: Any) -> Dict[str, Any]:
    """Serialize dataclass to dictionary."""
    if is_dataclass(v):
        return asdict(v)
    else:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_DATA_TYPE,
            "Object is not a dataclass"
        )


def serialize_enum_name(v: Any) -> str:
    """Serialize enum to name string."""
    if hasattr(v, 'name'):
        return v.name
    else:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_DATA_TYPE,
            "Object is not an enum"
        )


def serialize_enum_value(v: Any) -> Any:
    """Serialize enum to value."""
    if hasattr(v, 'value'):
        return v.value
    else:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_DATA_TYPE,
            "Object is not an enum"
        )


# =============================================================================
# Performance Optimized Serializers
# =============================================================================

class SerializationCache:
    """Cache for serialization results to improve performance."""
    
    def __init__(self, max_size: int = 1000):
        
    """__init__ function."""
self.cache: Dict[str, Any] = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached serialization result."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached serialization result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# Global serialization cache
serialization_cache = SerializationCache()


def serialize_with_cache(serializer_func: Callable[[Any], Any], value: Any) -> Any:
    """Serialize with caching for performance."""
    # Create cache key
    cache_key = f"{serializer_func.__name__}:{hash(str(value))}"
    
    # Check cache
    cached_result = serialization_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Perform serialization
    result = serializer_func(value)
    
    # Cache result
    serialization_cache.set(cache_key, result)
    
    return result


# =============================================================================
# Async Serializers
# =============================================================================

async def serialize_file_async(v: Path) -> str:
    """Async file serialization."""
    try:
        if not v.exists():
            raise PydanticCustomError(
                SerializationErrorCode.FILE_NOT_FOUND,
                f"File not found: {v}"
            )
        
        # Simulate async file reading
        await asyncio.sleep(0.01)
        
        with open(v, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return base64.b64encode(content).decode('utf-8')
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.SERIALIZATION_FAILED,
            f"Failed to serialize file: {e}"
        )


async def serialize_url_async(v: str) -> str:
    """Async URL serialization with validation."""
    try:
        # Simulate async URL validation
        await asyncio.sleep(0.01)
        
        return serialize_url_normalized(v)
    
    except Exception as e:
        raise PydanticCustomError(
            SerializationErrorCode.INVALID_URL,
            f"Invalid URL: {e}"
        )


# =============================================================================
# Batch Serializers
# =============================================================================

def serialize_batch_datetime(datetimes: List[datetime]) -> List[str]:
    """Serialize a batch of datetimes."""
    return [serialize_datetime_iso(dt) for dt in datetimes]


def serialize_batch_decimal(decimals: List[Decimal]) -> List[float]:
    """Serialize a batch of decimals."""
    return [serialize_decimal_float(d) for d in decimals]


def serialize_batch_paths(paths: List[Path]) -> List[str]:
    """Serialize a batch of paths."""
    return [serialize_file_path(p) for p in paths]


def serialize_batch_binary(binaries: List[bytes]) -> List[str]:
    """Serialize a batch of binary data."""
    return [serialize_binary_base64(b) for b in binaries]


# =============================================================================
# Custom Pydantic Serializers
# =============================================================================

def create_custom_serializer(
    serializer_func: Callable[[Any], Any],
    error_code: str,
    error_message: str
) -> Callable[[Any], Any]:
    """Create a custom serializer with error handling."""
    def wrapper(v: Any) -> Any:
        try:
            return serializer_func(v)
        except Exception as e:
            raise PydanticCustomError(error_code, error_message)
    
    return wrapper


# =============================================================================
# Serialization Utilities
# =============================================================================

def get_serialization_info(obj: Any) -> Dict[str, Any]:
    """Get serialization information for an object."""
    info = {
        'type': type(obj).__name__,
        'size': len(str(obj)) if hasattr(obj, '__len__') else None,
        'serializable': True,
        'methods': []
    }
    
    # Check for serialization methods
    if hasattr(obj, 'model_dump'):
        info['methods'].append('model_dump')
    if hasattr(obj, 'dict'):
        info['methods'].append('dict')
    if hasattr(obj, 'json'):
        info['methods'].append('json')
    if is_dataclass(obj):
        info['methods'].append('asdict')
    
    return info


def serialize_object_smart(obj: Any) -> Any:
    """Smart serialization that chooses the best method."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, 'json'):
        return obj.json()
    elif isinstance(obj, (datetime, date)):
        return serialize_datetime_iso(obj)
    elif isinstance(obj, Decimal):
        return serialize_decimal_float(obj)
    elif isinstance(obj, Path):
        return serialize_file_path(obj)
    elif isinstance(obj, bytes):
        return serialize_binary_base64(obj)
    else:
        return obj


def deserialize_object_smart(data: Any, target_type: type) -> Any:
    """Smart deserialization that chooses the best method."""
    if target_type == datetime:
        if isinstance(data, (int, float)):
            return deserialize_datetime_unix(data)
        elif isinstance(data, str):
            return deserialize_datetime_iso(data)
    elif target_type == Decimal:
        return deserialize_decimal_float(data)
    elif target_type == Path:
        return deserialize_file_path(data)
    elif target_type == bytes:
        if isinstance(data, str):
            return deserialize_binary_base64(data)
    
    return data


# =============================================================================
# Serialization Decorators
# =============================================================================

def serialize_output(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to automatically serialize function output."""
    def wrapper(*args, **kwargs) -> Any:
        result = func(*args, **kwargs)
        return serialize_object_smart(result)
    
    return wrapper


def deserialize_input(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to automatically deserialize function input."""
    def wrapper(*args, **kwargs) -> Any:
        # This would need type hints to work properly
        # For now, just pass through
        return func(*args, **kwargs)
    
    return wrapper


# =============================================================================
# Serialization Performance Monitoring
# =============================================================================

class SerializationMetrics:
    """Metrics for monitoring serialization performance."""
    
    def __init__(self) -> Any:
        self.total_serializations = 0
        self.total_deserializations = 0
        self.serialization_times: List[float] = []
        self.deserialization_times: List[float] = []
        self.errors: List[str] = []
    
    def record_serialization(self, duration: float, success: bool, error: Optional[str] = None):
        """Record serialization metrics."""
        self.total_serializations += 1
        if success:
            self.serialization_times.append(duration)
        if error:
            self.errors.append(error)
    
    def record_deserialization(self, duration: float, success: bool, error: Optional[str] = None):
        """Record deserialization metrics."""
        self.total_deserializations += 1
        if success:
            self.deserialization_times.append(duration)
        if error:
            self.errors.append(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics."""
        return {
            'total_serializations': self.total_serializations,
            'total_deserializations': self.total_deserializations,
            'avg_serialization_time': sum(self.serialization_times) / len(self.serialization_times) if self.serialization_times else 0,
            'avg_deserialization_time': sum(self.deserialization_times) / len(self.deserialization_times) if self.deserialization_times else 0,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-10:] if self.errors else []
        }


# Global metrics
serialization_metrics = SerializationMetrics()


def monitor_serialization(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to monitor serialization performance."""
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            serialization_metrics.record_serialization(duration, True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            serialization_metrics.record_serialization(duration, False, str(e))
            raise
    
    return wrapper


# =============================================================================
# Export all serializers
# =============================================================================

__all__ = [
    # Error codes
    "SerializationErrorCode",
    
    # DateTime serializers
    "serialize_datetime_iso", "serialize_datetime_unix", "serialize_datetime_readable",
    "deserialize_datetime_iso", "deserialize_datetime_unix",
    
    # Decimal serializers
    "serialize_decimal_float", "serialize_decimal_string", "serialize_decimal_int",
    "deserialize_decimal_float",
    
    # File serializers
    "serialize_file_path", "serialize_file_base64", "serialize_file_hash",
    "deserialize_file_path",
    
    # URL serializers
    "serialize_url_normalized", "serialize_url_components", "deserialize_url_components",
    
    # Binary serializers
    "serialize_binary_base64", "serialize_binary_hex",
    "deserialize_binary_base64", "deserialize_binary_hex",
    
    # JSON serializers
    "serialize_json_pretty", "serialize_json_compact", "deserialize_json",
    
    # Custom object serializers
    "serialize_dataclass", "serialize_enum_name", "serialize_enum_value",
    
    # Performance
    "SerializationCache", "serialize_with_cache",
    
    # Async serializers
    "serialize_file_async", "serialize_url_async",
    
    # Batch serializers
    "serialize_batch_datetime", "serialize_batch_decimal", "serialize_batch_paths", "serialize_batch_binary",
    
    # Utilities
    "create_custom_serializer", "get_serialization_info", "serialize_object_smart", "deserialize_object_smart",
    
    # Decorators
    "serialize_output", "deserialize_input", "monitor_serialization",
    
    # Metrics
    "SerializationMetrics", "serialization_metrics",
] 