"""
Advanced serialization system for KV cache.

This module provides multiple serialization formats and optimizations
for cache data storage and transmission.
"""

import time
import threading
import json
import pickle
import base64
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False


class SerializationFormat(Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    BASE64 = "base64"
    CUSTOM = "custom"


@dataclass
class SerializationResult:
    """Result of serialization."""
    data: bytes
    format: SerializationFormat
    original_size: int
    serialized_size: int
    compression_ratio: float
    serialization_time: float


class AdvancedSerializer:
    """Advanced serializer for cache data."""
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.PICKLE):
        self.default_format = default_format
        self._custom_serializers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
    def serialize(
        self,
        data: Any,
        format: Optional[SerializationFormat] = None
    ) -> SerializationResult:
        """Serialize data."""
        start_time = time.time()
        original_size = len(str(data).encode('utf-8'))
        
        fmt = format or self.default_format
        
        if fmt == SerializationFormat.JSON:
            serialized = json.dumps(data).encode('utf-8')
        elif fmt == SerializationFormat.PICKLE:
            serialized = pickle.dumps(data)
        elif fmt == SerializationFormat.MSGPACK:
            if MSGPACK_AVAILABLE:
                serialized = msgpack.packb(data)
            else:
                # Fallback to JSON
                serialized = json.dumps(data).encode('utf-8')
        elif fmt == SerializationFormat.BASE64:
            serialized = base64.b64encode(pickle.dumps(data))
        else:
            # Use custom serializer if available
            if fmt.value in self._custom_serializers:
                serialized = self._custom_serializers[fmt.value](data)
            else:
                serialized = pickle.dumps(data)
                
        serialized_size = len(serialized)
        compression_ratio = serialized_size / original_size if original_size > 0 else 1.0
        serialization_time = time.time() - start_time
        
        return SerializationResult(
            data=serialized,
            format=fmt,
            original_size=original_size,
            serialized_size=serialized_size,
            compression_ratio=compression_ratio,
            serialization_time=serialization_time
        )
        
    def deserialize(
        self,
        data: bytes,
        format: SerializationFormat
    ) -> Any:
        """Deserialize data."""
        if format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif format == SerializationFormat.MSGPACK:
            if MSGPACK_AVAILABLE:
                return msgpack.unpackb(data, raw=False)
            else:
                return json.loads(data.decode('utf-8'))
        elif format == SerializationFormat.BASE64:
            return pickle.loads(base64.b64decode(data))
        else:
            # Custom deserializer
            if format.value in self._custom_serializers:
                return self._custom_serializers[format.value](data)
            else:
                return pickle.loads(data)
                
    def register_custom_serializer(
        self,
        format_name: str,
        serializer: Callable[[Any], bytes],
        deserializer: Callable[[bytes], Any]
    ) -> None:
        """Register custom serializer."""
        with self._lock:
            # Store both in same dict with prefix
            self._custom_serializers[f"{format_name}_serialize"] = serializer
            self._custom_serializers[f"{format_name}_deserialize"] = deserializer
            
    def get_best_format(self, data: Any) -> SerializationFormat:
        """Determine best format for data."""
        # Try different formats and pick smallest
        formats = [SerializationFormat.JSON, SerializationFormat.PICKLE]
        if MSGPACK_AVAILABLE:
            formats.append(SerializationFormat.MSGPACK)
            
        best_format = SerializationFormat.PICKLE
        best_size = float('inf')
        
        for fmt in formats:
            try:
                result = self.serialize(data, fmt)
                if result.serialized_size < best_size:
                    best_size = result.serialized_size
                    best_format = fmt
            except Exception:
                continue
                
        return best_format


class SerializedCache:
    """Cache wrapper with serialization."""
    
    def __init__(
        self,
        cache: Any,
        serializer: Optional[AdvancedSerializer] = None,
        auto_serialize: bool = True
    ):
        self.cache = cache
        self.serializer = serializer or AdvancedSerializer()
        self.auto_serialize = auto_serialize
        
    def get(self, key: str) -> Any:
        """Get and deserialize value."""
        value = self.cache.get(key)
        if value is None:
            return None
            
        if isinstance(value, bytes) and self.auto_serialize:
            # Try to detect format (simplified - would store metadata)
            try:
                return self.serializer.deserialize(value, SerializationFormat.PICKLE)
            except Exception:
                try:
                    return self.serializer.deserialize(value, SerializationFormat.JSON)
                except Exception:
                    return value
                    
        return value
        
    def put(self, key: str, value: Any, format: Optional[SerializationFormat] = None) -> bool:
        """Serialize and put value."""
        if self.auto_serialize:
            result = self.serializer.serialize(value, format)
            return self.cache.put(key, result.data)
        else:
            return self.cache.put(key, value)
            
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.cache.delete(key)


