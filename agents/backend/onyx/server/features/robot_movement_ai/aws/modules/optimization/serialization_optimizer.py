"""
Serialization Optimizer
=======================

Ultra-fast serialization optimizations.
"""

import logging
from typing import Any, Optional
import orjson
import msgpack
from aws.modules.serialization import Serializer, SerializationFormat

logger = logging.getLogger(__name__)


class SerializationOptimizer:
    """Optimized serialization for maximum speed."""
    
    def __init__(self):
        self._orjson_available = True
        self._msgpack_available = True
    
    def serialize_fast(self, data: Any) -> bytes:
        """Ultra-fast serialization using orjson."""
        try:
            return orjson.dumps(data)
        except Exception:
            # Fallback to standard JSON
            import json
            return json.dumps(data).encode("utf-8")
    
    def deserialize_fast(self, data: bytes) -> Any:
        """Ultra-fast deserialization using orjson."""
        try:
            return orjson.loads(data)
        except Exception:
            # Fallback to standard JSON
            import json
            return json.loads(data.decode("utf-8"))
    
    def serialize_binary(self, data: Any) -> bytes:
        """Binary serialization using msgpack."""
        try:
            return msgpack.packb(data)
        except Exception:
            # Fallback to orjson
            return self.serialize_fast(data)
    
    def deserialize_binary(self, data: bytes) -> Any:
        """Binary deserialization using msgpack."""
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception:
            # Fallback to orjson
            return self.deserialize_fast(data)
    
    def optimize_for_size(self, data: Any) -> bytes:
        """Optimize serialization for minimum size."""
        # Try msgpack first (usually smaller)
        try:
            msgpack_data = msgpack.packb(data)
            orjson_data = orjson.dumps(data)
            
            # Use smaller one
            if len(msgpack_data) < len(orjson_data):
                return msgpack_data
            return orjson_data
        except Exception:
            return self.serialize_fast(data)
    
    def optimize_for_speed(self, data: Any) -> bytes:
        """Optimize serialization for maximum speed."""
        # orjson is usually fastest
        return self.serialize_fast(data)















