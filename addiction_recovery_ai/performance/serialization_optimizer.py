"""
Serialization Optimizer
Ultra-fast JSON and data serialization
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Try to use fastest JSON library
try:
    import orjson
    JSON_ENCODER = orjson
    JSON_AVAILABLE = True
except ImportError:
    try:
        import ujson
        JSON_ENCODER = ujson
        JSON_AVAILABLE = True
    except ImportError:
        import json
        JSON_ENCODER = json
        JSON_AVAILABLE = True
        logger.warning("Using standard json library. Install orjson for better performance.")


class FastJSON:
    """Fast JSON serialization wrapper"""
    
    @staticmethod
    def dumps(obj: Any, **kwargs) -> bytes:
        """Serialize to JSON (returns bytes for orjson)"""
        if JSON_ENCODER == orjson:
            return orjson.dumps(obj, **kwargs)
        elif JSON_ENCODER == ujson:
            return ujson.dumps(obj).encode()
        else:
            return json.dumps(obj, **kwargs).encode()
    
    @staticmethod
    def loads(data: bytes | str) -> Any:
        """Deserialize from JSON"""
        if isinstance(data, str):
            data = data.encode()
        
        if JSON_ENCODER == orjson:
            return orjson.loads(data)
        elif JSON_ENCODER == ujson:
            return ujson.loads(data.decode())
        else:
            return json.loads(data.decode())


class SerializationOptimizer:
    """
    Serialization optimizer
    
    Features:
    - Fast JSON (orjson/ujson)
    - MessagePack for binary
    - Pre-compiled serializers
    - Batch serialization
    """
    
    def __init__(self):
        self.json = FastJSON()
        self._msgpack_available = False
        
        try:
            import msgpack
            self.msgpack = msgpack
            self._msgpack_available = True
        except ImportError:
            logger.warning("msgpack not available. Install for binary serialization.")
    
    def serialize_json(self, obj: Any) -> bytes:
        """Fast JSON serialization"""
        return self.json.dumps(obj)
    
    def deserialize_json(self, data: bytes | str) -> Any:
        """Fast JSON deserialization"""
        return self.json.loads(data)
    
    def serialize_msgpack(self, obj: Any) -> bytes:
        """MessagePack serialization (faster than JSON)"""
        if not self._msgpack_available:
            return self.serialize_json(obj)
        return self.msgpack.packb(obj)
    
    def deserialize_msgpack(self, data: bytes) -> Any:
        """MessagePack deserialization"""
        if not self._msgpack_available:
            return self.deserialize_json(data)
        return self.msgpack.unpackb(data, raw=False)
    
    def batch_serialize(self, items: List[Any], format: str = "json") -> List[bytes]:
        """Batch serialize items"""
        if format == "json":
            return [self.serialize_json(item) for item in items]
        elif format == "msgpack":
            return [self.serialize_msgpack(item) for item in items]
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global serializer instance
_serializer: SerializationOptimizer = None


def get_serializer() -> SerializationOptimizer:
    """Get global serialization optimizer"""
    global _serializer
    if _serializer is None:
        _serializer = SerializationOptimizer()
    return _serializer















