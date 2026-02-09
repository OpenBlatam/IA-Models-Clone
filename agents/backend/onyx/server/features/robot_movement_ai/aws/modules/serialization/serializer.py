"""
Serializer
==========

Multi-format serializer.
"""

import logging
import json
import pickle
from typing import Any, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


class Serializer:
    """Multi-format serializer."""
    
    def __init__(self, default_format: SerializationFormat = SerializationFormat.JSON):
        self.default_format = default_format
    
    def serialize(self, data: Any, format: Optional[SerializationFormat] = None) -> bytes:
        """Serialize data."""
        format = format or self.default_format
        
        if format == SerializationFormat.JSON:
            return json.dumps(data).encode("utf-8")
        
        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        
        elif format == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.packb(data)
            except ImportError:
                logger.warning("msgpack not installed, falling back to JSON")
                return json.dumps(data).encode("utf-8")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def deserialize(self, data: bytes, format: Optional[SerializationFormat] = None) -> Any:
        """Deserialize data."""
        format = format or self.default_format
        
        if format == SerializationFormat.JSON:
            return json.loads(data.decode("utf-8"))
        
        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        
        elif format == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.unpackb(data, raw=False)
            except ImportError:
                logger.warning("msgpack not installed, falling back to JSON")
                return json.loads(data.decode("utf-8"))
        
        else:
            raise ValueError(f"Unsupported format: {format}")

