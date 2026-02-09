"""
Advanced Serialization System
==============================

Advanced serialization system with multiple formats and strategies.
"""

import logging
import json
import pickle
import base64
from typing import Dict, Any, Optional, Type, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Serialization format."""
    JSON = "json"
    PICKLE = "pickle"
    BASE64 = "base64"
    CUSTOM = "custom"


class Serializer(ABC):
    """Base serializer interface."""
    
    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        pass


class JSONSerializer(Serializer):
    """JSON serializer."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize from JSON bytes."""
        return json.loads(data.decode('utf-8'))


class PickleSerializer(Serializer):
    """Pickle serializer."""
    
    def serialize(self, data: Any) -> bytes:
        """Serialize to pickle bytes."""
        return pickle.dumps(data)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize from pickle bytes."""
        return pickle.loads(data)


class Base64Serializer(Serializer):
    """Base64 serializer (wraps another serializer)."""
    
    def __init__(self, inner_serializer: Serializer):
        """
        Initialize base64 serializer.
        
        Args:
            inner_serializer: Inner serializer to wrap
        """
        self.inner_serializer = inner_serializer
    
    def serialize(self, data: Any) -> bytes:
        """Serialize to base64 bytes."""
        serialized = self.inner_serializer.serialize(data)
        encoded = base64.b64encode(serialized)
        return encoded
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize from base64 bytes."""
        decoded = base64.b64decode(data)
        return self.inner_serializer.deserialize(decoded)


class SerializationManager:
    """Manager for serialization with multiple formats."""
    
    def __init__(self):
        """Initialize serialization manager."""
        self.serializers: Dict[SerializationFormat, Serializer] = {
            SerializationFormat.JSON: JSONSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
        }
        self.default_format = SerializationFormat.JSON
    
    def register_serializer(
        self,
        format_type: SerializationFormat,
        serializer: Serializer
    ):
        """
        Register a custom serializer.
        
        Args:
            format_type: Serialization format
            serializer: Serializer instance
        """
        self.serializers[format_type] = serializer
        logger.info(f"Registered serializer for format: {format_type.value}")
    
    def serialize(
        self,
        data: Any,
        format_type: Optional[SerializationFormat] = None
    ) -> bytes:
        """
        Serialize data.
        
        Args:
            data: Data to serialize
            format_type: Optional format type (uses default if not provided)
            
        Returns:
            Serialized bytes
        """
        format_type = format_type or self.default_format
        
        if format_type not in self.serializers:
            raise ValueError(f"Unsupported serialization format: {format_type}")
        
        serializer = self.serializers[format_type]
        return serializer.serialize(data)
    
    def deserialize(
        self,
        data: bytes,
        format_type: Optional[SerializationFormat] = None
    ) -> Any:
        """
        Deserialize data.
        
        Args:
            data: Serialized bytes
            format_type: Optional format type (uses default if not provided)
            
        Returns:
            Deserialized data
        """
        format_type = format_type or self.default_format
        
        if format_type not in self.serializers:
            raise ValueError(f"Unsupported serialization format: {format_type}")
        
        serializer = self.serializers[format_type]
        return serializer.deserialize(data)
    
    def serialize_to_string(
        self,
        data: Any,
        format_type: Optional[SerializationFormat] = None
    ) -> str:
        """
        Serialize data to string.
        
        Args:
            data: Data to serialize
            format_type: Optional format type
            
        Returns:
            Serialized string
        """
        if format_type == SerializationFormat.JSON or format_type is None:
            return json.dumps(data, ensure_ascii=False, default=str)
        elif format_type == SerializationFormat.BASE64:
            serialized = self.serialize(data, SerializationFormat.JSON)
            return base64.b64encode(serialized).decode('utf-8')
        else:
            # For other formats, encode bytes to base64 string
            serialized = self.serialize(data, format_type)
            return base64.b64encode(serialized).decode('utf-8')
    
    def deserialize_from_string(
        self,
        data: str,
        format_type: Optional[SerializationFormat] = None
    ) -> Any:
        """
        Deserialize data from string.
        
        Args:
            data: Serialized string
            format_type: Optional format type
            
        Returns:
            Deserialized data
        """
        if format_type == SerializationFormat.JSON or format_type is None:
            return json.loads(data)
        elif format_type == SerializationFormat.BASE64:
            decoded = base64.b64decode(data.encode('utf-8'))
            return self.deserialize(decoded, SerializationFormat.JSON)
        else:
            # For other formats, decode from base64
            decoded = base64.b64decode(data.encode('utf-8'))
            return self.deserialize(decoded, format_type)




