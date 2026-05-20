"""
Serializer System
=================

Advanced serialization system with multiple formats.
"""

import json
import pickle
import base64
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Serialization format."""
    JSON = "json"
    PICKLE = "pickle"
    BASE64 = "base64"
    YAML = "yaml"
    XML = "xml"


@dataclass
class SerializationResult:
    """Serialization result."""
    format: SerializationFormat
    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class Serializer:
    """Advanced serializer."""
    
    def __init__(self):
        """Initialize serializer."""
        self.serializers: Dict[SerializationFormat, Callable] = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.BASE64: self._serialize_base64,
        }
        self.deserializers: Dict[SerializationFormat, Callable] = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.BASE64: self._deserialize_base64,
        }
    
    def _serialize_json(self, data: Any) -> bytes:
        """Serialize to JSON."""
        return json.dumps(data, default=str).encode('utf-8')
    
    def _deserialize_json(self, data: bytes) -> Any:
        """Deserialize from JSON."""
        return json.loads(data.decode('utf-8'))
    
    def _serialize_pickle(self, data: Any) -> bytes:
        """Serialize to Pickle."""
        return pickle.dumps(data)
    
    def _deserialize_pickle(self, data: bytes) -> Any:
        """Deserialize from Pickle."""
        return pickle.loads(data)
    
    def _serialize_base64(self, data: Any) -> bytes:
        """Serialize to Base64."""
        if isinstance(data, str):
            return base64.b64encode(data.encode('utf-8'))
        elif isinstance(data, bytes):
            return base64.b64encode(data)
        else:
            # Convert to JSON first
            json_data = json.dumps(data, default=str)
            return base64.b64encode(json_data.encode('utf-8'))
    
    def _deserialize_base64(self, data: bytes) -> Any:
        """Deserialize from Base64."""
        decoded = base64.b64decode(data)
        try:
            return json.loads(decoded.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return decoded
    
    def serialize(
        self,
        data: Any,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> SerializationResult:
        """
        Serialize data.
        
        Args:
            data: Data to serialize
            format: Serialization format
            
        Returns:
            Serialization result
        """
        if format not in self.serializers:
            raise ValueError(f"Unsupported format: {format}")
        
        serializer = self.serializers[format]
        serialized_data = serializer(data)
        
        return SerializationResult(
            format=format,
            data=serialized_data,
            metadata={
                "size": len(serialized_data),
                "original_type": type(data).__name__
            }
        )
    
    def deserialize(
        self,
        data: bytes,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> Any:
        """
        Deserialize data.
        
        Args:
            data: Serialized data
            format: Serialization format
            
        Returns:
            Deserialized data
        """
        if format not in self.deserializers:
            raise ValueError(f"Unsupported format: {format}")
        
        deserializer = self.deserializers[format]
        return deserializer(data)
    
    def serialize_to_file(
        self,
        data: Any,
        file_path: Path,
        format: SerializationFormat = SerializationFormat.JSON
    ):
        """
        Serialize data to file.
        
        Args:
            data: Data to serialize
            file_path: Output file path
            format: Serialization format
        """
        result = self.serialize(data, format)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(result.data)
        
        logger.info(f"Serialized data to {file_path}")
    
    def deserialize_from_file(
        self,
        file_path: Path,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> Any:
        """
        Deserialize data from file.
        
        Args:
            file_path: Input file path
            format: Serialization format
            
        Returns:
            Deserialized data
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        
        return self.deserialize(data, format)




