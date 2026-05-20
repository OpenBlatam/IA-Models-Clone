"""
Advanced Serialization System
==============================

Advanced serialization system with multiple formats and compression.
"""

import asyncio
import logging
import json
import pickle
import base64
import gzip
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    BASE64 = "base64"
    COMPRESSED_JSON = "compressed_json"
    COMPRESSED_PICKLE = "compressed_pickle"


@dataclass
class SerializationMetadata:
    """Serialization metadata."""
    format: SerializationFormat
    size: int
    compressed: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedSerializer:
    """Advanced serializer with multiple formats."""
    
    def __init__(self):
        """Initialize advanced serializer."""
        self.serializers: Dict[SerializationFormat, Callable] = {
            SerializationFormat.JSON: self._serialize_json,
            SerializationFormat.PICKLE: self._serialize_pickle,
            SerializationFormat.BASE64: self._serialize_base64,
            SerializationFormat.COMPRESSED_JSON: self._serialize_compressed_json,
            SerializationFormat.COMPRESSED_PICKLE: self._serialize_compressed_pickle
        }
        self.deserializers: Dict[SerializationFormat, Callable] = {
            SerializationFormat.JSON: self._deserialize_json,
            SerializationFormat.PICKLE: self._deserialize_pickle,
            SerializationFormat.BASE64: self._deserialize_base64,
            SerializationFormat.COMPRESSED_JSON: self._deserialize_compressed_json,
            SerializationFormat.COMPRESSED_PICKLE: self._deserialize_compressed_pickle
        }
    
    def serialize(
        self,
        data: Any,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> bytes:
        """
        Serialize data.
        
        Args:
            data: Data to serialize
            format: Serialization format
            
        Returns:
            Serialized data as bytes
        """
        serializer = self.serializers.get(format)
        if not serializer:
            raise ValueError(f"Unsupported format: {format}")
        
        return serializer(data)
    
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
        deserializer = self.deserializers.get(format)
        if not deserializer:
            raise ValueError(f"Unsupported format: {format}")
        
        return deserializer(data)
    
    def _serialize_json(self, data: Any) -> bytes:
        """Serialize to JSON."""
        return json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
    
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
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = pickle.dumps(data)
        return base64.b64encode(data_bytes)
    
    def _deserialize_base64(self, data: bytes) -> Any:
        """Deserialize from Base64."""
        decoded = base64.b64decode(data)
        try:
            return pickle.loads(decoded)
        except:
            return decoded.decode('utf-8')
    
    def _serialize_compressed_json(self, data: Any) -> bytes:
        """Serialize to compressed JSON."""
        json_data = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        return gzip.compress(json_data)
    
    def _deserialize_compressed_json(self, data: bytes) -> Any:
        """Deserialize from compressed JSON."""
        decompressed = gzip.decompress(data)
        return json.loads(decompressed.decode('utf-8'))
    
    def _serialize_compressed_pickle(self, data: Any) -> bytes:
        """Serialize to compressed Pickle."""
        pickle_data = pickle.dumps(data)
        return gzip.compress(pickle_data)
    
    def _deserialize_compressed_pickle(self, data: bytes) -> Any:
        """Deserialize from compressed Pickle."""
        decompressed = gzip.decompress(data)
        return pickle.loads(decompressed)
    
    async def serialize_to_file(
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
        serialized = self.serialize(data, format)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(serialized)
        logger.info(f"Serialized data to {file_path}")
    
    async def deserialize_from_file(
        self,
        file_path: Path,
        format: Optional[SerializationFormat] = None
    ) -> Any:
        """
        Deserialize data from file.
        
        Args:
            file_path: Input file path
            format: Optional format (auto-detect if not provided)
            
        Returns:
            Deserialized data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect format
        if format is None:
            if file_path.suffix == '.json':
                format = SerializationFormat.JSON
            elif file_path.suffix == '.pkl':
                format = SerializationFormat.PICKLE
            elif file_path.suffix == '.gz':
                # Try compressed JSON first
                try:
                    data = file_path.read_bytes()
                    decompressed = gzip.decompress(data)
                    json.loads(decompressed.decode('utf-8'))
                    format = SerializationFormat.COMPRESSED_JSON
                except:
                    format = SerializationFormat.COMPRESSED_PICKLE
            else:
                format = SerializationFormat.JSON
        
        data = file_path.read_bytes()
        return self.deserialize(data, format)



