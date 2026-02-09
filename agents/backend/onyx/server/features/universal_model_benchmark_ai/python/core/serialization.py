"""
Serialization Module - Data serialization utilities.

Provides:
- Multiple serialization formats
- Schema validation
- Versioning support
- Compression
"""

import logging
import json
import pickle
import gzip
import base64
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import yaml

logger = logging.getLogger(__name__)


class SerializationFormat(str, Enum):
    """Serialization formats."""
    JSON = "json"
    PICKLE = "pickle"
    YAML = "yaml"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


@dataclass
class SerializationOptions:
    """Serialization options."""
    format: SerializationFormat = SerializationFormat.JSON
    compress: bool = False
    indent: Optional[int] = None
    ensure_ascii: bool = False
    version: Optional[str] = None


class Serializer:
    """Data serializer."""
    
    def __init__(self, options: Optional[SerializationOptions] = None):
        """
        Initialize serializer.
        
        Args:
            options: Serialization options
        """
        self.options = options or SerializationOptions()
    
    def serialize(self, data: Any, options: Optional[SerializationOptions] = None) -> bytes:
        """
        Serialize data.
        
        Args:
            data: Data to serialize
            options: Optional serialization options
            
        Returns:
            Serialized data as bytes
        """
        opts = options or self.options
        format_type = opts.format
        
        # Serialize based on format
        if format_type == SerializationFormat.JSON:
            serialized = json.dumps(
                data,
                indent=opts.indent,
                ensure_ascii=opts.ensure_ascii,
                default=str,
            ).encode('utf-8')
        
        elif format_type == SerializationFormat.PICKLE:
            serialized = pickle.dumps(data)
        
        elif format_type == SerializationFormat.YAML:
            serialized = yaml.dump(data, default_flow_style=False).encode('utf-8')
        
        elif format_type == SerializationFormat.MSGPACK:
            try:
                import msgpack
                serialized = msgpack.packb(data)
            except ImportError:
                logger.warning("msgpack not available, falling back to JSON")
                serialized = json.dumps(data, default=str).encode('utf-8')
        
        else:
            # Default to JSON
            serialized = json.dumps(data, default=str).encode('utf-8')
        
        # Compress if requested
        if opts.compress:
            serialized = gzip.compress(serialized)
        
        return serialized
    
    def deserialize(self, data: bytes, options: Optional[SerializationOptions] = None) -> Any:
        """
        Deserialize data.
        
        Args:
            data: Serialized data
            options: Optional serialization options
            
        Returns:
            Deserialized data
        """
        opts = options or self.options
        format_type = opts.format
        
        # Decompress if needed
        if opts.compress:
            try:
                data = gzip.decompress(data)
            except:
                # Not compressed, use as-is
                pass
        
        # Deserialize based on format
        if format_type == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        
        elif format_type == SerializationFormat.PICKLE:
            return pickle.loads(data)
        
        elif format_type == SerializationFormat.YAML:
            return yaml.safe_load(data.decode('utf-8'))
        
        elif format_type == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.unpackb(data, raw=False)
            except ImportError:
                logger.warning("msgpack not available, falling back to JSON")
                return json.loads(data.decode('utf-8'))
        
        else:
            # Default to JSON
            return json.loads(data.decode('utf-8'))
    
    def serialize_to_string(self, data: Any, options: Optional[SerializationOptions] = None) -> str:
        """
        Serialize to string (base64 encoded).
        
        Args:
            data: Data to serialize
            options: Optional serialization options
            
        Returns:
            Base64 encoded string
        """
        serialized = self.serialize(data, options)
        return base64.b64encode(serialized).decode('utf-8')
    
    def deserialize_from_string(self, data: str, options: Optional[SerializationOptions] = None) -> Any:
        """
        Deserialize from string (base64 encoded).
        
        Args:
            data: Base64 encoded string
            options: Optional serialization options
            
        Returns:
            Deserialized data
        """
        serialized = base64.b64decode(data.encode('utf-8'))
        return self.deserialize(serialized, options)












