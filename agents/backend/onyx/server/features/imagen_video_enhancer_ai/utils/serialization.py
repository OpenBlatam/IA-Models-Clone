"""
Serialization Utilities
=======================

Utilities for serializing and deserializing data.
"""

import json
import pickle
from typing import Any, Optional, Dict
from pathlib import Path
import base64


class Serializer:
    """Base serializer interface."""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data to bytes."""
        raise NotImplementedError
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize bytes to data."""
        raise NotImplementedError


class JSONSerializer(Serializer):
    """JSON serializer."""
    
    @staticmethod
    def serialize(data: Any, indent: Optional[int] = None) -> bytes:
        """Serialize data to JSON bytes."""
        return json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
            default=str
        ).encode('utf-8')
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize JSON bytes to data."""
        return json.loads(data.decode('utf-8'))


class PickleSerializer(Serializer):
    """Pickle serializer."""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data to pickle bytes."""
        return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize pickle bytes to data."""
        return pickle.loads(data)


class Base64Serializer:
    """Base64 encoding utilities."""
    
    @staticmethod
    def encode(data: bytes) -> str:
        """Encode bytes to base64 string."""
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def decode(data: str) -> bytes:
        """Decode base64 string to bytes."""
        return base64.b64decode(data.encode('utf-8'))
    
    @staticmethod
    def encode_json(data: Any) -> str:
        """Encode JSON data to base64 string."""
        json_bytes = JSONSerializer.serialize(data)
        return Base64Serializer.encode(json_bytes)
    
    @staticmethod
    def decode_json(data: str) -> Any:
        """Decode base64 string to JSON data."""
        json_bytes = Base64Serializer.decode(data)
        return JSONSerializer.deserialize(json_bytes)


def save_serialized(
    data: Any,
    file_path: Path | str,
    format: str = "json",
    **kwargs
) -> Path:
    """
    Save data in serialized format.
    
    Args:
        data: Data to serialize
        file_path: Output file path
        format: Serialization format (json, pickle)
        **kwargs: Additional serializer options
        
    Returns:
        Path to saved file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        serializer = JSONSerializer()
        content = serializer.serialize(data, **kwargs)
    elif format == "pickle":
        serializer = PickleSerializer()
        content = serializer.serialize(data)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    path.write_bytes(content)
    return path


def load_serialized(
    file_path: Path | str,
    format: str = "json"
) -> Any:
    """
    Load data from serialized file.
    
    Args:
        file_path: Input file path
        format: Serialization format (json, pickle)
        
    Returns:
        Deserialized data
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = path.read_bytes()
    
    if format == "json":
        serializer = JSONSerializer()
        return serializer.deserialize(content)
    elif format == "pickle":
        serializer = PickleSerializer()
        return serializer.deserialize(content)
    else:
        raise ValueError(f"Unsupported format: {format}")




