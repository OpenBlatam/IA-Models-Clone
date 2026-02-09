"""Serialization utilities."""

import json
from typing import Any, Dict, Optional
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import pickle
import base64

from utils.logger import get_logger

logger = get_logger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for special types."""
    
    def default(self, obj: Any) -> Any:
        """Handle special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """
    Convert object to JSON string.
    
    Args:
        obj: Object to serialize
        indent: Optional indentation
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=JSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Parse JSON string to object.
    
    Args:
        json_str: JSON string
        
    Returns:
        Parsed object
    """
    return json.loads(json_str)


def to_json_file(obj: Any, file_path: Path, indent: int = 2) -> None:
    """
    Save object to JSON file.
    
    Args:
        obj: Object to save
        file_path: Path to file
        indent: Indentation
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, cls=JSONEncoder, indent=indent)


def from_json_file(file_path: Path) -> Any:
    """
    Load object from JSON file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Loaded object
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def serialize_pickle(obj: Any) -> bytes:
    """
    Serialize object using pickle.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Pickled bytes
    """
    return pickle.dumps(obj)


def deserialize_pickle(data: bytes) -> Any:
    """
    Deserialize object from pickle.
    
    Args:
        data: Pickled bytes
        
    Returns:
        Deserialized object
    """
    return pickle.loads(data)


def encode_base64(data: bytes) -> str:
    """
    Encode bytes to base64 string.
    
    Args:
        data: Bytes to encode
        
    Returns:
        Base64 string
    """
    return base64.b64encode(data).decode('utf-8')


def decode_base64(encoded: str) -> bytes:
    """
    Decode base64 string to bytes.
    
    Args:
        encoded: Base64 string
        
    Returns:
        Decoded bytes
    """
    return base64.b64decode(encoded)

