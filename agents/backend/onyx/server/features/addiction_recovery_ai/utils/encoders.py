"""
Encoding utilities
Data encoding and decoding functions
"""

from typing import Any
import base64
import json
from utils.serialization import serialize_for_json


def encode_base64(value: str) -> str:
    """
    Encode string to base64
    
    Args:
        value: String to encode
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(value.encode()).decode()


def decode_base64(encoded: str) -> str:
    """
    Decode base64 string
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Decoded string
    """
    return base64.b64decode(encoded).decode()


def encode_json_base64(data: Any) -> str:
    """
    Encode data to JSON and then base64
    
    Args:
        data: Data to encode
    
    Returns:
        Base64 encoded JSON string
    """
    json_str = json.dumps(serialize_for_json(data))
    return encode_base64(json_str)


def decode_json_base64(encoded: str) -> Any:
    """
    Decode base64 JSON string
    
    Args:
        encoded: Base64 encoded JSON string
    
    Returns:
        Decoded data
    """
    json_str = decode_base64(encoded)
    return json.loads(json_str)


def url_encode(value: str) -> str:
    """
    URL encode string
    
    Args:
        value: String to encode
    
    Returns:
        URL encoded string
    """
    from urllib.parse import quote
    return quote(value, safe="")


def url_decode(encoded: str) -> str:
    """
    URL decode string
    
    Args:
        encoded: URL encoded string
    
    Returns:
        Decoded string
    """
    from urllib.parse import unquote
    return unquote(encoded)

