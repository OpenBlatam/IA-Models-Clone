"""Encoding and decoding utilities."""

import base64
import hashlib
from typing import Union
import binascii


def encode_base64_urlsafe(data: bytes) -> str:
    """
    Encode bytes to URL-safe base64.
    
    Args:
        data: Bytes to encode
        
    Returns:
        URL-safe base64 string
    """
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')


def decode_base64_urlsafe(encoded: str) -> bytes:
    """
    Decode URL-safe base64 string.
    
    Args:
        encoded: URL-safe base64 string
        
    Returns:
        Decoded bytes
    """
    # Add padding if needed
    padding = 4 - len(encoded) % 4
    if padding != 4:
        encoded += '=' * padding
    
    return base64.urlsafe_b64decode(encoded)


def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash string using specified algorithm.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


def hash_file(file_path: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Hash file using specified algorithm.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm
        chunk_size: Chunk size for reading
        
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def hex_encode(data: bytes) -> str:
    """
    Encode bytes to hexadecimal string.
    
    Args:
        data: Bytes to encode
        
    Returns:
        Hexadecimal string
    """
    return binascii.hexlify(data).decode('utf-8')


def hex_decode(hex_string: str) -> bytes:
    """
    Decode hexadecimal string to bytes.
    
    Args:
        hex_string: Hexadecimal string
        
    Returns:
        Decoded bytes
    """
    return binascii.unhexlify(hex_string)


def url_encode(text: str) -> str:
    """
    URL encode string.
    
    Args:
        text: Text to encode
        
    Returns:
        URL-encoded string
    """
    from urllib.parse import quote
    return quote(text, safe='')


def url_decode(encoded: str) -> str:
    """
    URL decode string.
    
    Args:
        encoded: URL-encoded string
        
    Returns:
        Decoded string
    """
    from urllib.parse import unquote
    return unquote(encoded)

