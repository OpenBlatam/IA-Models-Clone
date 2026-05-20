"""
Compression utilities
Data compression functions
"""

from typing import Optional
import gzip
import zlib
import base64
import json


def compress_gzip(data: str) -> bytes:
    """
    Compress data using gzip
    
    Args:
        data: Data to compress
    
    Returns:
        Compressed data
    """
    return gzip.compress(data.encode('utf-8'))


def decompress_gzip(compressed: bytes) -> str:
    """
    Decompress gzip data
    
    Args:
        compressed: Compressed data
    
    Returns:
        Decompressed data
    """
    return gzip.decompress(compressed).decode('utf-8')


def compress_zlib(data: str) -> bytes:
    """
    Compress data using zlib
    
    Args:
        data: Data to compress
    
    Returns:
        Compressed data
    """
    return zlib.compress(data.encode('utf-8'))


def decompress_zlib(compressed: bytes) -> str:
    """
    Decompress zlib data
    
    Args:
        compressed: Compressed data
    
    Returns:
        Decompressed data
    """
    return zlib.decompress(compressed).decode('utf-8')


def compress_json(data: Any, method: str = "gzip") -> str:
    """
    Compress JSON data
    
    Args:
        data: Data to compress
        method: Compression method (gzip or zlib)
    
    Returns:
        Base64 encoded compressed data
    """
    json_str = json.dumps(data)
    
    if method == "gzip":
        compressed = compress_gzip(json_str)
    else:
        compressed = compress_zlib(json_str)
    
    return base64.b64encode(compressed).decode('utf-8')


def decompress_json(compressed: str, method: str = "gzip") -> Any:
    """
    Decompress JSON data
    
    Args:
        compressed: Base64 encoded compressed data
        method: Compression method (gzip or zlib)
    
    Returns:
        Decompressed data
    """
    compressed_bytes = base64.b64decode(compressed.encode('utf-8'))
    
    if method == "gzip":
        decompressed = decompress_gzip(compressed_bytes)
    else:
        decompressed = decompress_zlib(compressed_bytes)
    
    return json.loads(decompressed)


def compress_string(data: str, method: str = "gzip") -> str:
    """
    Compress string
    
    Args:
        data: String to compress
        method: Compression method (gzip or zlib)
    
    Returns:
        Base64 encoded compressed data
    """
    if method == "gzip":
        compressed = compress_gzip(data)
    else:
        compressed = compress_zlib(data)
    
    return base64.b64encode(compressed).decode('utf-8')


def decompress_string(compressed: str, method: str = "gzip") -> str:
    """
    Decompress string
    
    Args:
        compressed: Base64 encoded compressed data
        method: Compression method (gzip or zlib)
    
    Returns:
        Decompressed string
    """
    compressed_bytes = base64.b64decode(compressed.encode('utf-8'))
    
    if method == "gzip":
        return decompress_gzip(compressed_bytes)
    
    return decompress_zlib(compressed_bytes)


def get_compression_ratio(original: str, compressed: str) -> float:
    """
    Get compression ratio
    
    Args:
        original: Original data
        compressed: Compressed data (base64 encoded)
    
    Returns:
        Compression ratio
    """
    original_size = len(original.encode('utf-8'))
    compressed_size = len(base64.b64decode(compressed.encode('utf-8')))
    
    if original_size == 0:
        return 0.0
    
    return compressed_size / original_size

