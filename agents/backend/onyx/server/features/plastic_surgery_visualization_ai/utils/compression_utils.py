"""Compression utilities."""

import gzip
import zlib
from typing import Union
from io import BytesIO


def compress_gzip(data: bytes) -> bytes:
    """
    Compress data using gzip.
    
    Args:
        data: Data to compress
        
    Returns:
        Compressed bytes
    """
    return gzip.compress(data)


def decompress_gzip(compressed: bytes) -> bytes:
    """
    Decompress gzip data.
    
    Args:
        compressed: Compressed bytes
        
    Returns:
        Decompressed bytes
    """
    return gzip.decompress(compressed)


def compress_zlib(data: bytes) -> bytes:
    """
    Compress data using zlib.
    
    Args:
        data: Data to compress
        
    Returns:
        Compressed bytes
    """
    return zlib.compress(data)


def decompress_zlib(compressed: bytes) -> bytes:
    """
    Decompress zlib data.
    
    Args:
        compressed: Compressed bytes
        
    Returns:
        Decompressed bytes
    """
    return zlib.decompress(compressed)


def compress_string(text: str, encoding: str = 'utf-8') -> bytes:
    """
    Compress string to bytes.
    
    Args:
        text: Text to compress
        encoding: Text encoding
        
    Returns:
        Compressed bytes
    """
    return compress_gzip(text.encode(encoding))


def decompress_string(compressed: bytes, encoding: str = 'utf-8') -> str:
    """
    Decompress bytes to string.
    
    Args:
        compressed: Compressed bytes
        encoding: Text encoding
        
    Returns:
        Decompressed string
    """
    return decompress_gzip(compressed).decode(encoding)


def get_compression_ratio(original: bytes, compressed: bytes) -> float:
    """
    Calculate compression ratio.
    
    Args:
        original: Original data size
        compressed: Compressed data size
        
    Returns:
        Compression ratio (0.0 to 1.0)
    """
    if len(original) == 0:
        return 0.0
    return len(compressed) / len(original)

