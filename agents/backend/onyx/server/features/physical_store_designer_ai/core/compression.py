"""
Compression utilities for request/response optimization
"""

import gzip
import zlib
from typing import Optional, Union
from io import BytesIO

from .logging_config import get_logger

logger = get_logger(__name__)


def compress_gzip(data: bytes, compresslevel: int = 6) -> bytes:
    """Compress data using gzip"""
    try:
        buffer = BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=compresslevel) as f:
            f.write(data)
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Gzip compression failed: {e}")
        raise ValueError(f"Failed to compress data: {e}")


def decompress_gzip(data: bytes) -> bytes:
    """Decompress gzip data"""
    try:
        buffer = BytesIO(data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Gzip decompression failed: {e}")
        raise ValueError(f"Failed to decompress data: {e}")


def compress_deflate(data: bytes, level: int = 6) -> bytes:
    """Compress data using deflate (zlib)"""
    try:
        return zlib.compress(data, level=level)
    except Exception as e:
        logger.error(f"Deflate compression failed: {e}")
        raise ValueError(f"Failed to compress data: {e}")


def decompress_deflate(data: bytes) -> bytes:
    """Decompress deflate data"""
    try:
        return zlib.decompress(data)
    except Exception as e:
        logger.error(f"Deflate decompression failed: {e}")
        raise ValueError(f"Failed to decompress data: {e}")


def compress_data(data: Union[str, bytes], method: str = "gzip", **kwargs) -> bytes:
    """Compress data using specified method"""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if method == "gzip":
        return compress_gzip(data, **kwargs)
    elif method == "deflate":
        return compress_deflate(data, **kwargs)
    else:
        raise ValueError(f"Unsupported compression method: {method}")


def decompress_data(data: bytes, method: str = "gzip") -> bytes:
    """Decompress data using specified method"""
    if method == "gzip":
        return decompress_gzip(data)
    elif method == "deflate":
        return decompress_deflate(data)
    else:
        raise ValueError(f"Unsupported decompression method: {method}")


def get_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio"""
    if original_size == 0:
        return 0.0
    return (1 - compressed_size / original_size) * 100.0


def should_compress(data: bytes, min_size: int = 1024) -> bool:
    """Determine if data should be compressed based on size"""
    return len(data) >= min_size

