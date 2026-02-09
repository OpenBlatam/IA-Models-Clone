"""
Request/Response Compression
Compresses responses to reduce bandwidth usage
"""

import gzip
import zlib
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression types"""
    GZIP = "gzip"
    DEFLATE = "deflate"
    NONE = "none"


class CompressionMiddleware:
    """Middleware for response compression"""
    
    def __init__(self, min_size: int = 1024, compression_type: CompressionType = CompressionType.GZIP):
        """
        Initialize compression middleware
        
        Args:
            min_size: Minimum response size to compress (bytes)
            compression_type: Compression algorithm to use
        """
        self.min_size = min_size
        self.compress_type = compression_type
    
    def compress(self, data: bytes, compression_type: Optional[CompressionType] = None) -> tuple[bytes, str]:
        """
        Compress data
        
        Args:
            data: Data to compress
            compression_type: Compression type (overrides default)
            
        Returns:
            Tuple of (compressed_data, content_encoding)
        """
        comp_type = compression_type or self.compress_type
        
        if comp_type == CompressionType.NONE or len(data) < self.min_size:
            return data, ""
        
        try:
            if comp_type == CompressionType.GZIP:
                compressed = gzip.compress(data, compresslevel=6)
                return compressed, "gzip"
            elif comp_type == CompressionType.DEFLATE:
                compressed = zlib.compress(data, level=6)
                return compressed, "deflate"
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return data, ""
        
        return data, ""
    
    def decompress(self, data: bytes, content_encoding: str) -> bytes:
        """
        Decompress data
        
        Args:
            data: Compressed data
            content_encoding: Content encoding header value
            
        Returns:
            Decompressed data
        """
        try:
            if content_encoding == "gzip":
                return gzip.decompress(data)
            elif content_encoding == "deflate":
                return zlib.decompress(data)
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return data
        
        return data
    
    def should_compress(self, content_type: str, content_length: int) -> bool:
        """
        Determine if content should be compressed
        
        Args:
            content_type: Content type
            content_length: Content length in bytes
            
        Returns:
            True if should compress
        """
        # Don't compress if too small
        if content_length < self.min_size:
            return False
        
        # Don't compress already compressed formats
        if content_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            return False
        
        # Don't compress video/audio
        if content_type.startswith("video/") or content_type.startswith("audio/"):
            return False
        
        return True


def compress_response(response, min_size: int = 1024) -> tuple[bytes, str]:
    """
    Compress response data
    
    Args:
        response: Response data (bytes or str)
        min_size: Minimum size to compress
        
    Returns:
        Tuple of (compressed_data, content_encoding)
    """
    if isinstance(response, str):
        data = response.encode('utf-8')
    else:
        data = response
    
    if len(data) < min_size:
        return data, ""
    
    try:
        compressed = gzip.compress(data, compresslevel=6)
        return compressed, "gzip"
    except Exception as e:
        logger.warning(f"Response compression failed: {e}")
        return data, ""















