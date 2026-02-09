"""
Compression module for cache system
"""

import zlib
from typing import Optional
from .constants import COMPRESSION_MARKER


class Compressor:
    """Handles data compression and decompression"""
    
    def __init__(self, enabled: bool = True, threshold: int = 1024):
        """Initialize compressor
        
        Args:
            enabled: Enable compression
            threshold: Minimum size in bytes to compress
        """
        self.enabled = enabled
        self.threshold = threshold
    
    def compress(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold, with marker
        
        Args:
            data: Data to compress
            
        Returns:
            Compressed data with marker, or original data
        """
        if not self.enabled or len(data) < self.threshold:
            return data
        compressed = zlib.compress(data, level=6)
        return COMPRESSION_MARKER + compressed
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data if marked as compressed
        
        Args:
            data: Data to decompress
            
        Returns:
            Decompressed data, or original if not compressed
        """
        if not isinstance(data, bytes):
            return data
        if len(data) < len(COMPRESSION_MARKER):
            return data
        if data[:len(COMPRESSION_MARKER)] == COMPRESSION_MARKER:
            try:
                return zlib.decompress(data[len(COMPRESSION_MARKER):])
            except zlib.error:
                return data
        return data
    
    def is_compressed(self, data: bytes) -> bool:
        """Check if data is compressed
        
        Args:
            data: Data to check
            
        Returns:
            True if compressed, False otherwise
        """
        if not isinstance(data, bytes) or len(data) < len(COMPRESSION_MARKER):
            return False
        return data[:len(COMPRESSION_MARKER)] == COMPRESSION_MARKER

