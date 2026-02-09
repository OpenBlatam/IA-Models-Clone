"""
Intelligent Compression for Flux2 Clothing Changer
===================================================

Intelligent compression and optimization.
"""

import zlib
import gzip
import bz2
import lzma
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression method."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    AUTO = "auto"


@dataclass
class CompressionResult:
    """Compression result."""
    method: CompressionMethod
    original_size: int
    compressed_size: int
    ratio: float
    compression_time: float


class IntelligentCompression:
    """Intelligent compression system."""
    
    def __init__(self):
        """Initialize intelligent compression."""
        self.compression_stats: Dict[str, Dict[str, Any]] = {}
    
    def compress(
        self,
        data: bytes,
        method: CompressionMethod = CompressionMethod.AUTO,
        level: int = 6,
    ) -> Tuple[bytes, CompressionResult]:
        """
        Compress data intelligently.
        
        Args:
            data: Data to compress
            method: Compression method
            level: Compression level (1-9)
            
        Returns:
            Tuple of (compressed_data, result)
        """
        original_size = len(data)
        start_time = time.time()
        
        if method == CompressionMethod.AUTO:
            method = self._select_best_method(data)
        
        compressed = self._compress_with_method(data, method, level)
        compression_time = time.time() - start_time
        
        compressed_size = len(compressed)
        ratio = compressed_size / original_size if original_size > 0 else 0.0
        
        result = CompressionResult(
            method=method,
            original_size=original_size,
            compressed_size=compressed_size,
            ratio=ratio,
            compression_time=compression_time,
        )
        
        return compressed, result
    
    def decompress(
        self,
        data: bytes,
        method: CompressionMethod,
    ) -> bytes:
        """
        Decompress data.
        
        Args:
            data: Compressed data
            method: Compression method used
            
        Returns:
            Decompressed data
        """
        return self._decompress_with_method(data, method)
    
    def _select_best_method(self, data: bytes) -> CompressionMethod:
        """Select best compression method for data."""
        # Quick test of different methods
        methods = [
            CompressionMethod.GZIP,
            CompressionMethod.ZLIB,
            CompressionMethod.BZ2,
        ]
        
        best_method = CompressionMethod.GZIP
        best_size = len(data)
        
        for method in methods:
            try:
                compressed = self._compress_with_method(data, method, 6)
                if len(compressed) < best_size:
                    best_size = len(compressed)
                    best_method = method
            except Exception:
                continue
        
        return best_method
    
    def _compress_with_method(
        self,
        data: bytes,
        method: CompressionMethod,
        level: int,
    ) -> bytes:
        """Compress with specific method."""
        if method == CompressionMethod.NONE:
            return data
        elif method == CompressionMethod.ZLIB:
            return zlib.compress(data, level=level)
        elif method == CompressionMethod.GZIP:
            return gzip.compress(data, compresslevel=level)
        elif method == CompressionMethod.BZ2:
            return bz2.compress(data, compresslevel=level)
        elif method == CompressionMethod.LZMA:
            return lzma.compress(data, preset=level)
        else:
            return data
    
    def _decompress_with_method(
        self,
        data: bytes,
        method: CompressionMethod,
    ) -> bytes:
        """Decompress with specific method."""
        if method == CompressionMethod.NONE:
            return data
        elif method == CompressionMethod.ZLIB:
            return zlib.decompress(data)
        elif method == CompressionMethod.GZIP:
            return gzip.decompress(data)
        elif method == CompressionMethod.BZ2:
            return bz2.decompress(data)
        elif method == CompressionMethod.LZMA:
            return lzma.decompress(data)
        else:
            return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            "methods_tested": len(self.compression_stats),
            "stats": self.compression_stats,
        }


