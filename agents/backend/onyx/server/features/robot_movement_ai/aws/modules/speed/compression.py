"""
Compression Manager
===================

Fast compression for responses and data.
"""

import logging
import gzip
import zlib
import lzma
from typing import Optional, Union, bytes
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Compression types."""
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    LZMA = "lzma"
    ZSTD = "zstd"


class CompressionManager:
    """Fast compression manager."""
    
    def __init__(self, default_type: CompressionType = CompressionType.GZIP):
        self.default_type = default_type
        self._brotli_available = False
        self._zstd_available = False
        
        # Check for optional libraries
        try:
            import brotli
            self._brotli_available = True
        except ImportError:
            pass
        
        try:
            import zstandard
            self._zstd_available = True
        except ImportError:
            pass
    
    def compress(
        self,
        data: Union[str, bytes],
        compression_type: Optional[CompressionType] = None
    ) -> bytes:
        """Compress data."""
        compression_type = compression_type or self.default_type
        
        if isinstance(data, str):
            data = data.encode("utf-8")
        
        if compression_type == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=6)
        
        elif compression_type == CompressionType.DEFLATE:
            return zlib.compress(data, level=6)
        
        elif compression_type == CompressionType.BROTLI:
            if not self._brotli_available:
                logger.warning("Brotli not available, using gzip")
                return gzip.compress(data)
            import brotli
            return brotli.compress(data, quality=6)
        
        elif compression_type == CompressionType.LZMA:
            return lzma.compress(data, preset=6)
        
        elif compression_type == CompressionType.ZSTD:
            if not self._zstd_available:
                logger.warning("Zstd not available, using gzip")
                return gzip.compress(data)
            import zstandard
            cctx = zstandard.ZstdCompressor(level=6)
            return cctx.compress(data)
        
        else:
            return gzip.compress(data)
    
    def decompress(
        self,
        data: bytes,
        compression_type: Optional[CompressionType] = None
    ) -> bytes:
        """Decompress data."""
        compression_type = compression_type or self.default_type
        
        if compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        
        elif compression_type == CompressionType.DEFLATE:
            return zlib.decompress(data)
        
        elif compression_type == CompressionType.BROTLI:
            if not self._brotli_available:
                return gzip.decompress(data)
            import brotli
            return brotli.decompress(data)
        
        elif compression_type == CompressionType.LZMA:
            return lzma.decompress(data)
        
        elif compression_type == CompressionType.ZSTD:
            if not self._zstd_available:
                return gzip.decompress(data)
            import zstandard
            dctx = zstandard.ZstdDecompressor()
            return dctx.decompress(data)
        
        else:
            return gzip.decompress(data)
    
    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Get compression ratio."""
        if len(original) == 0:
            return 0.0
        return (1 - len(compressed) / len(original)) * 100















