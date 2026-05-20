"""
Compression Manager - Gestor de Compresión
==========================================

Gestión de compresión:
- Request compression
- Response compression
- Multiple algorithms (gzip, brotli, deflate)
- Compression levels
- Auto-detection
"""

import logging
import gzip
import zlib
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionAlgorithm(str, Enum):
    """Algoritmos de compresión"""
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    NONE = "none"


class CompressionManager:
    """
    Gestor de compresión.
    """
    
    def __init__(self) -> None:
        self.default_algorithm = CompressionAlgorithm.GZIP
        self.compression_levels: Dict[CompressionAlgorithm, int] = {
            CompressionAlgorithm.GZIP: 6,
            CompressionAlgorithm.DEFLATE: 6,
            CompressionAlgorithm.BROTLI: 4
        }
        self.min_size_for_compression = 1024  # bytes
    
    def compress(
        self,
        data: bytes,
        algorithm: Optional[CompressionAlgorithm] = None,
        level: Optional[int] = None
    ) -> bytes:
        """Comprime datos"""
        algorithm = algorithm or self.default_algorithm
        
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        if len(data) < self.min_size_for_compression:
            return data
        
        compression_level = level or self.compression_levels.get(algorithm, 6)
        
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                return gzip.compress(data, compresslevel=compression_level)
            elif algorithm == CompressionAlgorithm.DEFLATE:
                return zlib.compress(data, level=compression_level)
            elif algorithm == CompressionAlgorithm.BROTLI:
                try:
                    import brotli
                    return brotli.compress(data, quality=compression_level)
                except ImportError:
                    logger.warning("brotli not available, using gzip")
                    return gzip.compress(data, compresslevel=compression_level)
            else:
                return data
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return data
    
    def decompress(
        self,
        data: bytes,
        algorithm: Optional[CompressionAlgorithm] = None
    ) -> bytes:
        """Descomprime datos"""
        algorithm = algorithm or self.default_algorithm
        
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                return gzip.decompress(data)
            elif algorithm == CompressionAlgorithm.DEFLATE:
                return zlib.decompress(data)
            elif algorithm == CompressionAlgorithm.BROTLI:
                try:
                    import brotli
                    return brotli.decompress(data)
                except ImportError:
                    logger.warning("brotli not available, using gzip")
                    return gzip.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return data
    
    def detect_algorithm(self, accept_encoding: Optional[str] = None) -> CompressionAlgorithm:
        """Detecta algoritmo preferido del cliente"""
        if not accept_encoding:
            return self.default_algorithm
        
        accept_encoding_lower = accept_encoding.lower()
        
        if "br" in accept_encoding_lower or "brotli" in accept_encoding_lower:
            return CompressionAlgorithm.BROTLI
        elif "gzip" in accept_encoding_lower:
            return CompressionAlgorithm.GZIP
        elif "deflate" in accept_encoding_lower:
            return CompressionAlgorithm.DEFLATE
        else:
            return self.default_algorithm
    
    def should_compress(
        self,
        data: bytes,
        content_type: Optional[str] = None
    ) -> bool:
        """Determina si debe comprimir"""
        # No comprimir si es muy pequeño
        if len(data) < self.min_size_for_compression:
            return False
        
        # No comprimir si ya está comprimido
        if content_type:
            compressed_types = [
                "image/jpeg", "image/png", "image/gif", "image/webp",
                "video/", "audio/", "application/zip", "application/gzip"
            ]
            if any(ct in content_type for ct in compressed_types):
                return False
        
        return True
    
    def set_compression_level(
        self,
        algorithm: CompressionAlgorithm,
        level: int
    ) -> None:
        """Establece nivel de compresión"""
        if 0 <= level <= 9:
            self.compression_levels[algorithm] = level
            logger.info(f"Compression level set for {algorithm.value}: {level}")


def get_compression_manager() -> CompressionManager:
    """Obtiene gestor de compresión"""
    return CompressionManager()















