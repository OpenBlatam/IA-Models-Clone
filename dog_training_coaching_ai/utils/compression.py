"""
Compression Utilities
=====================
Utilidades para compresión de datos.
"""

import gzip
import zlib
from typing import Optional, bytes
from io import BytesIO


def compress_gzip(data: bytes, level: int = 6) -> bytes:
    """
    Comprimir datos usando gzip.
    
    Args:
        data: Datos a comprimir
        level: Nivel de compresión (1-9)
        
    Returns:
        Datos comprimidos
    """
    return gzip.compress(data, compresslevel=level)


def decompress_gzip(data: bytes) -> bytes:
    """
    Descomprimir datos gzip.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
    """
    return gzip.decompress(data)


def compress_deflate(data: bytes, level: int = 6) -> bytes:
    """
    Comprimir datos usando deflate.
    
    Args:
        data: Datos a comprimir
        level: Nivel de compresión (0-9)
        
    Returns:
        Datos comprimidos
    """
    return zlib.compress(data, level=level)


def decompress_deflate(data: bytes) -> bytes:
    """
    Descomprimir datos deflate.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
    """
    return zlib.decompress(data)


def get_compression_ratio(original: bytes, compressed: bytes) -> float:
    """
    Calcular ratio de compresión.
    
    Args:
        original: Tamaño original
        compressed: Tamaño comprimido
        
    Returns:
        Ratio de compresión (0-1)
    """
    if len(original) == 0:
        return 0.0
    return len(compressed) / len(original)

