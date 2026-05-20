"""
Compression - Utilidades de Compresión
======================================

Utilidades para comprimir y descomprimir datos.
"""

import logging
import gzip
import zlib
import base64
from typing import Union, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar importar compresiones avanzadas
try:
    import bz2
    _has_bz2 = True
except ImportError:
    _has_bz2 = False

try:
    import lzma
    _has_lzma = True
except ImportError:
    _has_lzma = False


def compress_gzip(data: Union[str, bytes], level: int = 6) -> bytes:
    """
    Comprimir datos con gzip.
    
    Args:
        data: Datos a comprimir (string o bytes)
        level: Nivel de compresión (1-9)
        
    Returns:
        Datos comprimidos
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return gzip.compress(data, compresslevel=level)


def decompress_gzip(data: bytes) -> bytes:
    """
    Descomprimir datos con gzip.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
    """
    return gzip.decompress(data)


def compress_zlib(data: Union[str, bytes], level: int = 6) -> bytes:
    """
    Comprimir datos con zlib.
    
    Args:
        data: Datos a comprimir
        level: Nivel de compresión (1-9)
        
    Returns:
        Datos comprimidos
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return zlib.compress(data, level=level)


def decompress_zlib(data: bytes) -> bytes:
    """
    Descomprimir datos con zlib.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
    """
    return zlib.decompress(data)


def compress_bz2(data: Union[str, bytes], level: int = 9) -> bytes:
    """
    Comprimir datos con bz2.
    
    Args:
        data: Datos a comprimir
        level: Nivel de compresión (1-9)
        
    Returns:
        Datos comprimidos
        
    Raises:
        ImportError: Si bz2 no está disponible
    """
    if not _has_bz2:
        raise ImportError("bz2 is not available")
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return bz2.compress(data, compresslevel=level)


def decompress_bz2(data: bytes) -> bytes:
    """
    Descomprimir datos con bz2.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
        
    Raises:
        ImportError: Si bz2 no está disponible
    """
    if not _has_bz2:
        raise ImportError("bz2 is not available")
    
    return bz2.decompress(data)


def compress_lzma(data: Union[str, bytes], level: int = 6) -> bytes:
    """
    Comprimir datos con lzma.
    
    Args:
        data: Datos a comprimir
        level: Nivel de compresión (0-9)
        
    Returns:
        Datos comprimidos
        
    Raises:
        ImportError: Si lzma no está disponible
    """
    if not _has_lzma:
        raise ImportError("lzma is not available")
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return lzma.compress(data, preset=level)


def decompress_lzma(data: bytes) -> bytes:
    """
    Descomprimir datos con lzma.
    
    Args:
        data: Datos comprimidos
        
    Returns:
        Datos descomprimidos
        
    Raises:
        ImportError: Si lzma no está disponible
    """
    if not _has_lzma:
        raise ImportError("lzma is not available")
    
    return lzma.decompress(data)


def compress_to_base64(
    data: Union[str, bytes],
    method: str = "gzip"
) -> str:
    """
    Comprimir y codificar en base64.
    
    Args:
        data: Datos a comprimir
        method: Método de compresión (gzip, zlib, bz2, lzma)
        
    Returns:
        String base64 con datos comprimidos
    """
    compressors = {
        "gzip": compress_gzip,
        "zlib": compress_zlib,
        "bz2": compress_bz2 if _has_bz2 else None,
        "lzma": compress_lzma if _has_lzma else None
    }
    
    compressor = compressors.get(method)
    if not compressor:
        raise ValueError(f"Unsupported compression method: {method}")
    
    compressed = compressor(data)
    return base64.b64encode(compressed).decode('utf-8')


def decompress_from_base64(
    data: str,
    method: str = "gzip"
) -> bytes:
    """
    Descodificar base64 y descomprimir.
    
    Args:
        data: String base64 con datos comprimidos
        method: Método de compresión usado
        
    Returns:
        Datos descomprimidos
    """
    decompressors = {
        "gzip": decompress_gzip,
        "zlib": decompress_zlib,
        "bz2": decompress_bz2 if _has_bz2 else None,
        "lzma": decompress_lzma if _has_lzma else None
    }
    
    decompressor = decompressors.get(method)
    if not decompressor:
        raise ValueError(f"Unsupported compression method: {method}")
    
    compressed = base64.b64decode(data)
    return decompressor(compressed)


def compress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    method: str = "gzip"
) -> Path:
    """
    Comprimir archivo.
    
    Args:
        input_path: Ruta del archivo a comprimir
        output_path: Ruta de salida (opcional, se genera si no se proporciona)
        method: Método de compresión
        
    Returns:
        Ruta del archivo comprimido
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.with_suffix(f"{input_path.suffix}.{method}")
    else:
        output_path = Path(output_path)
    
    data = input_path.read_bytes()
    
    compressors = {
        "gzip": compress_gzip,
        "zlib": compress_zlib,
        "bz2": compress_bz2 if _has_bz2 else None,
        "lzma": compress_lzma if _has_lzma else None
    }
    
    compressor = compressors.get(method)
    if not compressor:
        raise ValueError(f"Unsupported compression method: {method}")
    
    compressed = compressor(data)
    output_path.write_bytes(compressed)
    
    logger.info(f"📦 File compressed: {input_path} -> {output_path}")
    return output_path


def decompress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    method: str = "gzip"
) -> Path:
    """
    Descomprimir archivo.
    
    Args:
        input_path: Ruta del archivo comprimido
        output_path: Ruta de salida (opcional)
        method: Método de compresión usado
        
    Returns:
        Ruta del archivo descomprimido
    """
    input_path = Path(input_path)
    
    if output_path is None:
        # Remover extensión de compresión
        output_path = input_path.with_suffix('')
    else:
        output_path = Path(output_path)
    
    compressed = input_path.read_bytes()
    
    decompressors = {
        "gzip": decompress_gzip,
        "zlib": decompress_zlib,
        "bz2": decompress_bz2 if _has_bz2 else None,
        "lzma": decompress_lzma if _has_lzma else None
    }
    
    decompressor = decompressors.get(method)
    if not decompressor:
        raise ValueError(f"Unsupported compression method: {method}")
    
    data = decompressor(compressed)
    output_path.write_bytes(data)
    
    logger.info(f"📦 File decompressed: {input_path} -> {output_path}")
    return output_path




