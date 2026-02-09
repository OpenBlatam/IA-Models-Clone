"""
Utilidades para compresión de respuestas

Incluye funciones para comprimir respuestas HTTP de forma eficiente.
"""

import gzip
import brotli
from typing import Optional, bytes as BytesType
import logging

logger = logging.getLogger(__name__)


def compress_gzip(data: bytes) -> bytes:
    """
    Comprime datos usando gzip.
    
    Args:
        data: Datos a comprimir
        
    Returns:
        Datos comprimidos
    """
    try:
        return gzip.compress(data, compresslevel=6)  # Balance entre velocidad y tamaño
    except Exception as e:
        logger.warning(f"Gzip compression failed: {e}")
        return data


def compress_brotli(data: bytes) -> bytes:
    """
    Comprime datos usando Brotli (mejor compresión que gzip).
    
    Args:
        data: Datos a comprimir
        
    Returns:
        Datos comprimidos
    """
    try:
        return brotli.compress(data, quality=4)  # Balance entre velocidad y tamaño
    except Exception as e:
        logger.warning(f"Brotli compression failed: {e}")
        return data


def get_best_compression(data: bytes, accept_encoding: Optional[str] = None) -> tuple[bytes, str]:
    """
    Obtiene la mejor compresión disponible según Accept-Encoding.
    
    Args:
        data: Datos a comprimir
        accept_encoding: Header Accept-Encoding del request
        
    Returns:
        Tupla (datos_comprimidos, encoding_usado)
    """
    if not accept_encoding:
        return data, "identity"
    
    accept_encoding_lower = accept_encoding.lower()
    
    # Brotli tiene mejor compresión
    if "br" in accept_encoding_lower or "brotli" in accept_encoding_lower:
        try:
            compressed = compress_brotli(data)
            if len(compressed) < len(data):
                return compressed, "br"
        except Exception:
            pass
    
    # Gzip como fallback
    if "gzip" in accept_encoding_lower:
        try:
            compressed = compress_gzip(data)
            if len(compressed) < len(data):
                return compressed, "gzip"
        except Exception:
            pass
    
    # Sin compresión si no hay beneficio o no está soportado
    return data, "identity"


def should_compress(data: bytes, min_size: int = 1024) -> bool:
    """
    Determina si vale la pena comprimir los datos.
    
    Args:
        data: Datos a evaluar
        min_size: Tamaño mínimo para considerar compresión
        
    Returns:
        True si se debe comprimir, False en caso contrario
    """
    return len(data) >= min_size

