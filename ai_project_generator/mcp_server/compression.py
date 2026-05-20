"""
MCP Compression - Compresión de respuestas
===========================================
"""

import gzip
import zlib
import logging
from typing import Any, Dict, Optional
from enum import Enum

from fastapi import Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class CompressionType(str, Enum):
    """Tipos de compresión soportados"""
    GZIP = "gzip"
    DEFLATE = "deflate"
    NONE = "none"


class ResponseCompressor:
    """
    Compresor de respuestas
    
    Comprime respuestas automáticamente según el Accept-Encoding del cliente.
    """
    
    def __init__(self, min_size: int = 1024):
        """
        Args:
            min_size: Tamaño mínimo en bytes para comprimir (default: 1KB)
        """
        self.min_size = min_size
    
    def compress(
        self,
        content: bytes,
        compression_type: CompressionType = CompressionType.GZIP,
    ) -> bytes:
        """
        Comprime contenido
        
        Args:
            content: Contenido a comprimir
            compression_type: Tipo de compresión
            
        Returns:
            Contenido comprimido
        """
        if len(content) < self.min_size:
            return content
        
        if compression_type == CompressionType.GZIP:
            return gzip.compress(content)
        elif compression_type == CompressionType.DEFLATE:
            return zlib.compress(content)
        else:
            return content
    
    def create_compressed_response(
        self,
        content: Any,
        compression_type: CompressionType = CompressionType.GZIP,
        content_type: str = "application/json",
    ) -> Response:
        """
        Crea respuesta comprimida
        
        Args:
            content: Contenido a comprimir
            compression_type: Tipo de compresión
            content_type: Tipo de contenido
            
        Returns:
            Response comprimida
        """
        import json
        
        # Convertir a bytes
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, dict):
            content_bytes = json.dumps(content).encode("utf-8")
        else:
            content_bytes = str(content).encode("utf-8")
        
        # Comprimir
        compressed = self.compress(content_bytes, compression_type)
        
        # Crear response
        response = Response(
            content=compressed,
            media_type=content_type,
        )
        
        # Agregar headers de compresión
        if compression_type == CompressionType.GZIP:
            response.headers["Content-Encoding"] = "gzip"
        elif compression_type == CompressionType.DEFLATE:
            response.headers["Content-Encoding"] = "deflate"
        
        response.headers["Content-Length"] = str(len(compressed))
        response.headers["Vary"] = "Accept-Encoding"
        
        return response
    
    def should_compress(self, content: Any) -> bool:
        """
        Determina si se debe comprimir contenido
        
        Args:
            content: Contenido a evaluar
            
        Returns:
            True si se debe comprimir
        """
        import json
        
        if isinstance(content, str):
            size = len(content.encode("utf-8"))
        elif isinstance(content, dict):
            size = len(json.dumps(content).encode("utf-8"))
        else:
            size = len(str(content).encode("utf-8"))
        
        return size >= self.min_size


def get_compression_type(accept_encoding: Optional[str]) -> CompressionType:
    """
    Determina tipo de compresión desde Accept-Encoding header
    
    Args:
        accept_encoding: Valor del header Accept-Encoding
        
    Returns:
        Tipo de compresión
    """
    if not accept_encoding:
        return CompressionType.NONE
    
    accept_encoding_lower = accept_encoding.lower()
    
    if "gzip" in accept_encoding_lower:
        return CompressionType.GZIP
    elif "deflate" in accept_encoding_lower:
        return CompressionType.DEFLATE
    else:
        return CompressionType.NONE

