"""
Helper utilities for MCP Server
===============================

Utilidades auxiliares para el servidor MCP.
"""

import logging
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """
    Generar ID único para request.
    
    Returns:
        UUID string
    """
    import uuid
    return str(uuid.uuid4())


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """
    Generar hash de un string.
    
    Args:
        value: String a hashear
        algorithm: Algoritmo de hash (default: sha256)
        
    Returns:
        Hash hexadecimal
    """
    if algorithm == "sha256":
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(value.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def sanitize_for_logging(data: Any, max_length: int = 1000) -> str:
    """
    Sanitizar datos para logging (evitar logs excesivamente largos).
    
    Args:
        data: Datos a sanitizar
        max_length: Longitud máxima (default: 1000)
        
    Returns:
        String sanitizado
    """
    if data is None:
        return "None"
    
    data_str = str(data)
    if len(data_str) > max_length:
        return f"{data_str[:max_length]}... (truncated, {len(data_str)} chars)"
    
    return data_str


def format_duration(seconds: float) -> str:
    """
    Formatear duración en formato legible.
    
    Args:
        seconds: Duración en segundos
        
    Returns:
        String formateado (ej: "1.234s", "123ms")
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def merge_metadata(*metadata_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combinar múltiples diccionarios de metadata.
    
    Args:
        *metadata_dicts: Diccionarios de metadata a combinar
        
    Returns:
        Diccionario combinado (últimos valores tienen prioridad)
    """
    result: Dict[str, Any] = {}
    for metadata in metadata_dicts:
        if metadata:
            result.update(metadata)
    return result


def validate_timestamp(timestamp: Any) -> Optional[datetime]:
    """
    Validar y convertir timestamp.
    
    Args:
        timestamp: Timestamp a validar (datetime, string ISO, o None)
        
    Returns:
        datetime object o None
    """
    if timestamp is None:
        return None
    
    if isinstance(timestamp, datetime):
        return timestamp
    
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except Exception:
            logger.warning(f"Invalid timestamp format: {timestamp}")
            return None
    
    return None

