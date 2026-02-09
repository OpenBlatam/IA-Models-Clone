"""
Utils - Utilidades comunes
==========================

Funciones de utilidad reutilizables.
"""

import hashlib
import json
from typing import Any, Dict, Optional
from datetime import datetime


def generate_id(prefix: str = "") -> str:
    """
    Genera ID único.
    
    Args:
        prefix: Prefijo opcional
    
    Returns:
        ID único
    """
    import uuid
    id_str = str(uuid.uuid4())
    return f"{prefix}_{id_str}" if prefix else id_str


def hash_data(data: Any) -> str:
    """
    Genera hash de datos.
    
    Args:
        data: Datos a hashear
    
    Returns:
        Hash MD5
    """
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    return hashlib.md5(data_str.encode()).hexdigest()


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitiza diccionario removiendo None values.
    
    Args:
        data: Diccionario original
    
    Returns:
        Diccionario sanitizado
    """
    return {k: v for k, v in data.items() if v is not None}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a combinar
    
    Returns:
        Diccionario combinado
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def format_duration(seconds: float) -> str:
    """
    Formatea duración en formato legible.
    
    Args:
        seconds: Segundos
    
    Returns:
        String formateado
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


def validate_required_fields(data: Dict[str, Any], required: list) -> None:
    """
    Valida que campos requeridos estén presentes.
    
    Args:
        data: Datos a validar
        required: Lista de campos requeridos
    
    Raises:
        ValueError: Si falta algún campo requerido
    """
    missing = [field for field in required if field not in data or data[field] is None]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """
    Parsea string de datetime.
    
    Args:
        dt_str: String de datetime
    
    Returns:
        Datetime o None
    """
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return None















