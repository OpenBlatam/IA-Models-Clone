"""
Serialization Utilities - Utilidades de serialización
======================================================

Utilidades para serialización y deserialización de datos.
"""

import json
import pickle
from typing import Any, Dict, Optional, TypeVar, Type
from datetime import datetime, date
from enum import Enum
import base64

T = TypeVar('T')


class JSONEncoder(json.JSONEncoder):
    """JSON encoder extendido que soporta más tipos."""
    
    def default(self, obj: Any) -> Any:
        """Serializar objetos especiales."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        return super().default(obj)


def to_json(
    obj: Any,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> str:
    """
    Convertir objeto a JSON string.
    
    Args:
        obj: Objeto a serializar
        indent: Indentación (None para compacto)
        ensure_ascii: Si True, escapa caracteres no ASCII
        sort_keys: Si True, ordena las keys
    
    Returns:
        JSON string
    """
    return json.dumps(
        obj,
        cls=JSONEncoder,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys
    )


def from_json(json_str: str) -> Any:
    """
    Convertir JSON string a objeto.
    
    Args:
        json_str: JSON string
    
    Returns:
        Objeto deserializado
    """
    return json.loads(json_str)


def to_pickle(obj: Any) -> bytes:
    """
    Serializar objeto a pickle bytes.
    
    Args:
        obj: Objeto a serializar
    
    Returns:
        Bytes serializados
    """
    return pickle.dumps(obj)


def from_pickle(data: bytes) -> Any:
    """
    Deserializar objeto desde pickle bytes.
    
    Args:
        data: Bytes serializados
    
    Returns:
        Objeto deserializado
    """
    return pickle.loads(data)


def to_base64(data: bytes) -> str:
    """
    Codificar bytes a base64 string.
    
    Args:
        data: Bytes a codificar
    
    Returns:
        Base64 string
    """
    return base64.b64encode(data).decode('utf-8')


def from_base64(encoded: str) -> bytes:
    """
    Decodificar base64 string a bytes.
    
    Args:
        encoded: Base64 string
    
    Returns:
        Bytes decodificados
    """
    return base64.b64decode(encoded)


def serialize_dict(
    data: Dict[str, Any],
    format: str = 'json'
) -> str:
    """
    Serializar diccionario a string.
    
    Args:
        data: Diccionario
        format: Formato ('json' o 'pickle_base64')
    
    Returns:
        String serializado
    """
    if format == 'json':
        return to_json(data)
    elif format == 'pickle_base64':
        pickled = to_pickle(data)
        return to_base64(pickled)
    else:
        raise ValueError(f"Unsupported format: {format}")


def deserialize_dict(
    data: str,
    format: str = 'json'
) -> Dict[str, Any]:
    """
    Deserializar string a diccionario.
    
    Args:
        data: String serializado
        format: Formato ('json' o 'pickle_base64')
    
    Returns:
        Diccionario deserializado
    """
    if format == 'json':
        return from_json(data)
    elif format == 'pickle_base64':
        decoded = from_base64(data)
        return from_pickle(decoded)
    else:
        raise ValueError(f"Unsupported format: {format}")


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Cargar JSON de forma segura con valor por defecto.
    
    Args:
        json_str: JSON string
        default: Valor por defecto si falla
    
    Returns:
        Objeto deserializado o default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

