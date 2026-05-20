"""
Serialization Utilities
=======================
Utilidades para serialización de datos.
"""

import json
import pickle
import base64
from typing import Any, Optional
from datetime import datetime


def serialize_json(obj: Any, pretty: bool = False) -> str:
    """
    Serializar objeto a JSON.
    
    Args:
        obj: Objeto a serializar
        pretty: Formatear de forma legible
        
    Returns:
        String JSON
    """
    indent = 2 if pretty else None
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def deserialize_json(json_str: str) -> Any:
    """
    Deserializar JSON a objeto.
    
    Args:
        json_str: String JSON
        
    Returns:
        Objeto deserializado
    """
    return json.loads(json_str)


def serialize_pickle(obj: Any) -> str:
    """
    Serializar objeto usando pickle (base64).
    
    Args:
        obj: Objeto a serializar
        
    Returns:
        String base64
    """
    pickled = pickle.dumps(obj)
    return base64.b64encode(pickled).decode()


def deserialize_pickle(pickle_str: str) -> Any:
    """
    Deserializar pickle desde base64.
    
    Args:
        pickle_str: String base64
        
    Returns:
        Objeto deserializado
    """
    pickled = base64.b64decode(pickle_str.encode())
    return pickle.loads(pickled)


def serialize_dict_safe(obj: Any) -> dict:
    """
    Serializar objeto a diccionario de forma segura.
    
    Args:
        obj: Objeto a serializar
        
    Returns:
        Diccionario serializado
    """
    if isinstance(obj, dict):
        return {k: serialize_dict_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_dict_safe(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)

