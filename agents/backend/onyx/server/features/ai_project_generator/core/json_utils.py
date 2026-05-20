"""
JSON Utilities - Utilidades para serialización JSON
==================================================

Utilidades centralizadas para serialización/deserialización JSON usando orjson.
Para casos que requieren indentación (archivos de configuración legibles), usa json estándar.
"""

from typing import Any, Union
import orjson
import json


def json_dumps(obj: Any, **kwargs) -> bytes:
    """
    Serializa objeto a JSON usando orjson (más rápido que json estándar).
    
    Args:
        obj: Objeto a serializar
        **kwargs: Opciones adicionales para orjson
        
    Returns:
        JSON como bytes
    """
    options = (
        orjson.OPT_SERIALIZE_NUMPY |
        orjson.OPT_SERIALIZE_DATACLASS |
        orjson.OPT_NON_STR_KEYS
    )
    return orjson.dumps(obj, option=options, **kwargs)


def json_loads(data: Union[bytes, str]) -> Any:
    """
    Deserializa JSON usando orjson.
    
    Args:
        data: JSON como bytes o string
        
    Returns:
        Objeto deserializado
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return orjson.loads(data)


def json_dumps_str(obj: Any, **kwargs) -> str:
    """
    Serializa objeto a JSON y retorna como string.
    Usa orjson para mejor rendimiento.
    
    Args:
        obj: Objeto a serializar
        **kwargs: Opciones adicionales
        
    Returns:
        JSON como string
    """
    return json_dumps(obj, **kwargs).decode('utf-8')


def json_dumps_pretty(obj: Any, indent: int = 2, ensure_ascii: bool = False, **kwargs) -> str:
    """
    Serializa objeto a JSON con formato legible (indentado).
    Usa json estándar porque orjson no soporta indentación.
    
    Args:
        obj: Objeto a serializar
        indent: Número de espacios para indentación
        ensure_ascii: Si False, permite caracteres Unicode
        **kwargs: Opciones adicionales para json.dumps
        
    Returns:
        JSON formateado como string
    """
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

