"""
Serialization Utilities - Utilidades avanzadas de serialización
=================================================================

Utilidades para serialización/deserialización de datos con soporte
para múltiples formatos y transformaciones avanzadas.
"""

import logging
import json
import pickle
import base64
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.debug("PyYAML not available")

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    logger.debug("msgpack not available")


class JSONEncoder(json.JSONEncoder):
    """JSON encoder personalizado con soporte para tipos especiales."""
    
    def default(self, obj: Any) -> Any:
        """Convertir objetos especiales a JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def serialize_json(
    data: Any,
    indent: Optional[int] = None,
    ensure_ascii: bool = False,
    sort_keys: bool = False
) -> str:
    """
    Serializar datos a JSON.
    
    Args:
        data: Datos a serializar
        indent: Indentación (None = compacto)
        ensure_ascii: Si True, escapa caracteres no ASCII
        sort_keys: Si True, ordena las keys
    
    Returns:
        String JSON
    
    Example:
        json_str = serialize_json({"key": "value"}, indent=2)
    """
    return json.dumps(
        data,
        cls=JSONEncoder,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys
    )


def deserialize_json(data: str) -> Any:
    """
    Deserializar JSON a objeto Python.
    
    Args:
        data: String JSON
    
    Returns:
        Objeto Python
    
    Example:
        obj = deserialize_json('{"key": "value"}')
    """
    return json.loads(data)


def serialize_yaml(data: Any, default_flow_style: bool = False) -> str:
    """
    Serializar datos a YAML.
    
    Args:
        data: Datos a serializar
        default_flow_style: Si True, usa estilo flow
    
    Returns:
        String YAML
    
    Raises:
        ImportError: Si PyYAML no está instalado
    
    Example:
        yaml_str = serialize_yaml({"key": "value"})
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    return yaml.safe_dump(data, default_flow_style=default_flow_style)


def deserialize_yaml(data: str) -> Any:
    """
    Deserializar YAML a objeto Python.
    
    Args:
        data: String YAML
    
    Returns:
        Objeto Python
    
    Raises:
        ImportError: Si PyYAML no está instalado
    
    Example:
        obj = deserialize_yaml("key: value")
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    return yaml.safe_load(data)


def serialize_msgpack(data: Any) -> bytes:
    """
    Serializar datos a MessagePack.
    
    Args:
        data: Datos a serializar
    
    Returns:
        Bytes MessagePack
    
    Raises:
        ImportError: Si msgpack no está instalado
    
    Example:
        packed = serialize_msgpack({"key": "value"})
    """
    if not MSGPACK_AVAILABLE:
        raise ImportError("msgpack not installed. Install with: pip install msgpack")
    
    return msgpack.packb(data, use_bin_type=True)


def deserialize_msgpack(data: bytes) -> Any:
    """
    Deserializar MessagePack a objeto Python.
    
    Args:
        data: Bytes MessagePack
    
    Returns:
        Objeto Python
    
    Raises:
        ImportError: Si msgpack no está instalado
    
    Example:
        obj = deserialize_msgpack(packed_bytes)
    """
    if not MSGPACK_AVAILABLE:
        raise ImportError("msgpack not installed. Install with: pip install msgpack")
    
    return msgpack.unpackb(data, raw=False)


def serialize_pickle(data: Any) -> bytes:
    """
    Serializar datos a Pickle.
    
    Args:
        data: Datos a serializar
    
    Returns:
        Bytes Pickle
    
    Example:
        pickled = serialize_pickle({"key": "value"})
    """
    return pickle.dumps(data)


def deserialize_pickle(data: bytes) -> Any:
    """
    Deserializar Pickle a objeto Python.
    
    Args:
        data: Bytes Pickle
    
    Returns:
        Objeto Python
    
    Example:
        obj = deserialize_pickle(pickled_bytes)
    """
    return pickle.loads(data)


def serialize_base64(data: bytes) -> str:
    """
    Serializar bytes a Base64.
    
    Args:
        data: Bytes a serializar
    
    Returns:
        String Base64
    
    Example:
        b64 = serialize_base64(b"data")
    """
    return base64.b64encode(data).decode('utf-8')


def deserialize_base64(data: str) -> bytes:
    """
    Deserializar Base64 a bytes.
    
    Args:
        data: String Base64
    
    Returns:
        Bytes
    
    Example:
        bytes_data = deserialize_base64(b64_str)
    """
    return base64.b64decode(data)


class Serializer:
    """
    Serializador genérico con soporte para múltiples formatos.
    """
    
    FORMATS = {
        'json': (serialize_json, deserialize_json),
        'yaml': (serialize_yaml, deserialize_yaml) if YAML_AVAILABLE else None,
        'msgpack': (serialize_msgpack, deserialize_msgpack) if MSGPACK_AVAILABLE else None,
        'pickle': (serialize_pickle, deserialize_pickle),
    }
    
    @classmethod
    def serialize(cls, data: Any, format: str = 'json', **kwargs) -> Union[str, bytes]:
        """
        Serializar datos.
        
        Args:
            data: Datos a serializar
            format: Formato ('json', 'yaml', 'msgpack', 'pickle')
            **kwargs: Argumentos adicionales para el serializador
        
        Returns:
            Datos serializados (str o bytes)
        
        Raises:
            ValueError: Si el formato no es soportado
        
        Example:
            serialized = Serializer.serialize({"key": "value"}, format="json")
        """
        if format not in cls.FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        serializer, _ = cls.FORMATS[format]
        if format == 'json':
            return serializer(data, **kwargs)
        else:
            return serializer(data)
    
    @classmethod
    def deserialize(cls, data: Union[str, bytes], format: str = 'json') -> Any:
        """
        Deserializar datos.
        
        Args:
            data: Datos serializados
            format: Formato ('json', 'yaml', 'msgpack', 'pickle')
        
        Returns:
            Objeto Python
        
        Raises:
            ValueError: Si el formato no es soportado
        
        Example:
            obj = Serializer.deserialize(serialized, format="json")
        """
        if format not in cls.FORMATS:
            raise ValueError(f"Unsupported format: {format}")
        
        _, deserializer = cls.FORMATS[format]
        return deserializer(data)
    
    @classmethod
    def list_formats(cls) -> List[str]:
        """
        Listar formatos disponibles.
        
        Returns:
            Lista de formatos disponibles
        """
        return [fmt for fmt, impl in cls.FORMATS.items() if impl is not None]


def to_dict(obj: Any, include_none: bool = False) -> Dict[str, Any]:
    """
    Convertir objeto a diccionario.
    
    Args:
        obj: Objeto a convertir
        include_none: Si True, incluye campos None
    
    Returns:
        Diccionario
    
    Example:
        d = to_dict(my_object)
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                if include_none or value is not None:
                    if isinstance(value, (str, int, float, bool, type(None))):
                        result[key] = value
                    else:
                        result[key] = to_dict(value, include_none)
        return result
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item, include_none) for item in obj]
    else:
        return obj


def from_dict(data: Dict[str, Any], cls: Optional[type] = None) -> Any:
    """
    Convertir diccionario a objeto.
    
    Args:
        data: Diccionario
        cls: Clase para instanciar (opcional)
    
    Returns:
        Objeto o diccionario
    
    Example:
        obj = from_dict({"key": "value"}, MyClass)
    """
    if cls:
        return cls(**data)
    return data


__all__ = [
    "JSONEncoder",
    "serialize_json",
    "deserialize_json",
    "serialize_yaml",
    "deserialize_yaml",
    "serialize_msgpack",
    "deserialize_msgpack",
    "serialize_pickle",
    "deserialize_pickle",
    "serialize_base64",
    "deserialize_base64",
    "Serializer",
    "to_dict",
    "from_dict",
]

