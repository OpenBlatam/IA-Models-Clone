"""
Serialization - Utilidades de serialización y deserialización
=============================================================

Utilidades mejoradas para serialización/deserialización de datos
con soporte para múltiples formatos y optimizaciones.
"""

import json
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Intentar usar orjson para mejor rendimiento
try:
    import orjson
    _has_orjson = True
except ImportError:
    _has_orjson = False

# Intentar usar msgpack para serialización binaria
try:
    import msgpack
    _has_msgpack = True
except ImportError:
    _has_msgpack = False


def serialize_json(
    data: Any,
    pretty: bool = False,
    use_orjson: bool = True
) -> str:
    """
    Serializar datos a JSON.
    
    Args:
        data: Datos a serializar
        pretty: Si formatear con indentación
        use_orjson: Si usar orjson (más rápido)
        
    Returns:
        String JSON
    """
    if use_orjson and _has_orjson:
        if pretty:
            return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
        return orjson.dumps(data).decode()
    else:
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False)


def deserialize_json(
    json_str: str,
    use_orjson: bool = True
) -> Any:
    """
    Deserializar JSON a objeto Python.
    
    Args:
        json_str: String JSON
        use_orjson: Si usar orjson (más rápido)
        
    Returns:
        Objeto Python deserializado
    """
    if use_orjson and _has_orjson:
        return orjson.loads(json_str)
    else:
        return json.loads(json_str)


def serialize_msgpack(data: Any) -> bytes:
    """
    Serializar datos a MessagePack (formato binario).
    
    Args:
        data: Datos a serializar
        
    Returns:
        Bytes en formato MessagePack
        
    Raises:
        ImportError: Si msgpack no está disponible
    """
    if not _has_msgpack:
        raise ImportError("msgpack is not available. Install with: pip install msgpack")
    
    return msgpack.packb(data, use_bin_type=True)


def deserialize_msgpack(data: bytes) -> Any:
    """
    Deserializar MessagePack a objeto Python.
    
    Args:
        data: Bytes en formato MessagePack
        
    Returns:
        Objeto Python deserializado
        
    Raises:
        ImportError: Si msgpack no está disponible
    """
    if not _has_msgpack:
        raise ImportError("msgpack is not available. Install with: pip install msgpack")
    
    return msgpack.unpackb(data, raw=False)


def serialize_to_file(
    data: Any,
    filepath: Union[str, Path],
    format: str = "json",
    pretty: bool = False
) -> None:
    """
    Serializar datos a archivo.
    
    Args:
        data: Datos a serializar
        filepath: Ruta del archivo
        format: Formato ("json" o "msgpack")
        pretty: Si formatear JSON (solo para JSON)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        content = serialize_json(data, pretty=pretty)
        filepath.write_text(content, encoding="utf-8")
    elif format == "msgpack":
        content = serialize_msgpack(data)
        filepath.write_bytes(content)
    else:
        raise ValueError(f"Unsupported format: {format}")


def deserialize_from_file(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Any:
    """
    Deserializar datos desde archivo.
    
    Args:
        filepath: Ruta del archivo
        format: Formato ("json" o "msgpack"). Si None, detecta por extensión
        
    Returns:
        Objeto Python deserializado
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:] if filepath.suffix else "json"
    
    if format == "json":
        content = filepath.read_text(encoding="utf-8")
        return deserialize_json(content)
    elif format == "msgpack":
        content = filepath.read_bytes()
        return deserialize_msgpack(content)
    else:
        raise ValueError(f"Unsupported format: {format}")


class JSONEncoder(json.JSONEncoder):
    """Encoder JSON personalizado para tipos especiales"""
    
    def default(self, obj: Any) -> Any:
        """Convertir objetos especiales a JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def safe_serialize(
    data: Any,
    fallback: Any = None,
    use_orjson: bool = True
) -> Optional[str]:
    """
    Serializar datos de forma segura con manejo de errores.
    
    Args:
        data: Datos a serializar
        fallback: Valor a retornar si falla (default: None)
        use_orjson: Si usar orjson
        
    Returns:
        String JSON o fallback si falla
    """
    try:
        return serialize_json(data, use_orjson=use_orjson)
    except Exception as e:
        logger.warning(f"Serialization failed: {e}")
        return fallback


def safe_deserialize(
    json_str: str,
    fallback: Any = None,
    use_orjson: bool = True
) -> Any:
    """
    Deserializar JSON de forma segura con manejo de errores.
    
    Args:
        json_str: String JSON
        fallback: Valor a retornar si falla (default: None)
        use_orjson: Si usar orjson
        
    Returns:
        Objeto deserializado o fallback si falla
    """
    try:
        return deserialize_json(json_str, use_orjson=use_orjson)
    except Exception as e:
        logger.warning(f"Deserialization failed: {e}")
        return fallback




