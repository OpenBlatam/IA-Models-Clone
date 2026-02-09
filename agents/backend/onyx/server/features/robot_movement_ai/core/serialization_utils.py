"""
Serialization Utilities
=======================

Utilidades para serialización y deserialización de datos.
"""

import json
import pickle
from typing import Any, Optional, Union
from pathlib import Path
import base64


def serialize_json(data: Any, indent: Optional[int] = None) -> str:
    """
    Serializar datos a JSON.
    
    Args:
        data: Datos a serializar
        indent: Indentación (None = compacto)
    
    Returns:
        String JSON
    
    Raises:
        TypeError: Si los datos no son serializables
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except TypeError as e:
        raise TypeError(f"Data not JSON serializable: {e}") from e


def deserialize_json(json_str: str) -> Any:
    """
    Deserializar JSON a datos.
    
    Args:
        json_str: String JSON
    
    Returns:
        Datos deserializados
    
    Raises:
        json.JSONDecodeError: Si el JSON es inválido
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON: {e.msg}",
            e.doc,
            e.pos
        ) from e


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Guardar datos en archivo JSON.
    
    Args:
        data: Datos a guardar
        file_path: Ruta del archivo
        indent: Indentación
    
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    from .config_utils import ensure_dir
    
    path = Path(file_path)
    ensure_dir(path.parent)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    except IOError as e:
        raise IOError(f"Cannot write JSON file: {e}") from e


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Cargar datos desde archivo JSON.
    
    Args:
        file_path: Ruta del archivo
    
    Returns:
        Datos cargados
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        json.JSONDecodeError: Si el JSON es inválido
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file: {e.msg}",
            e.doc,
            e.pos
        ) from e


def serialize_pickle(data: Any) -> bytes:
    """
    Serializar datos con pickle.
    
    Args:
        data: Datos a serializar
    
    Returns:
        Bytes serializados
    
    Raises:
        pickle.PicklingError: Si no se puede serializar
    """
    try:
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    except pickle.PicklingError as e:
        raise pickle.PicklingError(f"Cannot pickle data: {e}") from e


def deserialize_pickle(data: bytes) -> Any:
    """
    Deserializar datos con pickle.
    
    Args:
        data: Bytes serializados
    
    Returns:
        Datos deserializados
    
    Raises:
        pickle.UnpicklingError: Si no se puede deserializar
    """
    try:
        return pickle.loads(data)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Cannot unpickle data: {e}") from e


def encode_base64(data: bytes) -> str:
    """
    Codificar bytes a base64.
    
    Args:
        data: Bytes a codificar
    
    Returns:
        String base64
    """
    return base64.b64encode(data).decode('utf-8')


def decode_base64(data: str) -> bytes:
    """
    Decodificar base64 a bytes.
    
    Args:
        data: String base64
    
    Returns:
        Bytes decodificados
    
    Raises:
        ValueError: Si el string no es base64 válido
    """
    try:
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}") from e

