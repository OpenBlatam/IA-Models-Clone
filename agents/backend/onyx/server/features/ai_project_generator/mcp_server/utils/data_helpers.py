"""
Data Helpers - Utilidades para manipulación de datos
=====================================================

Funciones helper para transformación y validación de datos.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)

T = TypeVar('T')


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge profundo de diccionarios.
    
    Args:
        base: Diccionario base
        update: Diccionario con actualizaciones
    
    Returns:
        Diccionario combinado
    """
    result = base.copy()
    
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(
    data: Dict[str, Any],
    separator: str = ".",
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Aplanar diccionario anidado.
    
    Args:
        data: Diccionario a aplanar
        separator: Separador para keys anidadas
        prefix: Prefijo para keys (usado recursivamente)
    
    Returns:
        Diccionario aplanado
    """
    result = {}
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        else:
            result[new_key] = value
    
    return result


def unflatten_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """
    Desaplanar diccionario.
    
    Args:
        data: Diccionario aplanado
        separator: Separador usado en keys
    
    Returns:
        Diccionario anidado
    """
    result = {}
    
    for key, value in data.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def filter_dict(
    data: Dict[str, Any],
    keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
    predicate: Optional[Callable[[str, Any], bool]] = None
) -> Dict[str, Any]:
    """
    Filtrar diccionario por keys o predicado.
    
    Args:
        data: Diccionario a filtrar
        keys: Lista de keys a incluir (opcional)
        exclude_keys: Lista de keys a excluir (opcional)
        predicate: Función de filtrado (key, value) -> bool (opcional)
    
    Returns:
        Diccionario filtrado
    """
    result = {}
    
    for key, value in data.items():
        # Filtrar por keys incluidas
        if keys and key not in keys:
            continue
        
        # Filtrar por keys excluidas
        if exclude_keys and key in exclude_keys:
            continue
        
        # Filtrar por predicado
        if predicate and not predicate(key, value):
            continue
        
        result[key] = value
    
    return result


def transform_dict(
    data: Dict[str, Any],
    key_mapping: Optional[Dict[str, str]] = None,
    value_transformers: Optional[Dict[str, Callable[[Any], Any]]] = None
) -> Dict[str, Any]:
    """
    Transformar diccionario con mapeo de keys y transformación de valores.
    
    Args:
        data: Diccionario a transformar
        key_mapping: Mapeo de keys antiguas a nuevas (opcional)
        value_transformers: Transformadores de valores por key (opcional)
    
    Returns:
        Diccionario transformado
    """
    result = {}
    
    for key, value in data.items():
        # Mapear key
        new_key = key_mapping.get(key, key) if key_mapping else key
        
        # Transformar valor
        if value_transformers and new_key in value_transformers:
            value = value_transformers[new_key](value)
        
        result[new_key] = value
    
    return result


def safe_json_loads(
    data: str,
    default: Any = None
) -> Any:
    """
    Cargar JSON de forma segura.
    
    Args:
        data: String JSON
        default: Valor por defecto si falla
    
    Returns:
        Objeto parseado o default
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(
    data: Any,
    default: str = "{}",
    **kwargs
) -> str:
    """
    Serializar a JSON de forma segura.
    
    Args:
        data: Datos a serializar
        default: String por defecto si falla
        **kwargs: Argumentos adicionales para json.dumps
    
    Returns:
        String JSON o default
    """
    try:
        return json.dumps(data, **kwargs)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize to JSON: {e}")
        return default


def normalize_data(
    data: Any,
    max_depth: int = 10
) -> Any:
    """
    Normalizar datos para serialización.
    
    Convierte tipos no serializables a tipos básicos.
    
    Args:
        data: Datos a normalizar
        max_depth: Profundidad máxima de recursión
    
    Returns:
        Datos normalizados
    """
    if max_depth <= 0:
        return str(data)
    
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    
    if isinstance(data, datetime):
        return data.isoformat()
    
    if isinstance(data, Decimal):
        return float(data)
    
    if isinstance(data, dict):
        return {
            str(k): normalize_data(v, max_depth - 1)
            for k, v in data.items()
        }
    
    if isinstance(data, (list, tuple)):
        return [normalize_data(item, max_depth - 1) for item in data]
    
    if hasattr(data, '__dict__'):
        return normalize_data(data.__dict__, max_depth - 1)
    
    return str(data)


def chunk_list(
    items: List[Any],
    chunk_size: int
) -> List[List[Any]]:
    """
    Dividir lista en chunks.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
    
    Returns:
        Lista de chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def group_by(
    items: List[Any],
    key_func: Callable[[Any], str]
) -> Dict[str, List[Any]]:
    """
    Agrupar items por key.
    
    Args:
        items: Lista de items
        key_func: Función que extrae la key de un item
    
    Returns:
        Diccionario agrupado
    """
    result: Dict[str, List[Any]] = {}
    
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    
    return result


def deduplicate_list(
    items: List[Any],
    key_func: Optional[Callable[[Any], Any]] = None
) -> List[Any]:
    """
    Eliminar duplicados de lista.
    
    Args:
        items: Lista de items
        key_func: Función para extraer key de comparación (opcional)
    
    Returns:
        Lista sin duplicados
    """
    if key_func:
        seen = set()
        result = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    
    return list(dict.fromkeys(items))  # Preserva orden


def sort_by_multiple(
    items: List[Any],
    key_funcs: List[Callable[[Any], Any]],
    reverse: bool = False
) -> List[Any]:
    """
    Ordenar lista por múltiples criterios.
    
    Args:
        items: Lista a ordenar
        key_funcs: Lista de funciones de key (orden de prioridad)
        reverse: Si ordenar en reversa
    
    Returns:
        Lista ordenada
    """
    def sort_key(item: Any) -> tuple:
        return tuple(f(item) for f in key_funcs)
    
    return sorted(items, key=sort_key, reverse=reverse)

