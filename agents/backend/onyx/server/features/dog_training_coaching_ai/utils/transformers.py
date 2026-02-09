"""
Data Transformers
==================
Utilidades para transformar datos.
"""

from typing import List, Dict, Any, Optional, Callable


def transform_dict(
    data: Dict[str, Any],
    key_mapping: Dict[str, str],
    value_transformers: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Transformar diccionario con mapeo de claves y transformadores de valores.
    
    Args:
        data: Diccionario a transformar
        key_mapping: Mapeo de claves antiguas a nuevas
        value_transformers: Funciones de transformación por clave
        
    Returns:
        Diccionario transformado
    """
    result = {}
    value_transformers = value_transformers or {}
    
    for old_key, new_key in key_mapping.items():
        if old_key in data:
            value = data[old_key]
            
            # Aplicar transformador si existe
            if new_key in value_transformers:
                value = value_transformers[new_key](value)
            
            result[new_key] = value
    
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
        separator: Separador para claves anidadas
        prefix: Prefijo para claves
        
    Returns:
        Diccionario aplanado
    """
    items = []
    
    for key, value in data.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def nest_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """
    Anidar diccionario aplanado.
    
    Args:
        data: Diccionario aplanado
        separator: Separador usado en las claves
        
    Returns:
        Diccionario anidado
    """
    result = {}
    
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def group_by(
    items: List[Dict[str, Any]],
    key: str
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Agrupar items por una clave.
    
    Args:
        items: Lista de diccionarios
        key: Clave para agrupar
        
    Returns:
        Diccionario agrupado
    """
    result = {}
    
    for item in items:
        group_key = item.get(key)
        if group_key not in result:
            result[group_key] = []
        result[group_key].append(item)
    
    return result


def sort_dict_list(
    items: List[Dict[str, Any]],
    key: str,
    reverse: bool = False
) -> List[Dict[str, Any]]:
    """
    Ordenar lista de diccionarios por una clave.
    
    Args:
        items: Lista de diccionarios
        key: Clave para ordenar
        reverse: Orden inverso
        
    Returns:
        Lista ordenada
    """
    return sorted(items, key=lambda x: x.get(key, ""), reverse=reverse)

