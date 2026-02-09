"""
Collection Utilities
====================

Utilidades para trabajar con colecciones (listas, diccionarios, etc.).
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Callable
from collections import defaultdict, Counter

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def group_by(items: List[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    """
    Agrupar items por una función clave.
    
    Args:
        items: Lista de items
        key_func: Función que retorna la clave
    
    Returns:
        Diccionario agrupado
    """
    grouped: Dict[K, List[T]] = defaultdict(list)
    for item in items:
        grouped[key_func(item)].append(item)
    return dict(grouped)


def flatten(nested_list: List[List[T]]) -> List[T]:
    """
    Aplanar lista anidada.
    
    Args:
        nested_list: Lista de listas
    
    Returns:
        Lista aplanada
    """
    return [item for sublist in nested_list for item in sublist]


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Dividir lista en chunks.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
    
    Returns:
        Lista de chunks
    
    Raises:
        ValueError: Si chunk_size <= 0
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def remove_duplicates(items: List[T], key_func: Optional[Callable[[T], Any]] = None) -> List[T]:
    """
    Remover duplicados preservando orden.
    
    Args:
        items: Lista con posibles duplicados
        key_func: Función opcional para determinar clave de unicidad
    
    Returns:
        Lista sin duplicados
    """
    if key_func is None:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in items:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


def merge_dicts(*dicts: Dict[K, V], overwrite: bool = True) -> Dict[K, V]:
    """
    Fusionar múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios a fusionar
        overwrite: Si True, valores posteriores sobrescriben anteriores
    
    Returns:
        Diccionario fusionado
    """
    result: Dict[K, V] = {}
    for d in dicts:
        if overwrite:
            result.update(d)
        else:
            for key, value in d.items():
                if key not in result:
                    result[key] = value
    return result


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtener valor anidado desde diccionario usando path.
    
    Args:
        data: Diccionario
        path: Path separado por puntos (ej: "user.profile.name")
        default: Valor por defecto si no existe
    
    Returns:
        Valor encontrado o default
    """
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_nested_value(data: Dict[str, Any], path: str, value: Any) -> None:
    """
    Establecer valor anidado en diccionario usando path.
    
    Args:
        data: Diccionario
        path: Path separado por puntos (ej: "user.profile.name")
        value: Valor a establecer
    """
    keys = path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def most_common(items: List[T], n: int = 1) -> List[Tuple[T, int]]:
    """
    Obtener n items más comunes.
    
    Args:
        items: Lista de items
        n: Número de items a retornar
    
    Returns:
        Lista de tuplas (item, count)
    """
    counter = Counter(items)
    return counter.most_common(n)

