"""
Collection Helpers
==================
Utilidades para trabajar con colecciones.
"""

from typing import List, Dict, Any, Optional, Callable, Set, Tuple


def unique(items: List[Any]) -> List[Any]:
    """
    Obtener items únicos manteniendo orden.
    
    Args:
        items: Lista de items
        
    Returns:
        Lista con items únicos
    """
    seen: Set[Any] = set()
    result = []
    
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result


def filter_by(items: List[Dict[str, Any]], key: str, value: Any) -> List[Dict[str, Any]]:
    """
    Filtrar lista de diccionarios por clave-valor.
    
    Args:
        items: Lista de diccionarios
        key: Clave a filtrar
        value: Valor a buscar
        
    Returns:
        Lista filtrada
    """
    return [item for item in items if item.get(key) == value]


def find_first(items: List[Any], predicate: Callable[[Any], bool]) -> Optional[Any]:
    """
    Encontrar primer item que cumpla condición.
    
    Args:
        items: Lista de items
        predicate: Función de condición
        
    Returns:
        Primer item que cumpla o None
    """
    for item in items:
        if predicate(item):
            return item
    return None


def group_by_key(items: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Agrupar items por valor de clave.
    
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


def map_values(items: List[Any], func: Callable[[Any], Any]) -> List[Any]:
    """
    Aplicar función a cada item.
    
    Args:
        items: Lista de items
        func: Función a aplicar
        
    Returns:
        Lista transformada
    """
    return [func(item) for item in items]


def reduce_items(items: List[Any], func: Callable[[Any, Any], Any], initial: Any = None) -> Any:
    """
    Reducir lista a un valor.
    
    Args:
        items: Lista de items
        func: Función de reducción
        initial: Valor inicial
        
    Returns:
        Valor reducido
    """
    if not items:
        return initial
    
    if initial is None:
        result = items[0]
        start = 1
    else:
        result = initial
        start = 0
    
    for item in items[start:]:
        result = func(result, item)
    
    return result


def partition(items: List[Any], predicate: Callable[[Any], bool]) -> Tuple[List[Any], List[Any]]:
    """
    Dividir lista en dos según condición.
    
    Args:
        items: Lista de items
        predicate: Función de condición
        
    Returns:
        Tupla con (items que cumplen, items que no cumplen)
    """
    true_items = []
    false_items = []
    
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    
    return (true_items, false_items)

