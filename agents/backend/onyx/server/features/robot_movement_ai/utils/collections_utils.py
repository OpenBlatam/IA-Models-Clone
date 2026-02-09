"""
Collections Utilities - Utilidades de colecciones y estructuras de datos
========================================================================

Utilidades avanzadas para trabajar con listas, diccionarios, sets y otras colecciones.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Callable, Iterator
from collections import defaultdict, Counter, OrderedDict
from functools import reduce
import operator

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def group_by(items: List[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    """
    Agrupar items por una función clave.
    
    Args:
        items: Lista de items
        key_func: Función que extrae la clave
    
    Returns:
        Diccionario agrupado
    """
    grouped = defaultdict(list)
    for item in items:
        key = key_func(item)
        grouped[key].append(item)
    return dict(grouped)


def partition(items: List[T], predicate: Callable[[T], bool]) -> Tuple[List[T], List[T]]:
    """
    Dividir lista en dos según un predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
    
    Returns:
        Tupla (items que cumplen, items que no cumplen)
    """
    true_items = []
    false_items = []
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    return (true_items, false_items)


def unique(items: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """
    Obtener items únicos preservando orden.
    
    Args:
        items: Lista de items
        key: Función opcional para extraer clave de unicidad
    
    Returns:
        Lista con items únicos
    """
    seen = set()
    result = []
    for item in items:
        if key:
            item_key = key(item)
        else:
            item_key = item
        
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    return result


def sort_by(items: List[T], key_func: Callable[[T], Any], reverse: bool = False) -> List[T]:
    """
    Ordenar lista por función clave.
    
    Args:
        items: Lista de items
        key_func: Función que extrae la clave de ordenamiento
        reverse: Si True, orden descendente
    
    Returns:
        Lista ordenada
    """
    return sorted(items, key=key_func, reverse=reverse)


def find(items: List[T], predicate: Callable[[T], bool]) -> Optional[T]:
    """
    Encontrar primer item que cumple predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
    
    Returns:
        Primer item encontrado o None
    """
    for item in items:
        if predicate(item):
            return item
    return None


def find_all(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """
    Encontrar todos los items que cumplen predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
    
    Returns:
        Lista de items encontrados
    """
    return [item for item in items if predicate(item)]


def count_by(items: List[T], key_func: Callable[[T], K]) -> Dict[K, int]:
    """
    Contar items por función clave.
    
    Args:
        items: Lista de items
        key_func: Función que extrae la clave
    
    Returns:
        Diccionario con conteos
    """
    counts = Counter()
    for item in items:
        key = key_func(item)
        counts[key] += 1
    return dict(counts)


def map_dict(d: Dict[K, V], func: Callable[[K, V], Tuple[K, V]]) -> Dict[K, V]:
    """
    Aplicar función a cada par clave-valor de diccionario.
    
    Args:
        d: Diccionario
        func: Función (key, value) -> (new_key, new_value)
    
    Returns:
        Nuevo diccionario transformado
    """
    return {k: v for k, v in (func(key, val) for key, val in d.items())}


def filter_dict(d: Dict[K, V], predicate: Callable[[K, V], bool]) -> Dict[K, V]:
    """
    Filtrar diccionario por predicado.
    
    Args:
        d: Diccionario
        predicate: Función (key, value) -> bool
    
    Returns:
        Diccionario filtrado
    """
    return {k: v for k, v in d.items() if predicate(k, v)}


def invert_dict(d: Dict[K, V]) -> Dict[V, List[K]]:
    """
    Invertir diccionario (valores -> claves).
    
    Args:
        d: Diccionario
    
    Returns:
        Diccionario invertido con listas de claves
    """
    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return dict(inverted)


def deep_merge(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionar diccionarios de forma profunda (recursiva).
    
    Args:
        *dicts: Diccionarios a fusionar
    
    Returns:
        Diccionario fusionado
    """
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def dict_to_list(d: Dict[K, V], key_name: str = 'key', value_name: str = 'value') -> List[Dict[str, Any]]:
    """
    Convertir diccionario a lista de diccionarios.
    
    Args:
        d: Diccionario
        key_name: Nombre para la clave en el resultado
        value_name: Nombre para el valor en el resultado
    
    Returns:
        Lista de diccionarios
    """
    return [{key_name: k, value_name: v} for k, v in d.items()]


def list_to_dict(items: List[Dict[str, Any]], key_field: str, value_field: Optional[str] = None) -> Dict[str, Any]:
    """
    Convertir lista de diccionarios a diccionario.
    
    Args:
        items: Lista de diccionarios
        key_field: Campo a usar como clave
        value_field: Campo a usar como valor (None = todo el dict)
    
    Returns:
        Diccionario
    """
    result = {}
    for item in items:
        key = item[key_field]
        value = item[value_field] if value_field else item
        result[key] = value
    return result


def chunk_by(items: List[T], size: int) -> Iterator[List[T]]:
    """
    Dividir lista en chunks (generador).
    
    Args:
        items: Lista de items
        size: Tamaño del chunk
    
    Yields:
        Chunks de la lista
    """
    for i in range(0, len(items), size):
        yield items[i:i + size]


def zip_dicts(*dicts: Dict[K, V]) -> Iterator[Tuple[K, Tuple[V, ...]]]:
    """
    Hacer zip de múltiples diccionarios.
    
    Args:
        *dicts: Diccionarios
    
    Yields:
        Tuplas (key, (value1, value2, ...))
    """
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    for key in all_keys:
        values = tuple(d.get(key) for d in dicts)
        yield (key, values)


def reduce_dict(d: Dict[K, V], func: Callable[[V, V], V], initial: Optional[V] = None) -> V:
    """
    Reducir diccionario aplicando función a valores.
    
    Args:
        d: Diccionario
        func: Función de reducción
        initial: Valor inicial (opcional)
    
    Returns:
        Valor reducido
    """
    values = list(d.values())
    if initial is not None:
        return reduce(func, values, initial)
    return reduce(func, values)


def dict_diff(d1: Dict[K, V], d2: Dict[K, V]) -> Dict[str, Any]:
    """
    Calcular diferencia entre dos diccionarios.
    
    Args:
        d1: Primer diccionario
        d2: Segundo diccionario
    
    Returns:
        Diccionario con added, removed, changed
    """
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())
    
    added = {k: d2[k] for k in keys2 - keys1}
    removed = {k: d1[k] for k in keys1 - keys2}
    changed = {k: (d1[k], d2[k]) for k in keys1 & keys2 if d1[k] != d2[k]}
    
    return {
        'added': added,
        'removed': removed,
        'changed': changed
    }


def flatten_dict(d: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """
    Aplanar diccionario anidado.
    
    Args:
        d: Diccionario anidado
        separator: Separador para keys
        prefix: Prefijo para keys
    
    Returns:
        Diccionario aplanado
    """
    result = {}
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, separator, new_key))
        else:
            result[new_key] = value
    return result


def unflatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Desaplanar diccionario a estructura anidada.
    
    Args:
        d: Diccionario aplanado
        separator: Separador en keys
    
    Returns:
        Diccionario anidado
    """
    result = {}
    for key, value in d.items():
        keys = key.split(separator)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def set_operations(set1: Set[T], set2: Set[T]) -> Dict[str, Set[T]]:
    """
    Calcular todas las operaciones de sets.
    
    Args:
        set1: Primer set
        set2: Segundo set
    
    Returns:
        Diccionario con union, intersection, difference, symmetric_difference
    """
    return {
        'union': set1 | set2,
        'intersection': set1 & set2,
        'difference': set1 - set2,
        'symmetric_difference': set1 ^ set2
    }


def batch_map(items: List[T], func: Callable[[T], V], batch_size: int = 100) -> List[V]:
    """
    Aplicar función a items en batches.
    
    Args:
        items: Lista de items
        func: Función a aplicar
        batch_size: Tamaño del batch
    
    Returns:
        Lista de resultados
    """
    results = []
    for batch in chunk_by(items, batch_size):
        results.extend([func(item) for item in batch])
    return results


def take(items: List[T], n: int) -> List[T]:
    """
    Tomar primeros n items.
    
    Args:
        items: Lista de items
        n: Número de items
    
    Returns:
        Primeros n items
    """
    return items[:n]


def drop(items: List[T], n: int) -> List[T]:
    """
    Omitir primeros n items.
    
    Args:
        items: Lista de items
        n: Número de items a omitir
    
    Returns:
        Items restantes
    """
    return items[n:]


def take_while(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """
    Tomar items mientras se cumple predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
    
    Returns:
        Items tomados
    """
    result = []
    for item in items:
        if not predicate(item):
            break
        result.append(item)
    return result


def drop_while(items: List[T], predicate: Callable[[T], bool]) -> List[T]:
    """
    Omitir items mientras se cumple predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
    
    Returns:
        Items restantes
    """
    result = []
    dropping = True
    for item in items:
        if dropping and predicate(item):
            continue
        dropping = False
        result.append(item)
    return result

