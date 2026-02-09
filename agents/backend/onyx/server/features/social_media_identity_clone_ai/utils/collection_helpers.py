"""
Helper functions for collection operations (lists, dicts).
Eliminates repetitive list processing and filtering patterns.
"""

from typing import TypeVar, List, Callable, Optional, Any, Iterable, Dict
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


def safe_map(
    items: Iterable[T],
    func: Callable[[T], R],
    default: Optional[R] = None,
    skip_errors: bool = True,
    operation: Optional[str] = None
) -> List[R]:
    """
    Aplica una función a cada item de forma segura, saltando errores.
    
    Args:
        items: Iterable de items a procesar
        func: Función a aplicar a cada item
        default: Valor por defecto si hay error (opcional)
        skip_errors: Si saltar items con error (default: True)
        operation: Nombre de la operación para logging
        
    Returns:
        Lista de resultados (sin items que fallaron si skip_errors=True)
        
    Usage:
        >>> videos = safe_map(
        ...     video_list,
        ...     lambda v: VideoContent(**v),
        ...     operation="process_videos"
        ... )
    """
    results = []
    op_name = operation or func.__name__
    
    for item in items:
        try:
            result = func(item)
            if result is not None:
                results.append(result)
        except Exception as e:
            if skip_errors:
                logger.warning(f"Error in {op_name} for item: {e}")
                if default is not None:
                    results.append(default)
            else:
                raise
    
    return results


def filter_map(
    items: Iterable[T],
    func: Callable[[T], Optional[R]],
    predicate: Optional[Callable[[T], bool]] = None
) -> List[R]:
    """
    Filtra y transforma items en una sola operación.
    
    Args:
        items: Iterable de items
        func: Función de transformación (retorna None para filtrar)
        predicate: Función de filtrado adicional (opcional)
        
    Returns:
        Lista de items transformados y filtrados
        
    Usage:
        >>> valid_videos = filter_map(
        ...     video_list,
        ...     lambda v: VideoContent(**v) if v.get("video_id") else None
        ... )
    """
    results = []
    for item in items:
        if predicate and not predicate(item):
            continue
        result = func(item)
        if result is not None:
            results.append(result)
    return results


def group_by(
    items: Iterable[T],
    key_func: Callable[[T], Any]
) -> Dict[Any, List[T]]:
    """
    Agrupa items por una clave.
    
    Args:
        items: Iterable de items
        key_func: Función que retorna la clave de agrupación
        
    Returns:
        Diccionario agrupado por clave
        
    Usage:
        >>> grouped = group_by(videos, lambda v: v.platform)
        {'tiktok': [...], 'instagram': [...]}
    """
    grouped = {}
    for item in items:
        key = key_func(item)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    return grouped


def chunk_list(
    items: List[T],
    chunk_size: int
) -> List[List[T]]:
    """
    Divide una lista en chunks de tamaño fijo.
    
    Args:
        items: Lista a dividir
        chunk_size: Tamaño de cada chunk
        
    Returns:
        Lista de chunks
        
    Usage:
        >>> chunks = chunk_list(items, chunk_size=100)
        [[...], [...], [...]]
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten(
    nested_list: List[List[T]]
) -> List[T]:
    """
    Aplana una lista de listas.
    
    Args:
        nested_list: Lista de listas
        
    Returns:
        Lista aplanada
        
    Usage:
        >>> flatten([[1, 2], [3, 4]])
        [1, 2, 3, 4]
    """
    return [item for sublist in nested_list for item in sublist]


def unique(
    items: Iterable[T],
    key_func: Optional[Callable[[T], Any]] = None
) -> List[T]:
    """
    Retorna items únicos preservando orden.
    
    Args:
        items: Iterable de items
        key_func: Función para determinar unicidad (opcional, usa item directamente)
        
    Returns:
        Lista de items únicos
        
    Usage:
        >>> unique([1, 2, 2, 3])
        [1, 2, 3]
        
        >>> unique(users, key_func=lambda u: u.id)
        [user1, user2, ...]  # sin duplicados por ID
    """
    seen = set()
    result = []
    
    for item in items:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result


def partition(
    items: Iterable[T],
    predicate: Callable[[T], bool]
) -> tuple[List[T], List[T]]:
    """
    Divide items en dos grupos según un predicado.
    
    Args:
        items: Iterable de items
        predicate: Función que retorna True/False
        
    Returns:
        Tupla (items_que_cumplen, items_que_no_cumplen)
        
    Usage:
        >>> valid, invalid = partition(videos, lambda v: v.video_id is not None)
    """
    true_items = []
    false_items = []
    
    for item in items:
        if predicate(item):
            true_items.append(item)
        else:
            false_items.append(item)
    
    return true_items, false_items








