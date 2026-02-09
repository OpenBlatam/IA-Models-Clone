"""
Helper functions for consolidating data from multiple sources.
Eliminates repetitive data aggregation patterns.
"""

from typing import TypeVar, List, Dict, Any, Optional, Callable, Iterable
from collections import defaultdict

T = TypeVar('T')


def consolidate_lists(
    *lists: Optional[List[T]],
    filter_none: bool = True
) -> List[T]:
    """
    Consolida múltiples listas en una sola.
    
    Args:
        *lists: Listas a consolidar (pueden ser None)
        filter_none: Si filtrar None de las listas (default: True)
        
    Returns:
        Lista consolidada
        
    Usage:
        >>> all_videos = consolidate_lists(
        ...     tiktok_profile.videos if tiktok_profile else None,
        ...     youtube_profile.videos if youtube_profile else None
        ... )
    """
    result = []
    for lst in lists:
        if lst:
            if filter_none:
                result.extend([item for item in lst if item is not None])
            else:
                result.extend(lst)
    return result


def extract_text_fields(
    items: Iterable[T],
    field_getters: List[Callable[[T], Optional[str]]],
    filter_empty: bool = True
) -> List[str]:
    """
    Extrae campos de texto de items usando funciones getter.
    
    Args:
        items: Iterable de items
        field_getters: Lista de funciones que extraen campos de texto
        filter_empty: Si filtrar strings vacíos (default: True)
        
    Returns:
        Lista de textos extraídos
        
    Usage:
        >>> texts = extract_text_fields(
        ...     videos,
        ...     [
        ...         lambda v: v.transcript,
        ...         lambda v: v.description
        ...     ]
        ... )
    """
    texts = []
    for item in items:
        for getter in field_getters:
            text = getter(item)
            if text and (not filter_empty or text.strip()):
                texts.append(text)
    return texts


def merge_content_dicts(
    *content_dicts: Optional[Dict[str, List[Any]]],
    merge_keys: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    """
    Combina múltiples diccionarios de contenido.
    
    Args:
        *content_dicts: Diccionarios a combinar
        merge_keys: Claves a combinar (default: todas)
        
    Returns:
        Diccionario combinado
        
    Usage:
        >>> all_content = merge_content_dicts(
        ...     tiktok_content,
        ...     instagram_content,
        ...     youtube_content
        ... )
    """
    result = defaultdict(list)
    
    for content_dict in content_dicts:
        if not content_dict:
            continue
        
        keys_to_merge = merge_keys or content_dict.keys()
        
        for key in keys_to_merge:
            if key in content_dict and content_dict[key]:
                result[key].extend(content_dict[key])
    
    return dict(result)


def aggregate_stats(
    items: Iterable[T],
    stat_getters: Dict[str, Callable[[T], Any]]
) -> Dict[str, Any]:
    """
    Agrega estadísticas de items usando funciones getter.
    
    Args:
        items: Iterable de items
        stat_getters: Diccionario de nombre_stat -> función_getter
        
    Returns:
        Diccionario con estadísticas agregadas
        
    Usage:
        >>> stats = aggregate_stats(
        ...     videos,
        ...     {
        ...         "total_views": lambda v: v.views or 0,
        ...         "total_likes": lambda v: v.likes or 0
        ...     }
        ... )
        {"total_views": 1000, "total_likes": 500}
    """
    stats = {name: 0 for name in stat_getters.keys()}
    
    for item in items:
        for stat_name, getter in stat_getters.items():
            value = getter(item)
            if value is not None:
                stats[stat_name] += value
    
    return stats


def collect_optional_fields(
    *sources: Optional[Any],
    field_getters: Dict[str, Callable[[Any], Optional[Any]]]
) -> Dict[str, Any]:
    """
    Recolecta campos opcionales de múltiples fuentes.
    
    Args:
        *sources: Fuentes de datos (pueden ser None)
        field_getters: Diccionario de nombre_campo -> función_getter
        
    Returns:
        Diccionario con campos recolectados
        
    Usage:
        >>> fields = collect_optional_fields(
        ...     tiktok_profile,
        ...     instagram_profile,
        ...     youtube_profile,
        ...     field_getters={
        ...         "username": lambda p: p.username if p else None,
        ...         "display_name": lambda p: p.display_name if p else None
        ...     }
        ... )
    """
    result = {}
    
    for field_name, getter in field_getters.items():
        for source in sources:
            if source:
                value = getter(source)
                if value is not None:
                    result[field_name] = value
                    break  # Usar primer valor no None encontrado
    
    return result








