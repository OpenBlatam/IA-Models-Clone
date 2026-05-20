"""
Search Utils - Utilidades de Búsqueda y Filtrado
================================================

Utilidades avanzadas para búsqueda y filtrado de datos.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def search_in_list(
    items: List[Any],
    query: str,
    search_fields: Optional[List[str]] = None,
    case_sensitive: bool = False
) -> List[Any]:
    """
    Buscar en lista de items.
    
    Args:
        items: Lista de items
        query: Query de búsqueda
        search_fields: Campos a buscar (si items son dicts)
        case_sensitive: Si es case sensitive
        
    Returns:
        Lista de items que coinciden
    """
    if not query:
        return items
    
    query_lower = query if case_sensitive else query.lower()
    results = []
    
    for item in items:
        if isinstance(item, dict):
            # Buscar en campos específicos
            if search_fields:
                for field in search_fields:
                    if field in item:
                        value = str(item[field])
                        value_lower = value if case_sensitive else value.lower()
                        if query_lower in value_lower:
                            results.append(item)
                            break
            else:
                # Buscar en todos los valores
                for value in item.values():
                    value_str = str(value)
                    value_lower = value_str if case_sensitive else value_str.lower()
                    if query_lower in value_lower:
                        results.append(item)
                        break
        else:
            # Buscar en string representation
            item_str = str(item)
            item_lower = item_str if case_sensitive else item_str.lower()
            if query_lower in item_lower:
                results.append(item)
    
    return results


def filter_by_predicate(
    items: List[Any],
    predicate: Callable[[Any], bool]
) -> List[Any]:
    """
    Filtrar items por predicado.
    
    Args:
        items: Lista de items
        predicate: Función predicado
        
    Returns:
        Lista filtrada
    """
    return [item for item in items if predicate(item)]


def filter_by_field(
    items: List[Dict[str, Any]],
    field: str,
    value: Any,
    operator: str = "eq"
) -> List[Dict[str, Any]]:
    """
    Filtrar items por campo.
    
    Args:
        items: Lista de diccionarios
        field: Campo a filtrar
        value: Valor a comparar
        operator: Operador (eq, ne, gt, lt, gte, lte, in, contains)
        
    Returns:
        Lista filtrada
    """
    results = []
    
    for item in items:
        if field not in item:
            continue
        
        field_value = item[field]
        
        match = False
        
        if operator == "eq":
            match = field_value == value
        elif operator == "ne":
            match = field_value != value
        elif operator == "gt":
            match = field_value > value
        elif operator == "lt":
            match = field_value < value
        elif operator == "gte":
            match = field_value >= value
        elif operator == "lte":
            match = field_value <= value
        elif operator == "in":
            match = field_value in value if isinstance(value, (list, set, tuple)) else False
        elif operator == "contains":
            if isinstance(field_value, str) and isinstance(value, str):
                match = value in field_value
            elif isinstance(field_value, (list, tuple)):
                match = value in field_value
        
        if match:
            results.append(item)
    
    return results


def filter_by_date_range(
    items: List[Dict[str, Any]],
    field: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Filtrar items por rango de fechas.
    
    Args:
        items: Lista de diccionarios
        field: Campo de fecha
        start_date: Fecha inicio
        end_date: Fecha fin
        
    Returns:
        Lista filtrada
    """
    results = []
    
    for item in items:
        if field not in item:
            continue
        
        field_value = item[field]
        
        # Convertir a datetime si es necesario
        if isinstance(field_value, str):
            try:
                field_value = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue
        elif not isinstance(field_value, datetime):
            continue
        
        # Verificar rango
        if start_date and field_value < start_date:
            continue
        
        if end_date and field_value > end_date:
            continue
        
        results.append(item)
    
    return results


def sort_by_field(
    items: List[Dict[str, Any]],
    field: str,
    reverse: bool = False
) -> List[Dict[str, Any]]:
    """
    Ordenar items por campo.
    
    Args:
        items: Lista de diccionarios
        field: Campo a ordenar
        reverse: Si ordenar en reversa
        
    Returns:
        Lista ordenada
    """
    return sorted(
        items,
        key=lambda x: x.get(field, ""),
        reverse=reverse
    )


def paginate(
    items: List[Any],
    page: int = 1,
    per_page: int = 10
) -> Dict[str, Any]:
    """
    Paginar lista de items.
    
    Args:
        items: Lista de items
        page: Número de página (1-indexed)
        per_page: Items por página
        
    Returns:
        Diccionario con items, total, página, etc.
    """
    total = len(items)
    total_pages = (total + per_page - 1) // per_page
    page = max(1, min(page, total_pages))
    
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "items": items[start:end],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }


def fuzzy_search(
    items: List[Any],
    query: str,
    threshold: float = 0.6
) -> List[tuple[Any, float]]:
    """
    Búsqueda difusa (fuzzy search).
    
    Args:
        items: Lista de items
        query: Query de búsqueda
        threshold: Umbral de similitud (0-1)
        
    Returns:
        Lista de tuplas (item, score)
    """
    try:
        from difflib import SequenceMatcher
        
        results = []
        query_lower = query.lower()
        
        for item in items:
            item_str = str(item).lower()
            similarity = SequenceMatcher(None, query_lower, item_str).ratio()
            
            if similarity >= threshold:
                results.append((item, similarity))
        
        # Ordenar por score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
        
    except ImportError:
        logger.warning("difflib not available, using simple search")
        # Fallback a búsqueda simple
        return [(item, 1.0) for item in search_in_list(items, query)]


def regex_search(
    items: List[Any],
    pattern: str,
    flags: int = 0
) -> List[Any]:
    """
    Búsqueda con regex.
    
    Args:
        items: Lista de items
        pattern: Patrón regex
        flags: Flags de regex
        
    Returns:
        Lista de items que coinciden
    """
    try:
        regex = re.compile(pattern, flags)
        results = []
        
        for item in items:
            item_str = str(item)
            if regex.search(item_str):
                results.append(item)
        
        return results
    except re.error as e:
        logger.error(f"Invalid regex pattern: {e}")
        return []




