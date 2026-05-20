"""
Optimizador de queries para base de datos (optimizado)

Incluye funciones para optimizar queries SQLAlchemy y mejorar rendimiento.
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Query
from sqlalchemy import func, desc, asc, and_, or_


def optimize_query(query: Query, use_eager_loading: bool = True) -> Query:
    """
    Optimiza una query SQLAlchemy (optimizado).
    
    Args:
        query: Query a optimizar
        use_eager_loading: Si usar eager loading para relaciones
        
    Returns:
        Query optimizada
    """
    # En el futuro, aquí se pueden agregar optimizaciones como:
    # - Eager loading de relaciones
    # - Select only needed columns
    # - Query hints
    return query


def add_pagination_to_query(
    query: Query,
    page: int,
    page_size: int
) -> Tuple[Query, int]:
    """
    Agrega paginación a una query y obtiene el total (optimizado).
    
    Args:
        query: Query base
        page: Página (1-indexed)
        page_size: Tamaño de página
        
    Returns:
        Tupla con (query paginada, total)
    """
    # Obtener total antes de paginar
    total = query.count()
    
    # Calcular offset
    offset = (page - 1) * page_size
    
    # Aplicar paginación
    paginated_query = query.offset(offset).limit(page_size)
    
    return paginated_query, total


def optimize_search_query(
    query: Query,
    search_term: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Query:
    """
    Optimiza una query de búsqueda (optimizado).
    
    Args:
        query: Query base
        search_term: Término de búsqueda (opcional)
        filters: Filtros adicionales (opcional)
        
    Returns:
        Query optimizada
    """
    if search_term:
        # Normalizar término de búsqueda
        search_term = search_term.strip().lower()
        
        # Agregar filtros de búsqueda (ejemplo para PublishedChat)
        # En producción, usar full-text search si está disponible
        query = query.filter(
            or_(
                func.lower(query.column_descriptions[0]['entity'].title).contains(search_term),
                func.lower(query.column_descriptions[0]['entity'].description).contains(search_term)
            )
        )
    
    if filters:
        for key, value in filters.items():
            if value is not None:
                # Aplicar filtros dinámicos
                if hasattr(query.column_descriptions[0]['entity'], key):
                    query = query.filter(getattr(query.column_descriptions[0]['entity'], key) == value)
    
    return query


def add_ordering_to_query(
    query: Query,
    sort_by: str,
    order: str = "desc",
    default_sort: Optional[str] = None
) -> Query:
    """
    Agrega ordenamiento a una query (optimizado).
    
    Args:
        query: Query base
        sort_by: Campo por el cual ordenar
        order: Orden (asc o desc)
        default_sort: Campo de ordenamiento por defecto (opcional)
        
    Returns:
        Query con ordenamiento
    """
    entity = query.column_descriptions[0]['entity']
    
    # Obtener campo de ordenamiento
    sort_field = getattr(entity, sort_by, None)
    
    if sort_field is None:
        # Usar campo por defecto si está disponible
        if default_sort:
            sort_field = getattr(entity, default_sort, None)
        if sort_field is None:
            return query
    
    # Aplicar ordenamiento
    if order.lower() == "desc":
        query = query.order_by(desc(sort_field))
    else:
        query = query.order_by(asc(sort_field))
    
    # Agregar ordenamiento secundario para consistencia
    if hasattr(entity, 'created_at') and sort_by != 'created_at':
        query = query.order_by(desc(entity.created_at))
    
    return query


def optimize_count_query(query: Query) -> Query:
    """
    Optimiza una query de conteo (optimizado).
    
    Args:
        query: Query a optimizar
        
    Returns:
        Query optimizada para conteo
    """
    # Para conteos, solo necesitamos una columna
    # Esto puede mejorar el rendimiento en algunas bases de datos
    return query.with_entities(func.count())


def get_query_stats(query: Query) -> Dict[str, Any]:
    """
    Obtiene estadísticas de una query (optimizado).
    
    Args:
        query: Query a analizar
        
    Returns:
        Diccionario con estadísticas
    """
    # En el futuro, aquí se puede agregar análisis de:
    # - Número de joins
    # - Complejidad de la query
    # - Índices utilizados
    return {
        "query": str(query.statement),
        "estimated_rows": None  # Se puede implementar con EXPLAIN
    }

