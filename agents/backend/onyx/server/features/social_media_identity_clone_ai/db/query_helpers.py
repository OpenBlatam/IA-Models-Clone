"""
Helper functions for building database queries.
Eliminates repetitive query patterns.
"""

from typing import TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

T = TypeVar('T')


def query_one(
    db: Session,
    model_class: Type[T],
    filters: Dict[str, Any],
    order_by: Optional[str] = None
) -> Optional[T]:
    """
    Ejecuta una query que retorna un solo resultado.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo
        filters: Filtros para filter_by
        order_by: Campo para ordenar (opcional)
        
    Returns:
        Modelo o None
        
    Usage:
        identity = query_one(db, IdentityProfileModel, {"id": identity_id})
    """
    query = db.query(model_class).filter_by(**filters)
    
    if order_by:
        query = query.order_by(desc(getattr(model_class, order_by)))
    
    return query.first()


def query_many(
    db: Session,
    model_class: Type[T],
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[T]:
    """
    Ejecuta una query que retorna múltiples resultados.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo
        filters: Filtros para filter_by (opcional)
        order_by: Campo para ordenar (opcional)
        limit: Límite de resultados (opcional)
        offset: Offset para paginación (opcional)
        
    Returns:
        Lista de modelos
        
    Usage:
        contents = query_many(
            db,
            GeneratedContentModel,
            filters={"identity_profile_id": identity_id},
            order_by="generated_at",
            limit=10
        )
    """
    query = db.query(model_class)
    
    if filters:
        query = query.filter_by(**filters)
    
    if order_by:
        query = query.order_by(desc(getattr(model_class, order_by)))
    
    if offset:
        query = query.offset(offset)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()








