"""
Repository helper utilities for common operations.
"""

from typing import Dict, Any, List, Optional, Callable
from sqlalchemy.orm import Session, Query
from sqlalchemy import desc, asc, func
import logging

logger = logging.getLogger(__name__)


def build_query_filters(
    query: Query,
    model_class: Any,
    filters: Dict[str, Any],
    filter_mapping: Optional[Dict[str, Callable]] = None
) -> Query:
    """
    Build query filters dynamically.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class
        filters: Dictionary of filter conditions
        filter_mapping: Optional mapping of filter keys to filter functions
        
    Returns:
        Query with filters applied
    """
    if not filters:
        return query
    
    if filter_mapping is None:
        filter_mapping = {}
    
    for key, value in filters.items():
        if value is None:
            continue
        
        # Use custom filter function if provided
        if key in filter_mapping:
            query = filter_mapping[key](query, value)
            continue
        
        # Default: try to get attribute directly
        if hasattr(model_class, key):
            attr = getattr(model_class, key)
            if isinstance(value, (list, tuple)):
                query = query.filter(attr.in_(value))
            elif isinstance(value, dict):
                # Support for operators like {'gte': 10, 'lte': 20}
                if 'gte' in value:
                    query = query.filter(attr >= value['gte'])
                if 'lte' in value:
                    query = query.filter(attr <= value['lte'])
                if 'gt' in value:
                    query = query.filter(attr > value['gt'])
                if 'lt' in value:
                    query = query.filter(attr < value['lt'])
                if 'eq' in value:
                    query = query.filter(attr == value['eq'])
                if 'ne' in value:
                    query = query.filter(attr != value['ne'])
                if 'like' in value:
                    query = query.filter(attr.like(f"%{value['like']}%"))
                if 'ilike' in value:
                    query = query.filter(attr.ilike(f"%{value['ilike']}%"))
            else:
                query = query.filter(attr == value)
    
    return query


def apply_date_range_filter(
    query: Query,
    model_class: Any,
    date_field: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Query:
    """
    Apply date range filter to query.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class
        date_field: Name of the date field
        start_date: Start date (ISO format string)
        end_date: End date (ISO format string)
        
    Returns:
        Query with date range filter applied
    """
    from datetime import datetime
    
    if hasattr(model_class, date_field):
        attr = getattr(model_class, date_field)
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.filter(attr >= start_dt)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.filter(attr <= end_dt)
            except (ValueError, AttributeError):
                logger.warning(f"Invalid end_date format: {end_date}")
    
    return query


def apply_text_search_filter(
    query: Query,
    model_class: Any,
    search_fields: List[str],
    search_term: str,
    case_sensitive: bool = False
) -> Query:
    """
    Apply text search filter across multiple fields.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class
        search_fields: List of field names to search
        search_term: Search term
        case_sensitive: Whether search is case sensitive
        
    Returns:
        Query with text search filter applied
    """
    if not search_term or not search_fields:
        return query
    
    from sqlalchemy import or_
    
    conditions = []
    for field in search_fields:
        if hasattr(model_class, field):
            attr = getattr(model_class, field)
            if case_sensitive:
                conditions.append(attr.like(f"%{search_term}%"))
            else:
                conditions.append(attr.ilike(f"%{search_term}%"))
    
    if conditions:
        query = query.filter(or_(*conditions))
    
    return query


def get_aggregate_stats(
    query: Query,
    model_class: Any,
    group_by_field: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get aggregate statistics from query.
    
    Args:
        query: SQLAlchemy query object
        model_class: Model class
        group_by_field: Optional field to group by
        
    Returns:
        Dictionary with aggregate statistics
    """
    stats = {
        'count': query.count()
    }
    
    if group_by_field and hasattr(model_class, group_by_field):
        attr = getattr(model_class, group_by_field)
        grouped = query.with_entities(
            attr,
            func.count(model_class.id).label('count')
        ).group_by(attr).all()
        
        stats['by_' + group_by_field] = {
            str(key): count for key, count in grouped
        }
    
    return stats


def batch_update(
    db: Session,
    model_class: Any,
    updates: List[Dict[str, Any]],
    id_field: str = 'id'
) -> int:
    """
    Batch update multiple entities.
    
    Args:
        db: Database session
        model_class: Model class
        updates: List of update dictionaries with id and fields to update
        id_field: Name of the ID field
        
    Returns:
        Number of updated entities
    """
    updated_count = 0
    
    for update_data in updates:
        entity_id = update_data.pop(id_field, None)
        if not entity_id:
            continue
        
        entity = db.query(model_class).filter(
            getattr(model_class, id_field) == entity_id
        ).first()
        
        if entity:
            for key, value in update_data.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            updated_count += 1
    
    if updated_count > 0:
        db.commit()
    
    return updated_count


def batch_delete(
    db: Session,
    model_class: Any,
    ids: List[str],
    id_field: str = 'id'
) -> int:
    """
    Batch delete multiple entities.
    
    Args:
        db: Database session
        model_class: Model class
        ids: List of entity IDs to delete
        id_field: Name of the ID field
        
    Returns:
        Number of deleted entities
    """
    if not ids:
        return 0
    
    deleted_count = db.query(model_class).filter(
        getattr(model_class, id_field).in_(ids)
    ).delete(synchronize_session=False)
    
    db.commit()
    return deleted_count






