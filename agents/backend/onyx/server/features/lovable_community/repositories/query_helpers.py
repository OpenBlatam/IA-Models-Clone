"""
Query helper functions

Utility functions for building common SQLAlchemy queries.
"""

from typing import Optional, List, TypeVar
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc

T = TypeVar('T')


def apply_ordering(
    query: Query,
    order_by: Optional[str],
    order_direction: str = "desc",
    model_class: Optional[type] = None
) -> Query:
    """
    Apply ordering to a query.
    
    Args:
        query: SQLAlchemy query
        order_by: Field name to order by
        order_direction: Order direction (asc/desc)
        model_class: Model class to get field from (optional, inferred from query if not provided)
        
    Returns:
        Query with ordering applied
        
    Raises:
        ValueError: If order_by field doesn't exist on model
    """
    if not order_by:
        return query
    
    # Normalize order direction
    order_direction = order_direction.lower().strip()
    if order_direction not in ("asc", "desc"):
        order_direction = "desc"  # Default to desc if invalid
    
    # Get order field
    if model_class:
        order_field = getattr(model_class, order_by, None)
        if not order_field:
            # Field doesn't exist, return query without ordering
            return query
    else:
        try:
            entity = query.column_descriptions[0]['entity']
            order_field = getattr(entity, order_by, None)
            if not order_field:
                # Field doesn't exist, return query without ordering
                return query
        except (IndexError, KeyError, AttributeError):
            # Can't determine entity, return query without ordering
            return query
    
    # Apply ordering
    if order_direction == "desc":
        return query.order_by(desc(order_field))
    else:
        return query.order_by(asc(order_field))


def apply_pagination(
    query: Query,
    skip: int = 0,
    limit: int = 100
) -> Query:
    """
    Apply pagination (offset and limit) to a query.
    
    This helper encapsulates the common pattern of applying pagination
    that appears repeatedly across repository methods.
    
    Args:
        query: SQLAlchemy query
        skip: Number of records to skip (offset)
        limit: Maximum number of records to return
        
    Returns:
        Query with pagination applied
        
    Example:
        >>> query = apply_pagination(query, skip=10, limit=20)
    """
    if skip < 0:
        skip = 0
    
    if limit < 0:
        limit = 100  # Default limit
    
    return query.offset(skip).limit(limit)


def apply_ordering_and_pagination(
    query: Query,
    order_by: Optional[str] = None,
    order_direction: str = "desc",
    skip: int = 0,
    limit: int = 100,
    model_class: Optional[type] = None
) -> Query:
    """
    Apply ordering and pagination to a query in one call.
    
    This helper encapsulates the common pattern of applying both ordering
    and pagination that appears repeatedly across repository methods.
    
    Args:
        query: SQLAlchemy query
        order_by: Field name to order by (optional)
        order_direction: Order direction (asc/desc)
        skip: Number of records to skip
        limit: Maximum number of records to return
        model_class: Model class to get field from (optional)
        
    Returns:
        Query with ordering and pagination applied
        
    Example:
        >>> query = apply_ordering_and_pagination(
        >>>     query, "score", "desc", skip=0, limit=20, model_class=PublishedChat
        >>> )
    """
    query = apply_ordering(query, order_by, order_direction, model_class)
    query = apply_pagination(query, skip, limit)
    return query


def filter_public_chats(query: Query, model_class) -> Query:
    """
    Filter query to only include public chats.
    
    This helper encapsulates the common pattern of filtering by is_public == True
    that appears repeatedly across repository methods.
    
    Args:
        query: SQLAlchemy query
        model_class: Model class with is_public attribute
        
    Returns:
        Query filtered to public chats only
        
    Example:
        >>> query = filter_public_chats(query, PublishedChat)
    """
    if hasattr(model_class, 'is_public'):
        return query.filter(model_class.is_public == True)
    return query


def execute_query_with_pagination(
    query: Query,
    skip: int = 0,
    limit: int = 100,
    order_by: Optional[str] = None,
    order_direction: str = "desc",
    model_class: Optional[type] = None
) -> List[T]:
    """
    Execute a query with ordering and pagination, returning results.
    
    This helper encapsulates the common pattern of applying ordering, pagination,
    and executing the query that appears repeatedly across repository methods.
    
    Args:
        query: SQLAlchemy query
        skip: Number of records to skip
        limit: Maximum number of records to return
        order_by: Field name to order by (optional)
        order_direction: Order direction (asc/desc)
        model_class: Model class to get field from (optional)
        
    Returns:
        List of results
        
    Example:
        >>> chats = execute_query_with_pagination(
        >>>     query, skip=0, limit=20, order_by="score", model_class=PublishedChat
        >>> )
    """
    query = apply_ordering_and_pagination(
        query, order_by, order_direction, skip, limit, model_class
    )
    return query.all()






