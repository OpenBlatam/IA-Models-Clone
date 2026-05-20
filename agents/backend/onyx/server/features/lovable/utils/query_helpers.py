"""
Query helper utilities for database operations.
"""

from typing import List, Tuple, Optional, Callable
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc
import logging

logger = logging.getLogger(__name__)


def apply_pagination(
    query: Query,
    page: int = 1,
    page_size: int = 20
) -> Tuple[Query, int]:
    """
    Apply pagination to a query.
    
    Args:
        query: SQLAlchemy query
        page: Page number (1-indexed)
        page_size: Items per page
        
    Returns:
        Tuple of (paginated query, total count)
    """
    total = query.count()
    paginated_query = query.offset((page - 1) * page_size).limit(page_size)
    
    return paginated_query, total


def apply_sorting(
    query: Query,
    sort_by: str,
    order: str = "desc",
    sort_mapping: Optional[dict] = None
) -> Query:
    """
    Apply sorting to a query.
    
    Args:
        query: SQLAlchemy query
        sort_by: Field to sort by
        order: Sort order ('asc' or 'desc')
        sort_mapping: Optional mapping of sort_by values to model attributes
        
    Returns:
        Query with sorting applied
    """
    if sort_mapping and sort_by in sort_mapping:
        sort_field = sort_mapping[sort_by]
    else:
        # Try to get attribute directly
        try:
            sort_field = getattr(query.column_descriptions[0]['entity'], sort_by)
        except (AttributeError, IndexError, KeyError):
            logger.warning(f"Could not find sort field: {sort_by}, using default")
            return query
    
    if order.lower() == "asc":
        return query.order_by(asc(sort_field))
    else:
        return query.order_by(desc(sort_field))


def apply_filters(
    query: Query,
    filters: dict
) -> Query:
    """
    Apply filters to a query.
    
    Args:
        query: SQLAlchemy query
        filters: Dictionary of filter conditions
        
    Returns:
        Query with filters applied
    """
    for key, value in filters.items():
        if value is not None:
            try:
                # Try to get attribute
                attr = getattr(query.column_descriptions[0]['entity'], key)
                query = query.filter(attr == value)
            except (AttributeError, IndexError, KeyError):
                logger.warning(f"Could not apply filter {key}={value}")
    
    return query


def safe_query_execute(
    query: Query,
    error_message: str = "Query execution failed"
) -> List:
    """
    Safely execute a query with error handling.
    
    Args:
        query: SQLAlchemy query
        error_message: Error message if execution fails
        
    Returns:
        List of results
        
    Raises:
        Exception: If query execution fails
    """
    try:
        return query.all()
    except Exception as e:
        logger.error(f"{error_message}: {e}", exc_info=True)
        raise


def build_filter_conditions(**kwargs) -> dict:
    """
    Build filter conditions dictionary from keyword arguments.
    Only includes arguments that are not None.
    
    Args:
        **kwargs: Keyword arguments to filter by
        
    Returns:
        Dictionary of filter conditions
    """
    return {k: v for k, v in kwargs.items() if v is not None}






