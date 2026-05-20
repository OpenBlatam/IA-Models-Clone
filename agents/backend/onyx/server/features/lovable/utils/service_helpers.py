"""
Helper utilities for services.
"""

from typing import Dict, Any, List, Optional, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import logging

logger = logging.getLogger(__name__)


def build_filter_conditions(
    filters: Dict[str, Any],
    model_class: type
) -> List:
    """
    Build SQLAlchemy filter conditions from a dictionary.
    
    Args:
        filters: Dictionary of filter conditions
        model_class: SQLAlchemy model class
        
    Returns:
        List of filter conditions
    """
    conditions = []
    
    for key, value in filters.items():
        if value is None:
            continue
        
        try:
            attr = getattr(model_class, key)
            
            # Handle different filter types
            if isinstance(value, list):
                conditions.append(attr.in_(value))
            elif isinstance(value, dict):
                # Support for operators like {"gt": 10}, {"lt": 20}, etc.
                for op, op_value in value.items():
                    if op == "gt":
                        conditions.append(attr > op_value)
                    elif op == "gte":
                        conditions.append(attr >= op_value)
                    elif op == "lt":
                        conditions.append(attr < op_value)
                    elif op == "lte":
                        conditions.append(attr <= op_value)
                    elif op == "ne":
                        conditions.append(attr != op_value)
                    elif op == "like":
                        conditions.append(attr.like(f"%{op_value}%"))
                    elif op == "ilike":
                        conditions.append(attr.ilike(f"%{op_value}%"))
            else:
                conditions.append(attr == value)
        except AttributeError:
            logger.warning(f"Attribute {key} not found in {model_class.__name__}")
    
    return conditions


def apply_common_filters(
    query,
    filters: Dict[str, Any],
    model_class: type
):
    """
    Apply common filters to a query.
    
    Args:
        query: SQLAlchemy query
        filters: Dictionary of filter conditions
        model_class: SQLAlchemy model class
        
    Returns:
        Query with filters applied
    """
    conditions = build_filter_conditions(filters, model_class)
    
    if conditions:
        query = query.filter(and_(*conditions))
    
    return query


def safe_get_or_none(
    query_func: Callable,
    *args,
    **kwargs
) -> Optional[Any]:
    """
    Safely execute a query function and return None on error.
    
    Args:
        query_func: Query function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Query result or None if error
    """
    try:
        return query_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        return None


def batch_process(
    items: List[Any],
    batch_size: int,
    process_func: Callable
) -> Dict[str, Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "processed": 0,
        "failed": 0,
        "errors": []
    }
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            process_func(batch)
            results["processed"] += len(batch)
        except Exception as e:
            results["failed"] += len(batch)
            results["errors"].append(str(e))
            logger.error(f"Batch processing failed: {e}")
    
    return results


def build_filter_dict(
    category: Optional[str] = None,
    user_id: Optional[str] = None,
    featured: Optional[bool] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build filter dictionary from common parameters.
    
    Args:
        category: Optional category filter
        user_id: Optional user ID filter
        featured: Optional featured filter
        **kwargs: Additional filter parameters
        
    Returns:
        Dictionary of filter conditions
    """
    filters = {}
    
    if category is not None:
        filters["category"] = category
    if user_id is not None:
        filters["user_id"] = user_id
    if featured is not None:
        filters["is_featured"] = featured
    
    # Add any additional kwargs
    filters.update(kwargs)
    
    return filters


def safe_service_call(
    service_func: Callable,
    *args,
    **kwargs
) -> Optional[Any]:
    """
    Safely call a service function with error handling.
    
    Args:
        service_func: Service function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Service result or None if error
    """
    try:
        return service_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Service call failed: {e}", exc_info=True)
        return None






