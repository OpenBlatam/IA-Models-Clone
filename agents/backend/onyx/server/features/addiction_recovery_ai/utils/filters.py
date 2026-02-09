"""
Filtering utilities for API queries
"""

from typing import List, Dict, Any, Callable, Optional, TypeVar
from datetime import datetime

T = TypeVar('T')


def filter_by_field(
    items: List[Dict[str, Any]],
    field: str,
    value: Any,
    exact_match: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter items by field value
    
    Args:
        items: List of items to filter
        field: Field name to filter by
        value: Value to match
        exact_match: Whether to use exact match (default) or contains
    
    Returns:
        Filtered list of items
    """
    if exact_match:
        return [item for item in items if item.get(field) == value]
    
    value_lower = str(value).lower()
    return [
        item for item in items
        if value_lower in str(item.get(field, "")).lower()
    ]


def filter_by_date_range(
    items: List[Dict[str, Any]],
    date_field: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Filter items by date range
    
    Args:
        items: List of items to filter
        date_field: Field name containing date
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
    
    Returns:
        Filtered list of items
    """
    filtered = items
    
    if start_date:
        filtered = [
            item for item in filtered
            if _parse_date(item.get(date_field)) >= start_date
        ]
    
    if end_date:
        filtered = [
            item for item in filtered
            if _parse_date(item.get(date_field)) <= end_date
        ]
    
    return filtered


def filter_by_custom_predicate(
    items: List[T],
    predicate: Callable[[T], bool]
) -> List[T]:
    """
    Filter items using custom predicate function
    
    Args:
        items: List of items to filter
        predicate: Function that returns True for items to include
    
    Returns:
        Filtered list of items
    """
    return [item for item in items if predicate(item)]


def sort_items(
    items: List[Dict[str, Any]],
    sort_by: str,
    reverse: bool = False
) -> List[Dict[str, Any]]:
    """
    Sort items by field
    
    Args:
        items: List of items to sort
        sort_by: Field name to sort by
        reverse: Whether to sort in reverse order
    
    Returns:
        Sorted list of items
    """
    return sorted(
        items,
        key=lambda x: x.get(sort_by),
        reverse=reverse
    )


def _parse_date(date_value: Any) -> Optional[datetime]:
    """Parse date value to datetime"""
    if date_value is None:
        return None
    
    if isinstance(date_value, datetime):
        return date_value
    
    if isinstance(date_value, str):
        try:
            return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        except ValueError:
            return None
    
    return None

