"""
Sorting validation functions

Functions for validating sort fields, order, and periods.
"""

from typing import List


def validate_sort_by(sort_by: str, allowed_fields: List[str]) -> str:
    """
    Validates a sort field.
    
    Args:
        sort_by: Sort field to validate
        allowed_fields: List of allowed fields
        
    Returns:
        Validated sort field
        
    Raises:
        ValueError: If the sort field is invalid
    """
    if not sort_by or not isinstance(sort_by, str):
        raise ValueError("Sort by field is required and must be a string")
    
    sort_by = sort_by.strip().lower()
    
    if sort_by not in allowed_fields:
        raise ValueError(f"Sort by field must be one of: {', '.join(allowed_fields)}")
    
    return sort_by


def validate_order(order: str) -> str:
    """
    Validates an order (asc/desc).
    
    Args:
        order: Order to validate
        
    Returns:
        Validated order
        
    Raises:
        ValueError: If the order is invalid
    """
    if not order or not isinstance(order, str):
        raise ValueError("Order is required and must be a string")
    
    order = order.strip().lower()
    
    if order not in ("asc", "desc"):
        raise ValueError("Order must be 'asc' or 'desc'")
    
    return order


def validate_period(period: str) -> str:
    """
    Validates a time period.
    
    Args:
        period: Period to validate
        
    Returns:
        Validated period
        
    Raises:
        ValueError: If the period is invalid
    """
    if not period or not isinstance(period, str):
        raise ValueError("Period is required and must be a string")
    
    period = period.strip().lower()
    
    if period not in ("hour", "day", "week", "month"):
        raise ValueError("Period must be one of: hour, day, week, month")
    
    return period













