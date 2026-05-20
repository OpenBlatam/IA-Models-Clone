"""
Query parameter parsing and validation utilities
"""

from typing import Any, Optional, List
from datetime import datetime
from fastapi import Query


def parse_date_query(
    date_str: Optional[str] = None,
    param_name: str = "date"
) -> Optional[datetime]:
    """
    Parse date from query parameter
    
    Args:
        date_str: Date string from query parameter
        param_name: Name of parameter for error messages
    
    Returns:
        Parsed datetime or None
    
    Raises:
        ValueError if date format is invalid
    """
    if not date_str:
        return None
    
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(f"Invalid {param_name} format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")


def parse_list_query(
    value: Optional[str] = None,
    separator: str = ","
) -> List[str]:
    """
    Parse comma-separated list from query parameter
    
    Args:
        value: Comma-separated string
        separator: Separator character
    
    Returns:
        List of strings
    """
    if not value:
        return []
    
    return [item.strip() for item in value.split(separator) if item.strip()]


def create_date_range_query(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Create date range query parameters
    
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    start = parse_date_query(start_date, "start_date") if start_date else None
    end = parse_date_query(end_date, "end_date") if end_date else None
    
    if start and end and start > end:
        raise ValueError("start_date must be before end_date")
    
    return start, end


def create_filter_query(
    field: Optional[str] = Query(None, description="Field to filter by"),
    value: Optional[str] = Query(None, description="Value to filter for"),
    operator: str = Query("equals", description="Filter operator (equals, contains, gt, lt)")
) -> Dict[str, Any] | None:
    """
    Create filter query parameters
    
    Returns:
        Dictionary with filter parameters or None
    """
    if not field or not value:
        return None
    
    valid_operators = ["equals", "contains", "gt", "lt", "gte", "lte"]
    if operator not in valid_operators:
        raise ValueError(f"Invalid operator. Must be one of: {valid_operators}")
    
    return {
        "field": field,
        "value": value,
        "operator": operator
    }

