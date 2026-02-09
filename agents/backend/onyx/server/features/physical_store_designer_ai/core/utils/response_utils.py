"""
Response Utilities

Utilities for creating standardized API responses.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    resource_id: Optional[str] = None,
    **extra_fields: Any
) -> Dict[str, Any]:
    """
    Create a standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        resource_id: Optional resource ID
        **extra_fields: Additional fields to include
        
    Returns:
        Standardized success response dictionary
    """
    response: Dict[str, Any] = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        **extra_fields
    }
    
    if resource_id:
        response["resource_id"] = resource_id
    
    if message:
        response["message"] = message
    
    if isinstance(data, dict):
        response.update(data)
    else:
        response["data"] = data
    
    return response


def create_error_response(
    error: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400
) -> Dict[str, Any]:
    """
    Create a standardized error response
    
    Args:
        error: Error code or type
        message: Error message
        details: Optional error details
        status_code: HTTP status code
        
    Returns:
        Standardized error response dictionary
    """
    response: Dict[str, Any] = {
        "success": False,
        "error": error,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "status_code": status_code
    }
    
    if details:
        response["details"] = details
    
    return response


def create_paginated_response(
    items: List[Any],
    page: int,
    page_size: int,
    total: Optional[int] = None,
    **extra_fields: Any
) -> Dict[str, Any]:
    """
    Create a standardized paginated response
    
    Args:
        items: List of items for current page
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total: Total number of items (optional)
        **extra_fields: Additional fields to include
        
    Returns:
        Standardized paginated response dictionary
    """
    total_items = total if total is not None else len(items)
    total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 0
    
    response: Dict[str, Any] = {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "timestamp": datetime.now().isoformat(),
        **extra_fields
    }
    
    return response








