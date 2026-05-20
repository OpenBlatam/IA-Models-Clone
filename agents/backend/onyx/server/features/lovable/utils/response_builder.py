"""
Response builder utilities for consistent API responses.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


def build_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a success response.
    
    Args:
        data: Response data
        message: Optional success message
        metadata: Optional metadata
        
    Returns:
        Dictionary with success response structure
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def build_error_response(
    error: str,
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build an error response.
    
    Args:
        error: Error type
        message: Error message
        status_code: HTTP status code
        details: Optional error details
        
    Returns:
        Dictionary with error response structure
    """
    response = {
        "success": False,
        "error": error,
        "message": message,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        response["details"] = details
    
    return response


def build_paginated_response(
    items: List[Any],
    page: int,
    page_size: int,
    total: int,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a paginated response.
    
    Args:
        items: List of items
        page: Current page number
        page_size: Items per page
        total: Total number of items
        metadata: Optional additional metadata
        
    Returns:
        Dictionary with paginated response structure
    """
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    
    response = {
        "success": True,
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if metadata:
        response["metadata"] = metadata
    
    return response






