"""
Response builder utilities
Helper functions for building consistent API responses
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from utils.date_helpers import get_current_utc


def build_success_response(
    data: Any,
    message: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build success response
    
    Args:
        data: Response data
        message: Optional success message
        meta: Optional metadata
    
    Returns:
        Success response dictionary
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": get_current_utc().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if meta:
        response["meta"] = meta
    
    return response


def build_error_response(
    error: str,
    details: Optional[Dict[str, Any]] = None,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build error response
    
    Args:
        error: Error message
        details: Optional error details
        code: Optional error code
    
    Returns:
        Error response dictionary
    """
    response = {
        "success": False,
        "error": error,
        "timestamp": get_current_utc().isoformat()
    }
    
    if details:
        response["details"] = details
    
    if code:
        response["code"] = code
    
    return response


def build_paginated_response(
    items: List[Any],
    page: int,
    page_size: int,
    total_items: int,
    total_pages: int
) -> Dict[str, Any]:
    """
    Build paginated response
    
    Args:
        items: List of items
        page: Current page number
        page_size: Items per page
        total_items: Total number of items
        total_pages: Total number of pages
    
    Returns:
        Paginated response dictionary
    """
    return {
        "success": True,
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        },
        "timestamp": get_current_utc().isoformat()
    }


def build_list_response(
    items: List[Any],
    count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build list response
    
    Args:
        items: List of items
        count: Optional count (if None, uses len(items))
    
    Returns:
        List response dictionary
    """
    return {
        "success": True,
        "data": items,
        "count": count if count is not None else len(items),
        "timestamp": get_current_utc().isoformat()
    }

