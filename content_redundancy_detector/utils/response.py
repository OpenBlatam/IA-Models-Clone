"""
Response Utilities
Helper functions for creating standardized API responses
"""

import time
from typing import Any, Dict, Optional


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    processing_time: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        request_id: Optional request identifier
        processing_time: Optional processing time in seconds
        
    Returns:
        Standardized success response dictionary
    """
    response = {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": time.time()
    }
    
    if message:
        response["message"] = message
    
    if request_id:
        response["request_id"] = request_id
    
    if processing_time is not None:
        response["processing_time"] = processing_time
    
    return response


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: str = "Error",
    detail: Optional[Any] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_type: Error type identifier
        detail: Optional error details
        request_id: Optional request identifier
        
    Returns:
        Standardized error response dictionary
    """
    error = {
        "message": message,
        "status_code": status_code,
        "type": error_type
    }
    
    if detail:
        error["detail"] = detail
    
    response = {
        "success": False,
        "data": None,
        "error": error,
        "timestamp": time.time()
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def create_paginated_response(
    items: list,
    page: int,
    page_size: int,
    total: int,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create paginated response
    
    Args:
        items: List of items for current page
        page: Current page number
        page_size: Number of items per page
        total: Total number of items
        message: Optional message
        
    Returns:
        Paginated response dictionary
    """
    total_pages = (total + page_size - 1) // page_size
    
    data = {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
    }
    
    return create_success_response(
        data=data,
        message=message or f"Retrieved {len(items)} items"
    )





