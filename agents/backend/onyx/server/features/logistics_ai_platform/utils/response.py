"""
Response utilities

This module provides standardized response formatting functions
for consistent API responses across the application.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import status


def success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK,
    metadata: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code (default: 200)
        metadata: Optional metadata to include
        
    Returns:
        JSONResponse with success format
        
    Example:
        {
            "success": true,
            "data": {...},
            "message": "Operation successful",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    """
    content: Dict[str, Any] = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        content["message"] = message
    
    if metadata:
        content["metadata"] = metadata
    
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json"
    )


def error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None,
    field: Optional[str] = None
) -> JSONResponse:
    """
    Create a standardized error response
    
    Args:
        message: Error message
        error_code: Optional error code
        status_code: HTTP status code (default: 400)
        details: Optional error details
        field: Optional field name that caused the error
        
    Returns:
        JSONResponse with error format
        
    Example:
        {
            "success": false,
            "error": {
                "message": "Validation failed",
                "code": "VALIDATION_ERROR",
                "field": "email",
                "details": {...}
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
    """
    error_content: Dict[str, Any] = {
        "message": message,
        "code": error_code or "ERROR"
    }
    
    if field:
        error_content["field"] = field
    
    if details:
        error_content["details"] = details
    
    content = {
        "success": False,
        "error": error_content,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return JSONResponse(
        content=content,
        status_code=status_code,
        media_type="application/json"
    )


def paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 100,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    Create a standardized paginated response
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
        message: Optional message
        metadata: Optional metadata to include
        
    Returns:
        JSONResponse with pagination format
        
    Example:
        {
            "success": true,
            "data": [...],
            "pagination": {
                "page": 1,
                "page_size": 100,
                "total": 500,
                "total_pages": 5,
                "has_next": true,
                "has_prev": false
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
    """
    if page < 1:
        page = 1
    
    if page_size < 1:
        page_size = 100
    
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    content: Dict[str, Any] = {
        "success": True,
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        content["message"] = message
    
    if metadata:
        content["metadata"] = metadata
    
    return JSONResponse(
        content=content,
        status_code=status.HTTP_200_OK,
        media_type="application/json"
    )


def created_response(
    data: Any,
    message: Optional[str] = None,
    location: Optional[str] = None
) -> JSONResponse:
    """
    Create a standardized created response (201)
    
    Args:
        data: Created resource data
        message: Optional success message
        location: Optional location header value
        
    Returns:
        JSONResponse with 201 status code
    """
    headers = {}
    if location:
        headers["Location"] = location
    
    return success_response(
        data=data,
        message=message or "Resource created successfully",
        status_code=status.HTTP_201_CREATED,
        metadata={"location": location} if location else None
    )


def no_content_response() -> JSONResponse:
    """
    Create a no content response (204)
    
    Returns:
        JSONResponse with 204 status code
    """
    return JSONResponse(
        content=None,
        status_code=status.HTTP_204_NO_CONTENT
    )

