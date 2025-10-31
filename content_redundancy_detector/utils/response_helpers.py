"""
Response Helpers - Standardized API response formatting
"""

import time
from typing import Any, Dict, Optional, List
from fastapi import Request
from fastapi.responses import JSONResponse
import uuid


def get_request_id(request: Request) -> str:
    """Get or generate request ID"""
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    return str(uuid.uuid4())


def set_request_id(request: Request, request_id: Optional[str] = None) -> str:
    """Set request ID in request state"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        metadata: Optional metadata
        request_id: Request ID for tracing
        
    Returns:
        Formatted success response
    """
    response = {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": time.time()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def create_error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        code: Error code
        message: Error message
        details: Additional error details
        request_id: Request ID for tracing
        
    Returns:
        Formatted error response
    """
    response = {
        "success": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
        },
        "timestamp": time.time()
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 20,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized paginated response
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        page_size: Items per page
        request_id: Request ID for tracing
        
    Returns:
        Formatted paginated response
    """
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    response = create_success_response(
        data=items,
        metadata={
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        },
        request_id=request_id
    )
    
    return response


def json_response(
    data: Any,
    status_code: int = 200,
    headers: Optional[Dict[str, str]] = None,
    request_id: Optional[str] = None
) -> JSONResponse:
    """
    Create JSON response with standard format
    
    Args:
        data: Response data (dict with success/error structure)
        status_code: HTTP status code
        headers: Additional headers
        request_id: Request ID for tracing
        
    Returns:
        JSONResponse
    """
    if headers is None:
        headers = {}
    
    # Ensure Content-Type header
    headers["Content-Type"] = "application/json"
    
    # Add request ID to headers if provided
    if request_id:
        headers["X-Request-ID"] = request_id
    
    return JSONResponse(
        content=data,
        status_code=status_code,
        headers=headers
    )






