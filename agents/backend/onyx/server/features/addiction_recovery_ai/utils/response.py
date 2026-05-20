"""
Response utilities for consistent API responses
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import status


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK
) -> JSONResponse:
    """Create a standardized success response"""
    response_data = {
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if message:
        response_data["message"] = message
    
    return JSONResponse(content=response_data, status_code=status_code)


def create_error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
) -> JSONResponse:
    """Create a standardized error response"""
    response_data = {
        "status": "error",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if error_code:
        response_data["error_code"] = error_code
    
    if details:
        response_data["details"] = details
    
    return JSONResponse(content=response_data, status_code=status_code)


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """Create a paginated response structure"""
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    return {
        "items": items,
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        }
    }

