"""
Standardized API response helpers.
Centralized functions for creating consistent API responses.
"""

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


def success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = 200
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
    
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": True,
        "data": data
    }
    if message:
        response["message"] = message
    return response


def error_response(
    error: str,
    error_code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        error: Error message
        error_code: Optional error code
        status_code: HTTP status code
    
    Returns:
        JSONResponse with error details
    """
    response = {
        "success": False,
        "error": error
    }
    if error_code:
        response["error_code"] = error_code
    
    return JSONResponse(
        content=response,
        status_code=status_code
    )


def paginated_response(
    items: list,
    total: int,
    page: int = 1,
    page_size: int = 10,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized paginated response.
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        page_size: Number of items per page
        message: Optional message
    
    Returns:
        Standardized paginated response dictionary
    """
    total_pages = (total + page_size - 1) // page_size
    
    response = {
        "success": True,
        "data": {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
    }
    if message:
        response["message"] = message
    
    return response






