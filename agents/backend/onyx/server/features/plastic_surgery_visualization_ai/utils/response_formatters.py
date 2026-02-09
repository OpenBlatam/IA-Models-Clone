"""Response formatting utilities."""

from typing import Any, Dict, Optional
from datetime import datetime
from fastapi.responses import JSONResponse


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = 200
) -> JSONResponse:
    """
    Create success response.
    
    Args:
        data: Response data
        message: Optional message
        status_code: HTTP status code
        
    Returns:
        JSONResponse
    """
    response_data = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if message:
        response_data["message"] = message
    
    if data is not None:
        response_data["data"] = data
    
    return JSONResponse(content=response_data, status_code=status_code)


def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create error response.
    
    Args:
        message: Error message
        error_code: Optional error code
        details: Optional error details
        status_code: HTTP status code
        
    Returns:
        JSONResponse
    """
    response_data = {
        "success": False,
        "error": {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    }
    
    if error_code:
        response_data["error"]["code"] = error_code
    
    if details:
        response_data["error"]["details"] = details
    
    return JSONResponse(content=response_data, status_code=status_code)


def paginated_response(
    items: list,
    page: int,
    page_size: int,
    total: Optional[int] = None,
    message: Optional[str] = None,
    as_dict: bool = False
) -> Any:
    """
    Create paginated response.
    
    Args:
        items: List of items
        page: Current page number
        page_size: Items per page
        total: Total number of items
        message: Optional message
        as_dict: Return as dict instead of JSONResponse
        
    Returns:
        JSONResponse or Dict
    """
    total_pages = (total + page_size - 1) // page_size if total and total > 0 else 0
    
    response_data = {
        "success": True,
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if message:
        response_data["message"] = message
    
    if as_dict:
        return response_data
    
    return JSONResponse(content=response_data, status_code=200)

