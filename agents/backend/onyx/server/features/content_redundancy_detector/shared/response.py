"""
Shared Response Utilities
Standardized response formatting across the application
"""

import time
from typing import Any, Dict, Optional


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        meta: Optional metadata
        
    Returns:
        Standardized success response
    """
    response = {
        "success": True,
        "data": data,
        "error": None,
        "timestamp": time.time()
    }
    
    if message:
        response["message"] = message
    
    if meta:
        response["meta"] = meta
    
    return response


def create_error_response(
    message: str,
    status_code: int = 500,
    error_type: str = "Error",
    detail: Optional[Any] = None,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_type: Error type/class
        detail: Optional detailed error information
        code: Optional error code
        
    Returns:
        Standardized error response
    """
    error = {
        "message": message,
        "status_code": status_code,
        "type": error_type
    }
    
    if detail:
        error["detail"] = detail
    
    if code:
        error["code"] = code
    
    return {
        "success": False,
        "data": None,
        "error": error,
        "timestamp": time.time()
    }






