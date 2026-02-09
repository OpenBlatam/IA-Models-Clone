"""
Error Utilities

Utilities for error handling and exception management.
"""

from typing import Dict, Any, Optional, Type
from ..exceptions import PhysicalStoreDesignerError


def get_error_response(exception: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response
    
    Args:
        exception: Exception to convert
        
    Returns:
        Standardized error response dictionary
    """
    if isinstance(exception, PhysicalStoreDesignerError):
        return {
            "success": False,
            "error": exception.error_code,
            "message": exception.message,
            "status_code": exception.status_code,
            "details": exception.details
        }
    
    # Generic exception
    return {
        "success": False,
        "error": type(exception).__name__,
        "message": str(exception),
        "status_code": 500,
        "details": {}
    }


def is_client_error(status_code: int) -> bool:
    """
    Check if status code represents a client error (4xx)
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if status code is 4xx
    """
    return 400 <= status_code < 500


def is_server_error(status_code: int) -> bool:
    """
    Check if status code represents a server error (5xx)
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if status code is 5xx
    """
    return 500 <= status_code < 600


def get_retryable_status_codes() -> list[int]:
    """
    Get list of HTTP status codes that are typically retryable
    
    Returns:
        List of retryable status codes
    """
    return [
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    ]


def should_retry(status_code: int) -> bool:
    """
    Check if a request with given status code should be retried
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if request should be retried
    """
    return status_code in get_retryable_status_codes()








