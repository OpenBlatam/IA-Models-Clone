"""
Response builder utilities
"""

from typing import Any, Dict, List, Optional
from datetime import datetime


def build_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a standardized success response
    
    Args:
        data: Response data
        message: Optional message
        metadata: Optional metadata
    
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def build_error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a standardized error response
    
    Args:
        message: Error message
        error_code: Optional error code
        details: Optional error details
    
    Returns:
        Standardized error response dictionary
    """
    response = {
        "success": False,
        "error": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    return response


def build_list_response(
    items: List[Any],
    total: Optional[int] = None,
    page: Optional[int] = None,
    page_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build a standardized list response
    
    Args:
        items: List of items
        total: Total count (if paginated)
        page: Current page (if paginated)
        page_size: Page size (if paginated)
    
    Returns:
        Standardized list response dictionary
    """
    response = {
        "items": items,
        "count": len(items)
    }
    
    if total is not None:
        response["total"] = total
    
    if page is not None and page_size is not None:
        response["pagination"] = {
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if total else 0
        }
    
    return response

