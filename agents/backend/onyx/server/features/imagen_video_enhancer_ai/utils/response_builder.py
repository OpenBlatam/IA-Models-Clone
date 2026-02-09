"""
Response Builder
================

Utilities for building consistent API responses.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from enum import Enum


class ResponseStatus(Enum):
    """Response status codes."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ResponseBuilder:
    """Builder for consistent API responses."""
    
    @staticmethod
    def success(
        data: Any = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build success response.
        
        Args:
            data: Response data
            message: Optional message
            metadata: Optional metadata
            
        Returns:
            Response dictionary
        """
        response = {
            "status": ResponseStatus.SUCCESS.value,
            "timestamp": datetime.now().isoformat(),
        }
        
        if data is not None:
            response["data"] = data
        
        if message:
            response["message"] = message
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def error(
        error: str | Exception,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build error response.
        
        Args:
            error: Error message or exception
            code: Optional error code
            details: Optional error details
            
        Returns:
            Error response dictionary
        """
        error_message = str(error) if isinstance(error, Exception) else error
        
        response = {
            "status": ResponseStatus.ERROR.value,
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        
        if code:
            response["error_code"] = code
        
        if details:
            response["details"] = details
        
        return response
    
    @staticmethod
    def paginated(
        items: List[Any],
        page: int,
        page_size: int,
        total: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build paginated response.
        
        Args:
            items: List of items
            page: Current page number
            page_size: Items per page
            total: Total number of items
            metadata: Optional metadata
            
        Returns:
            Paginated response dictionary
        """
        response = {
            "status": ResponseStatus.SUCCESS.value,
            "data": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "has_more": total is None or (page * page_size) < total,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        if total is not None:
            response["pagination"]["total"] = total
            response["pagination"]["total_pages"] = (total + page_size - 1) // page_size
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def list_response(
        items: List[Any],
        count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build list response.
        
        Args:
            items: List of items
            count: Optional item count
            metadata: Optional metadata
            
        Returns:
            List response dictionary
        """
        response = {
            "status": ResponseStatus.SUCCESS.value,
            "data": items,
            "count": count if count is not None else len(items),
            "timestamp": datetime.now().isoformat(),
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response




