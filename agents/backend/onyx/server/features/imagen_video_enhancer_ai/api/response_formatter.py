"""
Response Formatter
==================

Consistent response formatting for API endpoints.
"""

from typing import Any, Dict, Optional, List
from fastapi.responses import JSONResponse
from datetime import datetime


class ResponseFormatter:
    """Response formatter for consistent API responses."""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status_code: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Format success response.
        
        Args:
            data: Response data
            message: Success message
            status_code: HTTP status code
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return JSONResponse(content=response, status_code=status_code)
    
    @staticmethod
    def error(
        message: str = "Error",
        status_code: int = 400,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Format error response.
        
        Args:
            message: Error message
            status_code: HTTP status code
            errors: Optional list of errors
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if errors:
            response["errors"] = errors
        
        if metadata:
            response["metadata"] = metadata
        
        return JSONResponse(content=response, status_code=status_code)
    
    @staticmethod
    def paginated(
        data: List[Any],
        page: int,
        page_size: int,
        total: int,
        message: str = "Success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Format paginated response.
        
        Args:
            data: List of items
            page: Current page
            page_size: Items per page
            total: Total items
            message: Success message
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size if page_size > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return JSONResponse(content=response, status_code=200)




