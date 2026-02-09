"""
Response formatting utilities for consistent API responses.

Provides standardized response formatting across the application.
"""

from typing import Any, Dict, Optional, List
from fastapi import Response
from fastapi.responses import JSONResponse, ORJSONResponse


class ResponseFormatter:
    """Formatter for API responses."""
    
    @staticmethod
    def success(
        data: Any = None,
        message: Optional[str] = None,
        status_code: int = 200,
        use_orjson: bool = True
    ) -> Response:
        """
        Format a successful response.
        
        Args:
            data: Response data
            message: Optional success message
            status_code: HTTP status code
            use_orjson: Whether to use ORJSONResponse
        
        Returns:
            Formatted response
        """
        response_data: Dict[str, Any] = {
            "success": True,
            "status": "success"
        }
        
        if message:
            response_data["message"] = message
        
        if data is not None:
            response_data["data"] = data
        
        response_class = ORJSONResponse if use_orjson else JSONResponse
        return response_class(
            content=response_data,
            status_code=status_code
        )
    
    @staticmethod
    def error(
        message: str,
        status_code: int = 400,
        errors: Optional[List[str]] = None,
        use_orjson: bool = True
    ) -> Response:
        """
        Format an error response.
        
        Args:
            message: Error message
            status_code: HTTP status code
            errors: Optional list of detailed errors
            use_orjson: Whether to use ORJSONResponse
        
        Returns:
            Formatted error response
        """
        response_data: Dict[str, Any] = {
            "success": False,
            "status": "error",
            "message": message
        }
        
        if errors:
            response_data["errors"] = errors
        
        response_class = ORJSONResponse if use_orjson else JSONResponse
        return response_class(
            content=response_data,
            status_code=status_code
        )
    
    @staticmethod
    def paginated(
        items: List[Any],
        page: int,
        page_size: int,
        total: int,
        use_orjson: bool = True
    ) -> Response:
        """
        Format a paginated response.
        
        Args:
            items: List of items
            page: Current page number
            page_size: Items per page
            total: Total number of items
            use_orjson: Whether to use ORJSONResponse
        
        Returns:
            Formatted paginated response
        """
        response_data = {
            "success": True,
            "status": "success",
            "data": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "pages": (total + page_size - 1) // page_size if page_size > 0 else 0
            }
        }
        
        response_class = ORJSONResponse if use_orjson else JSONResponse
        return response_class(content=response_data)


# Global formatter instance
formatter = ResponseFormatter()


def format_success(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = 200
) -> Response:
    """Convenience function for success responses."""
    return formatter.success(data, message, status_code)


def format_error(
    message: str,
    status_code: int = 400,
    errors: Optional[List[str]] = None
) -> Response:
    """Convenience function for error responses."""
    return formatter.error(message, status_code, errors)


def format_paginated(
    items: List[Any],
    page: int,
    page_size: int,
    total: int
) -> Response:
    """Convenience function for paginated responses."""
    return formatter.paginated(items, page, page_size, total)

