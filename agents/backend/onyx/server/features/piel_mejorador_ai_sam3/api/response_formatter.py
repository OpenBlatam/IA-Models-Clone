"""
Response Formatter for Piel Mejorador AI SAM3
=============================================

Enhanced API response formatting.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)


class ResponseFormatter:
    """
    Formats API responses consistently.
    
    Features:
    - Consistent response format
    - Error handling
    - Metadata inclusion
    - Pagination support
    """
    
    @staticmethod
    def success(
        data: Any = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create success response.
        
        Args:
            data: Response data
            message: Optional message
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        response = APIResponse(
            success=True,
            data=data,
            message=message,
            metadata=metadata
        )
        return JSONResponse(content=response.dict(), status_code=200)
    
    @staticmethod
    def error(
        message: str,
        errors: Optional[List[str]] = None,
        status_code: int = 400,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create error response.
        
        Args:
            message: Error message
            errors: Optional list of errors
            status_code: HTTP status code
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        response = APIResponse(
            success=False,
            message=message,
            errors=errors or [],
            metadata=metadata
        )
        return JSONResponse(content=response.dict(), status_code=status_code)
    
    @staticmethod
    def paginated(
        items: List[Any],
        page: int,
        page_size: int,
        total: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        Create paginated response.
        
        Args:
            items: List of items
            page: Current page
            page_size: Items per page
            total: Total items
            metadata: Optional metadata
            
        Returns:
            Formatted JSON response
        """
        total_pages = (total + page_size - 1) // page_size
        
        pagination_metadata = {
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }
        }
        
        if metadata:
            pagination_metadata.update(metadata)
        
        response = APIResponse(
            success=True,
            data=items,
            metadata=pagination_metadata
        )
        return JSONResponse(content=response.dict(), status_code=200)




