"""
Response Formatter for Color Grading API
==========================================

Standardized response formatting.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import status

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Standardized response formatter.
    
    Features:
    - Consistent response format
    - Error formatting
    - Pagination support
    - Metadata inclusion
    """
    
    @staticmethod
    def success(
        data: Any,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format success response.
        
        Args:
            data: Response data
            message: Optional message
            metadata: Optional metadata
            
        Returns:
            Formatted response
        """
        response = {
            "success": True,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if message:
            response["message"] = message
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def error(
        error: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> JSONResponse:
        """
        Format error response.
        
        Args:
            error: Error message
            code: Optional error code
            details: Optional error details
            status_code: HTTP status code
            
        Returns:
            JSONResponse with error
        """
        response_data = {
            "success": False,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if code:
            response_data["code"] = code
        
        if details:
            response_data["details"] = details
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    @staticmethod
    def paginated(
        items: List[Any],
        page: int,
        page_size: int,
        total: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format paginated response.
        
        Args:
            items: List of items
            page: Current page
            page_size: Items per page
            total: Total items
            metadata: Optional metadata
            
        Returns:
            Formatted paginated response
        """
        total_pages = (total + page_size - 1) // page_size
        
        response = {
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
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    @staticmethod
    def created(
        data: Any,
        resource_id: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format created response.
        
        Args:
            data: Created resource data
            resource_id: Optional resource ID
            location: Optional resource location
            
        Returns:
            Formatted response
        """
        response = {
            "success": True,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if resource_id:
            response["id"] = resource_id
        
        if location:
            response["location"] = location
        
        return response




