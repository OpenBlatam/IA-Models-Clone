"""
Error Response Formatter
Standardizes error response format across the API
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)


class ErrorFormatter:
    """Formats error responses consistently"""
    
    @staticmethod
    def format_error(
        error: Exception,
        status_code: int = 500,
        request_id: Optional[str] = None,
        include_traceback: bool = False,
        user_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error response
        
        Args:
            error: Exception instance
            status_code: HTTP status code
            request_id: Request ID for tracing
            include_traceback: Whether to include traceback
            user_message: User-friendly error message
            
        Returns:
            Formatted error response
        """
        error_response = {
            "error": {
                "type": error.__class__.__name__,
                "message": user_message or str(error),
                "status_code": status_code,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }
        
        if request_id:
            error_response["error"]["request_id"] = request_id
        
        if include_traceback:
            error_response["error"]["traceback"] = traceback.format_exc()
        
        # Add error code if available
        if hasattr(error, "error_code"):
            error_response["error"]["code"] = error.error_code
        
        # Add details if available
        if hasattr(error, "details"):
            error_response["error"]["details"] = error.details
        
        return error_response
    
    @staticmethod
    def format_validation_error(
        errors: list,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format validation error response
        
        Args:
            errors: List of validation errors
            request_id: Request ID for tracing
            
        Returns:
            Formatted validation error response
        """
        return {
            "error": {
                "type": "ValidationError",
                "message": "Validation failed",
                "status_code": 400,
                "timestamp": datetime.utcnow().isoformat(),
                "validation_errors": errors,
            },
            **({"request_id": request_id} if request_id else {})
        }
    
    @staticmethod
    def format_not_found_error(
        resource: str,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format not found error response
        
        Args:
            resource: Resource type
            resource_id: Resource ID
            request_id: Request ID for tracing
            
        Returns:
            Formatted not found error response
        """
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        
        return {
            "error": {
                "type": "NotFoundError",
                "message": message,
                "status_code": 404,
                "timestamp": datetime.utcnow().isoformat(),
                "resource": resource,
                **({"resource_id": resource_id} if resource_id else {})
            },
            **({"request_id": request_id} if request_id else {})
        }
    
    @staticmethod
    def format_rate_limit_error(
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format rate limit error response
        
        Args:
            retry_after: Seconds to wait before retrying
            request_id: Request ID for tracing
            
        Returns:
            Formatted rate limit error response
        """
        return {
            "error": {
                "type": "RateLimitError",
                "message": "Rate limit exceeded",
                "status_code": 429,
                "timestamp": datetime.utcnow().isoformat(),
                **({"retry_after": retry_after} if retry_after else {})
            },
            **({"request_id": request_id} if request_id else {})
        }
    
    @staticmethod
    def format_timeout_error(
        timeout: float,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format timeout error response
        
        Args:
            timeout: Timeout value in seconds
            request_id: Request ID for tracing
            
        Returns:
            Formatted timeout error response
        """
        return {
            "error": {
                "type": "TimeoutError",
                "message": f"Request exceeded timeout of {timeout} seconds",
                "status_code": 504,
                "timestamp": datetime.utcnow().isoformat(),
                "timeout": timeout
            },
            **({"request_id": request_id} if request_id else {})
        }


# Global error formatter
_error_formatter = ErrorFormatter()


def get_error_formatter() -> ErrorFormatter:
    """Get global error formatter"""
    return _error_formatter















