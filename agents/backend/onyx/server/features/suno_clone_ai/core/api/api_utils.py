"""
API Utilities

Utilities for API handling.
"""

import logging
from typing import Dict, Any, Optional, Callable
import json

logger = logging.getLogger(__name__)


class APIHandler:
    """API request/response handler."""
    
    def __init__(self):
        """Initialize API handler."""
        self.middlewares: list = []
    
    def add_middleware(self, middleware: Callable) -> 'APIHandler':
        """
        Add middleware.
        
        Args:
            middleware: Middleware function
            
        Returns:
            Self for chaining
        """
        self.middlewares.append(middleware)
        return self
    
    def handle(
        self,
        request: Dict[str, Any],
        handler: Callable
    ) -> Dict[str, Any]:
        """
        Handle API request.
        
        Args:
            request: Request dictionary
            handler: Handler function
            
        Returns:
            Response dictionary
        """
        # Apply middleware
        for middleware in self.middlewares:
            request = middleware(request)
        
        # Call handler
        try:
            response = handler(request)
            return self._format_success(response)
        except Exception as e:
            logger.error(f"API error: {e}")
            return self._format_error(str(e))
    
    def _format_success(self, data: Any) -> Dict[str, Any]:
        """Format success response."""
        return {
            'success': True,
            'data': data
        }
    
    def _format_error(self, error: str) -> Dict[str, Any]:
        """Format error response."""
        return {
            'success': False,
            'error': error
        }


def create_api_handler() -> APIHandler:
    """Create API handler."""
    return APIHandler()


def validate_request(
    request: Dict[str, Any],
    required_fields: list,
    **kwargs
) -> tuple:
    """
    Validate API request.
    
    Args:
        request: Request dictionary
        required_fields: List of required field names
        **kwargs: Additional validation rules
        
    Returns:
        (is_valid, error_message)
    """
    for field in required_fields:
        if field not in request:
            return False, f"Missing required field: {field}"
    
    return True, None


def format_response(
    data: Any,
    success: bool = True,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format API response.
    
    Args:
        data: Response data
        success: Success flag
        error: Error message
        
    Returns:
        Formatted response
    """
    response = {
        'success': success,
        'data': data if success else None
    }
    
    if error:
        response['error'] = error
    
    return response



