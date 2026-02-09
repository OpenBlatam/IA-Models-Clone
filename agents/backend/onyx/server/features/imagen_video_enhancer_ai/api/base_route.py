"""
Base Route
==========

Base class for API routes with common functionality.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .route_helpers import (
    handle_route_error,
    create_success_response,
    create_error_response,
    with_error_handling
)
from .dependencies import get_agent
from ..utils.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)


class BaseRoute:
    """Base class for route handlers."""
    
    def __init__(self, router: APIRouter, tags: list[str] = None):
        """
        Initialize base route.
        
        Args:
            router: FastAPI router
            tags: Route tags
        """
        self.router = router
        self.tags = tags or []
    
    def get_agent(self):
        """Get agent instance."""
        return get_agent()
    
    def success_response(
        self,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        **kwargs
    ) -> JSONResponse:
        """Create success response."""
        return create_success_response(data, message, **kwargs)
    
    def error_response(
        self,
        error: str | Exception,
        code: Optional[str] = None,
        status_code: int = 400
    ) -> JSONResponse:
        """Create error response."""
        return create_error_response(error, code, status_code)
    
    def handle_error(self, error: Exception, default_message: str = "An error occurred") -> HTTPException:
        """Handle route error."""
        return handle_route_error(error, default_message)
    
    def safe_execute(
        self,
        func: callable,
        error_message: str = "An error occurred",
        status_code: int = 500
    ):
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            error_message: Error message on failure
            status_code: HTTP status code on failure
            
        Returns:
            Function result or raises HTTPException
        """
        try:
            return func()
        except Exception as e:
            raise self.handle_error(e, error_message)




