"""
Error Handler Middleware
========================

FastAPI middleware for centralized error handling.
"""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Callable

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling.
    
    Features:
    - Catch all exceptions
    - Format error responses consistently
    - Log errors with context
    - Return appropriate status codes
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """
        Process request with error handling.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response or error response
        """
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Re-raise HTTP exceptions (they're already formatted)
            raise
            
        except ValueError as e:
            # Validation errors
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": str(e),
                    "error_type": "validation_error"
                }
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {e}",
                exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "error_type": "internal_error",
                    "message": str(e) if logger.level <= logging.DEBUG else None
                }
            )

