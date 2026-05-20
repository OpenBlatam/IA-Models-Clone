"""
Error Handling Middleware
========================

Middleware for consistent error handling across the API.
"""

import logging
from typing import Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.exceptions import ClothingChangerError

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle errors consistently."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request and handle errors.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response with consistent error format
        """
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "success": False,
                    "error": e.detail if isinstance(e.detail, str) else e.detail,
                    "status_code": e.status_code,
                }
            )
        except ClothingChangerError as e:
            # Custom ClothingChanger exceptions
            status_code = 400 if e.code.startswith("VALIDATION") else 500
            logger.error(f"ClothingChangerError: {e.message}", extra=e.details)
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False,
                    "error": e.to_dict(),
                    "status_code": status_code,
                }
            )
        except RuntimeError as e:
            # Runtime errors (model initialization, etc.)
            logger.error(f"Runtime error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                    "type": "runtime_error",
                }
            )
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "type": "unexpected_error",
                    "message": str(e) if logger.level <= logging.DEBUG else None,
                }
            )

