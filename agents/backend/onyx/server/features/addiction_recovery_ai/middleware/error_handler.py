"""
Error handling middleware for FastAPI
"""

import logging
from typing import Callable
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from utils.errors import APIError, handle_service_error

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle errors consistently across the API"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        try:
            response = await call_next(request)
            return response
        except APIError as error:
            logger.warning(
                f"API Error: {error.message}",
                extra={
                    "error_code": error.error_code,
                    "status_code": error.status_code,
                    "path": request.url.path
                }
            )
            http_exception = handle_service_error(error)
            return JSONResponse(
                status_code=http_exception.status_code,
                content=http_exception.detail
            )
        except Exception as error:
            logger.error(
                f"Unexpected error: {str(error)}",
                exc_info=True,
                extra={"path": request.url.path}
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "error",
                    "message": "Error interno del servidor",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "timestamp": None
                }
            )

