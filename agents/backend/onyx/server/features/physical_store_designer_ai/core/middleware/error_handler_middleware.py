"""
Error Handler Middleware

Middleware for centralized error handling.
"""

from typing import Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging_config import get_logger
from ..exceptions import PhysicalStoreDesignerError
from ...config.settings import settings

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware para manejo centralizado de errores"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and handle errors"""
        try:
            response = await call_next(request)
            return response
        except PhysicalStoreDesignerError as e:
            logger.error(
                f"PhysicalStoreDesignerError: {e.message}",
                extra={
                    "error_code": e.error_code,
                    "status_code": e.status_code,
                    "details": e.details,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
        except HTTPException as e:
            logger.warning(
                f"HTTPException: {e.detail}",
                extra={
                    "status_code": e.status_code,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            return JSONResponse(
                status_code=e.status_code,
                content={"error": "HTTP_ERROR", "message": e.detail}
            )
        except Exception as e:
            logger.exception(
                f"Unhandled exception: {str(e)}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "exception_type": type(e).__name__
                }
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "Error interno del servidor",
                    "details": {"exception": str(e)} if settings.log_level == "DEBUG" else {}
                }
            )

