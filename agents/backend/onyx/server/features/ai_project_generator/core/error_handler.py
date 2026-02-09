"""
Error Handler - Manejador de errores centralizado
==================================================

Manejo centralizado de errores con logging y tracking.
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .exceptions import AIProjectGeneratorError
from ..debug.error_tracker import get_error_tracker
from ..debug.debug_logger import get_debug_logger

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Manejador centralizado de errores"""
    
    def __init__(self):
        self.error_tracker = get_error_tracker()
        self.debug_logger = get_debug_logger()
    
    async def handle_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Maneja excepción y retorna respuesta JSON.
        
        Args:
            request: Request de FastAPI
            exc: Excepción
        
        Returns:
            Respuesta JSON con error
        """
        # Trackear error
        request_id = request.headers.get("X-Request-ID", "unknown")
        self.error_tracker.track_error(
            error=exc,
            context={
                "method": request.method,
                "path": request.url.path
            },
            request_id=request_id
        )
        
        # Log error
        self.debug_logger.log_exception(
            exception=exc,
            context={
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id
            }
        )
        
        # Determinar status code y mensaje
        if isinstance(exc, AIProjectGeneratorError):
            status_code = 400
            message = str(exc)
        elif isinstance(exc, HTTPException):
            status_code = exc.status_code
            message = exc.detail
        elif isinstance(exc, RequestValidationError):
            status_code = 422
            message = "Validation error"
            errors = exc.errors()
        else:
            status_code = 500
            message = "Internal server error"
            errors = None
        
        # Construir respuesta
        response_data = {
            "error": {
                "type": type(exc).__name__,
                "message": message,
                "request_id": request_id
            }
        }
        
        if errors:
            response_data["error"]["details"] = errors
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    async def handle_validation_error(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Maneja errores de validación"""
        return await self.handle_exception(request, exc)
    
    async def handle_http_exception(
        self,
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """Maneja excepciones HTTP"""
        return await self.handle_exception(request, exc)


# Instancia global
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Obtiene instancia de error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def setup_error_handlers(app):
    """Configura handlers de errores en la app"""
    error_handler = get_error_handler()
    
    @app.exception_handler(AIProjectGeneratorError)
    async def custom_exception_handler(request: Request, exc: AIProjectGeneratorError):
        return await error_handler.handle_exception(request, exc)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return await error_handler.handle_validation_error(request, exc)
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return await error_handler.handle_http_exception(request, exc)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return await error_handler.handle_exception(request, exc)















