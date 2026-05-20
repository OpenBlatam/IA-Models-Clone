"""
Middleware global para manejo de errores (optimizado)

Maneja errores de forma consistente y user-friendly.

Este middleware:
- Captura todas las excepciones no manejadas
- Proporciona respuestas consistentes usando ErrorResponse
- Registra errores con contexto completo
- Oculta detalles internos en producción
- Proporciona información útil en modo debug
"""

from typing import Callable
from datetime import datetime
from fastapi import Request, status, HTTPException
from fastapi.responses import ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..exceptions import (
    BaseCommunityException,
    ChatNotFoundError,
    InvalidChatError,
    DatabaseError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError
)
from ..schemas import ErrorResponse
from ..config import settings
from ..utils.logging_config import StructuredLogger

logger = StructuredLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware para manejo centralizado de errores (optimizado).
    
    Maneja errores de forma consistente usando excepciones personalizadas
    y proporciona respuestas user-friendly.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Procesa el request con manejo de errores centralizado.
        
        Args:
            request: Request object
            call_next: Función para continuar el pipeline
            
        Returns:
            Response o error response
        """
        try:
            response = await call_next(request)
            return response
        except HTTPException:
            # Re-raise HTTPExceptions (ya están manejadas)
            raise
        except BaseCommunityException as exc:
            # Manejar excepciones personalizadas de la API
            user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
            
            # Extraer detalles del error de forma segura
            error_detail = getattr(exc, 'detail', str(exc))
            status_code = getattr(exc, 'status_code', status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            if status_code < 500:
                logger.warning(
                    "API exception",
                    path=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    exception_type=type(exc).__name__,
                    message=error_detail,
                    user_id=user_id,
                    query_params=dict(request.query_params) if request.query_params else None
                )
            else:
                logger.error(
                    "API exception",
                    path=request.url.path,
                    method=request.method,
                    status_code=status_code,
                    exception_type=type(exc).__name__,
                    message=error_detail,
                    user_id=user_id,
                    query_params=dict(request.query_params) if request.query_params else None
                )
            
            # Usar ErrorResponse schema para consistencia
            error_response = ErrorResponse(
                error=type(exc).__name__,
                message=error_detail,
                path=request.url.path,
                timestamp=datetime.utcnow()
            )
            
            return ORJSONResponse(
                status_code=status_code,
                content=error_response.model_dump()
            )
        except ValueError as exc:
            # Errores de validación
            user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
            logger.warning(
                "Validation error",
                path=request.url.path,
                method=request.method,
                exception_type="ValueError",
                message=str(exc),
                user_id=user_id
            )
            
            error_response = ErrorResponse(
                error="ValidationError",
                message=str(exc),
                path=request.url.path,
                timestamp=datetime.utcnow()
            )
            
            return ORJSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=error_response.model_dump()
            )
        except Exception as exc:
            # Errores no manejados
            error_message = "Internal server error"
            error_details = None
            
            if settings.debug:
                error_message = str(exc)
                error_details = {
                    "exception_type": type(exc).__name__,
                    "traceback": str(exc.__traceback__) if hasattr(exc, '__traceback__') else None
                }
            
            user_id = request.headers.get("X-User-ID") or request.query_params.get("user_id")
            logger.exception(
                "Unhandled exception",
                path=request.url.path,
                method=request.method,
                client_host=request.client.host if request.client else None,
                exception_type=type(exc).__name__,
                user_id=user_id
            )
            
            error_response = ErrorResponse(
                error="InternalServerError",
                message=error_message,
                details=error_details,
                path=request.url.path,
                timestamp=datetime.utcnow()
            )
            
            return ORJSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response.model_dump()
            )

