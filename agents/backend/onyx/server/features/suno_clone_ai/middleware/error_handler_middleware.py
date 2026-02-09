"""
Middleware global para manejo de errores (optimizado)

Maneja errores de forma consistente y user-friendly.
"""

import logging
from typing import Callable
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse, ORJSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from api.exceptions import BaseAPIException
    from api.utils.request_helpers import get_request_metadata
    from config.settings import settings
except ImportError:
    from ..api.exceptions import BaseAPIException
    from ..api.utils.request_helpers import get_request_metadata
    from ..config.settings import settings

logger = logging.getLogger(__name__)


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
        except BaseAPIException as exc:
            # Manejar excepciones personalizadas de la API
            request_metadata = get_request_metadata(request)
            logger.warning(
                f"API exception: {exc.detail}",
                extra=request_metadata
            )
            return ORJSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "message": exc.detail,
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except ValueError as exc:
            # Errores de validación
            request_metadata = get_request_metadata(request)
            logger.warning(
                f"Validation error: {exc}",
                extra=request_metadata
            )
            return ORJSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": True,
                    "message": str(exc),
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except FileNotFoundError as exc:
            # Archivos no encontrados
            request_metadata = get_request_metadata(request)
            logger.warning(
                f"File not found: {exc}",
                extra=request_metadata
            )
            return ORJSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": True,
                    "message": str(exc),
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except PermissionError as exc:
            # Errores de permisos
            request_metadata = get_request_metadata(request)
            logger.warning(
                f"Permission error: {exc}",
                extra=request_metadata
            )
            return ORJSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": True,
                    "message": str(exc),
                    "path": request.url.path,
                    "method": request.method
                }
            )
        except Exception as exc:
            # Errores no manejados
            request_metadata = get_request_metadata(request)
            logger.error(
                f"Unhandled exception: {exc}",
                exc_info=True,
                extra=request_metadata
            )
            return ORJSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": True,
                    "message": "Internal server error" if not settings.debug else str(exc),
                    "path": request.url.path,
                    "method": request.method
                }
            )

