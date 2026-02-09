"""
Error Middleware
================
"""

from typing import Callable
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from datetime import datetime
from ..core.exceptions import (
    DogTrainingException,
    OpenRouterException,
    ValidationException,
    ServiceException
)
from ..core.error_codes import ErrorCode, get_error_message
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ErrorMiddleware(BaseHTTPMiddleware):
    """Middleware para manejo centralizado de errores."""
    
    def _create_error_response(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: dict = None,
        field: str = None
    ) -> JSONResponse:
        """
        Crear respuesta de error estandarizada.
        
        Args:
            status_code: Código HTTP
            error_code: Código de error
            message: Mensaje de error
            details: Detalles adicionales
            field: Campo relacionado (opcional)
            
        Returns:
            JSONResponse con error
        """
        error_data = {
            "code": error_code,
            "message": message,
            "details": details or {}
        }
        if field:
            error_data["field"] = field
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": error_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con manejo de errores."""
        try:
            return await call_next(request)
        except ValidationException as e:
            logger.warning(f"Validation error: {e.message}", error_code=e.error_code.value, field=e.field)
            return self._create_error_response(
                status.HTTP_400_BAD_REQUEST,
                e.error_code.value,
                e.message,
                e.details,
                e.field
            )
        except OpenRouterException as e:
            logger.error(f"OpenRouter error: {e.message}", error_code=e.error_code.value)
            return self._create_error_response(
                status.HTTP_502_BAD_GATEWAY,
                e.error_code.value,
                e.message,
                e.details
            )
        except (ServiceException, DogTrainingException) as e:
            logger.error(f"Service error: {e.message}", error_code=e.error_code.value)
            return self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                e.error_code.value,
                e.message,
                e.details
            )
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return self._create_error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                ErrorCode.UNEXPECTED_ERROR.value,
                "An unexpected error occurred",
                {}
            )

