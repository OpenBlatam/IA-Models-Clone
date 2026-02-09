"""
Request ID Middleware
=====================
Middleware para agregar request IDs a todas las requests.
"""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.request_id import get_request_id
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware para agregar request IDs."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request agregando request ID."""
        # Obtener o generar request ID
        request_id = get_request_id(request)
        
        # Agregar al state
        request.state.request_id = request_id
        
        # Agregar header a la respuesta
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response

