"""
Middleware - Middleware
======================

Middleware personalizado para la API.
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable
from loguru import logger

from ..domain.exceptions import DomainException


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request y response con logging."""
        # Generar ID único para el request
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log del request
        start_time = time.time()
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        # Procesar request
        try:
            response = await call_next(request)
            
            # Calcular tiempo de procesamiento
            process_time = time.time() - start_time
            
            # Log del response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            # Agregar headers de respuesta
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Log del error
            process_time = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time
                },
                exc_info=True
            )
            
            # Crear response de error
            error_response = JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )
            
            error_response.headers["X-Request-ID"] = request_id
            error_response.headers["X-Process-Time"] = str(process_time)
            
            return error_response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware para manejo de errores."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con manejo de errores."""
        try:
            response = await call_next(request)
            return response
            
        except DomainException as e:
            # Manejar excepciones de dominio
            logger.error(
                f"Domain exception: {e.message}",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "error_code": e.error_code,
                    "details": e.details
                }
            )
            
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": e.error_code,
                    "message": e.message,
                    "details": e.details,
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            # Manejar excepciones generales
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "request_id": getattr(request.state, "request_id", None)
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": time.time()
                }
            )


class CORSMiddleware(BaseHTTPMiddleware):
    """Middleware CORS personalizado."""
    
    def __init__(self, app: ASGIApp, allow_origins: list = None, allow_methods: list = None, allow_headers: list = None):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["*"]
        self.allow_headers = allow_headers or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con CORS."""
        # Manejar preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response)
            return response
        
        # Procesar request normal
        response = await call_next(request)
        self._add_cors_headers(response)
        return response
    
    def _add_cors_headers(self, response: Response):
        """Agregar headers CORS a la respuesta."""
        response.headers["Access-Control-Allow-Origin"] = ", ".join(self.allow_origins)
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 horas


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting."""
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # En producción, usar Redis o similar
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Procesar request con rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Limpiar requests antiguos
        self._cleanup_old_requests(current_time)
        
        # Verificar rate limit
        if self._is_rate_limited(client_ip, current_time):
            logger.warning(
                f"Rate limit exceeded for client: {client_ip}",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "client_ip": client_ip
                }
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "retry_after": 60,
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": current_time
                }
            )
        
        # Registrar request
        self._record_request(client_ip, current_time)
        
        # Procesar request
        response = await call_next(request)
        return response
    
    def _cleanup_old_requests(self, current_time: float):
        """Limpiar requests antiguos."""
        cutoff_time = current_time - 60  # Último minuto
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip] 
                if req_time > cutoff_time
            ]
            if not self.requests[client_ip]:
                del self.requests[client_ip]
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Verificar si el cliente está rate limited."""
        if client_ip not in self.requests:
            return False
        
        return len(self.requests[client_ip]) >= self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Registrar request del cliente."""
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)




