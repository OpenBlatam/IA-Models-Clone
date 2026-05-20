"""
Middleware - Middleware para la API
====================================

Middleware para logging, autenticación, rate limiting, etc.
Con soporte para CloudWatch en AWS.
"""

import os
import time
import logging
import json
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Intentar importar CloudWatch logger
try:
    from .aws_adapter import CloudWatchLogger
    HAS_CLOUDWATCH = True
except ImportError:
    HAS_CLOUDWATCH = False
    CloudWatchLogger = None

# CloudWatch logger global (lazy initialization)
_cloudwatch_logger: Optional[CloudWatchLogger] = None


def get_cloudwatch_logger() -> Optional[CloudWatchLogger]:
    """Obtener o crear CloudWatch logger."""
    global _cloudwatch_logger
    
    if not HAS_CLOUDWATCH:
        return None
    
    if _cloudwatch_logger is None and os.getenv("AWS_REGION"):
        try:
            _cloudwatch_logger = CloudWatchLogger(
                log_group=os.getenv("CLOUDWATCH_LOG_GROUP", "/aws/cursor-agent-24-7"),
                region=os.getenv("AWS_REGION", "us-east-1")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize CloudWatch logger: {e}")
    
    return _cloudwatch_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging de requests con soporte CloudWatch"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        log_data = {
            "method": request.method,
            "path": str(request.url.path),
            "query": str(request.url.query),
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        logger.info(f"📥 {request.method} {request.url.path}")
        
        # Enviar a CloudWatch si está disponible
        cw_logger = get_cloudwatch_logger()
        if cw_logger:
            cw_logger.log(
                f"Request: {request.method} {request.url.path}",
                level="INFO",
                extra=log_data
            )
        
        # Procesar request
        error = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            error = str(e)
            # Re-lanzar excepción después de logging
            raise
        finally:
            # Calcular tiempo
            process_time = time.time() - start_time
            
            # Log response
            response_data = {
                **log_data,
                "status_code": status_code,
                "process_time": process_time,
            }
            if error:
                response_data["error"] = error
            
            log_level = "ERROR" if status_code >= 500 else "WARNING" if status_code >= 400 else "INFO"
            logger.log(
                getattr(logging, log_level, logging.INFO),
                f"📤 {request.method} {request.url.path} - "
                f"Status: {status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Enviar a CloudWatch
            if cw_logger:
                cw_logger.log(
                    f"Response: {request.method} {request.url.path} - {status_code}",
                    level=log_level,
                    extra=response_data
                )
        
        # Agregar headers (solo si response existe)
        if 'response' in locals():
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request.headers.get("x-request-id", "unknown")
            
            # Registrar métricas Prometheus
            try:
                from .observability import record_http_metric
                request_size = len(request.headers.get("content-length", "0") or "0")
                record_http_metric(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status=status_code,
                    duration=process_time,
                    size=int(request_size) if request_size.isdigit() else 0
                )
            except Exception as e:
                logger.debug(f"Failed to record metrics: {e}")
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware para rate limiting"""
    
    def __init__(self, app: ASGIApp, max_requests: int = 100, window: float = 60.0):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window
        self.requests = {}  # client_ip -> [timestamps]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        
        # Verificar rate limit
        if not self._check_rate_limit(client_ip):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            self.max_requests - len(self.requests.get(client_ip, []))
        )
        
        return response
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Verificar rate limit para IP"""
        import time
        now = time.time()
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Limpiar requests antiguos
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if now - ts < self.window
        ]
        
        # Verificar límite
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Agregar request actual
        self.requests[client_ip].append(now)
        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware para headers de seguridad"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Agregar headers de seguridad
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware para manejo de errores"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            from fastapi import HTTPException
            from fastapi.responses import JSONResponse
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e),
                    "path": str(request.url.path)
                }
            )



