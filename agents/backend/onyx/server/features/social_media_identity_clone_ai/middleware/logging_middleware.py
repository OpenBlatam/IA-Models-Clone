"""
Logging Middleware para requests
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..analytics.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware para logging estructurado de requests"""
    
    async def dispatch(self, request: Request, call_next):
        metrics = get_metrics_collector()
        start_time = time.time()
        
        # Obtener información del request
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "method": method,
                "path": path,
                "query_params": query_params,
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Incrementar contador de requests
        metrics.increment("http_requests_total", tags={"method": method, "path": path})
        
        try:
            response = await call_next(request)
            
            # Calcular tiempo de respuesta
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "client_ip": client_ip
                }
            )
            
            # Registrar métricas
            metrics.histogram("http_request_duration_ms", process_time * 1000, tags={
                "method": method,
                "path": path,
                "status_code": str(response.status_code)
            })
            metrics.increment("http_requests", tags={
                "method": method,
                "status_code": str(response.status_code)
            })
            
            # Agregar header con tiempo de procesamiento
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed",
                extra={
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "process_time": process_time,
                    "client_ip": client_ip
                },
                exc_info=True
            )
            
            # Registrar métrica de error
            metrics.increment("http_requests_errors", tags={
                "method": method,
                "path": path
            })
            
            raise




