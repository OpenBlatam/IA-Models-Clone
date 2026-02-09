"""
Structured Logging Plugin
=========================
"""

import time
import uuid
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware
from aws.core.interfaces import MiddlewarePlugin

logger = logging.getLogger(__name__)


class LoggingMiddlewarePlugin(MiddlewarePlugin):
    """Structured logging middleware plugin."""
    
    def get_name(self) -> str:
        return "logging"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        middleware_config = config.get("middleware", {})
        return middleware_config.get("enable_logging", True)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        """Setup structured logging middleware."""
        class LoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                start_time = time.time()
                request_id = str(uuid.uuid4())
                request.state.request_id = request_id
                
                logger.info(
                    "Request started",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "client_ip": request.client.host if request.client else None,
                    }
                )
                
                try:
                    response = await call_next(request)
                    process_time = time.time() - start_time
                    
                    logger.info(
                        "Request completed",
                        extra={
                            "request_id": request_id,
                            "method": request.method,
                            "path": request.url.path,
                            "status_code": response.status_code,
                            "process_time": process_time,
                        }
                    )
                    
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Process-Time"] = str(process_time)
                    return response
                    
                except Exception as e:
                    process_time = time.time() - start_time
                    logger.error(
                        "Request failed",
                        extra={
                            "request_id": request_id,
                            "method": request.method,
                            "path": request.url.path,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "process_time": process_time,
                        },
                        exc_info=True
                    )
                    raise
        
        app.add_middleware(LoggingMiddleware)
        logger.info("Structured logging enabled")
        
        return app















