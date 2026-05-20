"""Logging middleware with structured logging"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import json
from typing import Dict, Any

from utils.logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses with structured context"""
    
    def _get_request_context(self, request: Request) -> Dict[str, Any]:
        """Extract request context for logging"""
        request_id = getattr(request.state, "request_id", "unknown")
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        context = {
            "request_id": request_id,
            "client_host": client_host,
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params) if request.query_params else {},
            "user_agent": user_agent,
        }
        
        # Add content type if present
        if "content-type" in request.headers:
            context["content_type"] = request.headers["content-type"]
        
        # Add content length if present
        if "content-length" in request.headers:
            try:
                context["content_length"] = int(request.headers["content-length"])
            except ValueError:
                pass
        
        return context
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response with structured context"""
        start_time = time.time()
        context = self._get_request_context(request)
        
        # Skip logging for health and metrics endpoints to reduce noise
        skip_logging = request.url.path in ["/health", "/ready", "/metrics", "/metrics/info"]
        
        if not skip_logging:
            logger.bind(**context).info(
                f"→ {request.method} {request.url.path}",
                **context
            )
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Add response context
            response_context = {
                **context,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "process_time": round(process_time, 4),
            }
            
            if not skip_logging:
                log_level = "warning" if process_time > 1.0 else "info"
                log_level = "error" if response.status_code >= 500 else log_level
                
                getattr(logger.bind(**response_context), log_level)(
                    f"← {request.method} {request.url.path} {response.status_code} "
                    f"({process_time_ms}ms)",
                    **response_context
                )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            error_context = {
                **context,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "process_time_ms": round(process_time * 1000, 2),
                "process_time": round(process_time, 4),
            }
            
            logger.bind(**error_context).error(
                f"✗ {request.method} {request.url.path} - {type(e).__name__}: {str(e)}",
                **error_context,
                exc_info=True
            )
            raise








