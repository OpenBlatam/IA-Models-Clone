"""Advanced request logging middleware."""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class AdvancedRequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""
    
    def __init__(self, app: ASGIApp, log_body: bool = False, log_headers: bool = False):
        super().__init__(app)
        self.log_body = log_body
        self.log_headers = log_headers
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
        }
        
        if self.log_headers:
            log_data["headers"] = dict(request.headers)
        
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    log_data["body_size"] = len(body)
                    if len(body) < 1000:  # Only log small bodies
                        log_data["body_preview"] = body[:100].decode("utf-8", errors="ignore")
            except Exception:
                pass
        
        logger.info("Request received", extra={"request": log_data})
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            response_data = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time": process_time,
                "response_size": response.headers.get("content-length", "unknown"),
            }
            
            logger.info(
                "Request completed",
                extra={
                    "request": log_data,
                    "response": response_data
                }
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request": log_data,
                    "error": str(e),
                    "process_time": process_time
                },
                exc_info=True
            )
            raise


