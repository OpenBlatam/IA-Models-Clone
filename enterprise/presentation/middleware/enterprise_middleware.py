from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import uuid
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from ...shared.config import EnterpriseConfig
from ...core.interfaces.metrics_interface import IMetricsService
from ...core.interfaces.rate_limit_interface import IRateLimitService
from ...shared.utils import safe_get_client_ip, create_response_headers
import logging
from typing import Any, List, Dict, Optional
import asyncio
"""
Enterprise Middleware Stack
===========================

Collection of enterprise-grade middleware components.
"""


logger = logging.getLogger(__name__)


class EnterpriseMiddlewareStack:
    """Collection of enterprise middleware components."""
    
    def __init__(self, config: EnterpriseConfig, 
                 metrics_service: IMetricsService,
                 rate_limit_service: IRateLimitService):
        
    """__init__ function."""
self.config = config
        self.metrics_service = metrics_service
        self.rate_limit_service = rate_limit_service
    
    async def request_id_middleware(self, request: Request, call_next):
        """Add unique request ID for tracing."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    async def performance_monitoring_middleware(self, request: Request, call_next):
        """Monitor request performance and record metrics."""
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        duration = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{duration:.4f}"
        
        # Record metrics
        self.metrics_service.record_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        # Log slow requests
        if duration > 1.0:  # Log requests taking more than 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} - {duration:.4f}s"
            )
        
        return response
    
    async def security_headers_middleware(self, request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        
        security_headers = self.config.get_security_headers()
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    async def rate_limiting_middleware(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        client_ip = safe_get_client_ip(request)
        identifier = f"{client_ip}:{request.url.path}"
        
        # Check rate limit
        rate_limit_info = await self.rate_limit_service.is_allowed(identifier)
        
        if not rate_limit_info.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded",
                        "retry_after": rate_limit_info.retry_after,
                        "request_id": getattr(request.state, "request_id", "unknown")
                    }
                },
                headers=rate_limit_info.get_headers()
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        rate_limit_headers = rate_limit_info.get_headers()
        for header, value in rate_limit_headers.items():
            response.headers[header] = value
        
        return response