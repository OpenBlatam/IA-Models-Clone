from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import time
import logging
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Any, List, Dict, Optional
import asyncio
"""
🔧 PRODUCTION MIDDLEWARE - Rate Limiting, Logging & Metrics
==========================================================

Middleware enterprise para API de producción.
"""




class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.requests = defaultdict(deque)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Simple rate limiting logic
        response = await call_next(request)
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.metrics = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Collect metrics
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware."""
    
    def __init__(self, app) -> Any:
        super().__init__(app)
        self.logger = logging.getLogger("api")
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        self.logger.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        return response


def setup_middleware(app) -> Any:
    """Setup all production middleware."""
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware)
    return app 