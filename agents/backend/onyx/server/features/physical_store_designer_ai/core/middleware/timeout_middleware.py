"""
Timeout Middleware

Middleware for request timeout handling.
"""

import asyncio
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..logging_config import get_logger
from ..exceptions import TimeoutError

logger = get_logger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware para timeout de requests"""
    
    def __init__(self, app, timeout_seconds: float = 30.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with timeout"""
        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "timeout": self.timeout_seconds
                }
            )
            raise TimeoutError(f"{request.method} {request.url.path}", self.timeout_seconds)

