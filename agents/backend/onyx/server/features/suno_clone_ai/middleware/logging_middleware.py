"""
Middleware de logging para Suno Clone AI
"""

import logging
from fastapi import Request
from starlette.responses import Response
from .base_middleware import BaseMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Middleware para registrar todas las peticiones HTTP"""
    
    async def process_request(self, request: Request) -> Request:
        """Log incoming request"""
        self.logger.info(f"Request: {request.method} {request.url.path}")
        return request
    
    async def record_metrics(
        self,
        request: Request,
        response: Response,
        elapsed: float
    ) -> None:
        """Log response with metrics"""
        self.logger.info(
            f"Response: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {elapsed:.3f}s"
        )

