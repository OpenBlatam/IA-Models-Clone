"""
Stability Middleware
====================

Middleware for improving application stability.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class StabilityMiddleware(BaseHTTPMiddleware):
    """Middleware for application stability."""

    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        request_timeout: float = 300.0,  # 5 minutes
    ):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.request_timeout = request_timeout

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with stability checks."""
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request too large. Maximum size: {self.max_request_size / (1024 * 1024):.1f}MB",
                    )
            except ValueError:
                pass

        # Add timeout protection
        start_time = time.time()
        try:
            # Wrap in timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.request_timeout
            )

            # Check if request took too long (warning only)
            duration = time.time() - start_time
            if duration > self.request_timeout * 0.8:  # Warn at 80% of timeout
                logger.warning(
                    f"Request took {duration:.2f}s (approaching timeout: {self.request_timeout}s)",
                    extra={
                        "path": request.url.path,
                        "method": request.method,
                        "duration": duration,
                    },
                )

            return response

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(
                "Request timeout",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "duration": duration,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request timeout",
            )

