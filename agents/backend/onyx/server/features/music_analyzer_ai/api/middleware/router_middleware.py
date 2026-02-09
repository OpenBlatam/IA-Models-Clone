"""
Router middleware for common functionality
"""

from fastapi import Request, Response
from typing import Callable
import time
import logging

logger = logging.getLogger(__name__)


async def timing_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware to track request timing"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Middleware for request logging"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} - {response.status_code}")
    return response

