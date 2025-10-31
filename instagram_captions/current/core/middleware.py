"""
Middleware for Instagram Captions API v10.0

Core middleware functionality.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Logging middleware for request/response tracking."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {response.status_code} - {response_time:.3f}s")
    
    return response

async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """Rate limiting middleware."""
    # Extract client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Simple rate limiting check (you can integrate with RateLimiter class)
    # For now, just pass through
    return await call_next(request)

async def security_middleware(request: Request, call_next: Callable) -> Response:
    """Security middleware for basic security checks."""
    # Basic security checks
    user_agent = request.headers.get("user-agent", "")
    
    # Check for suspicious user agents
    suspicious_patterns = ["sqlmap", "nikto", "nmap", "dirb"]
    if any(pattern in user_agent.lower() for pattern in suspicious_patterns):
        logger.warning(f"Suspicious user agent detected: {user_agent}")
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied"}
        )
    
    # Check for suspicious headers
    if "x-forwarded-for" in request.headers:
        # Log but allow (common in production)
        logger.info(f"X-Forwarded-For: {request.headers['x-forwarded-for']}")
    
    return await call_next(request)






