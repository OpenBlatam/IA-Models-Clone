"""
API Middleware
==============

Middleware for the FastAPI application.
"""

import logging
from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_rate_limiter

logger = logging.getLogger(__name__)


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    try:
        rate_limiter = get_rate_limiter()
    except HTTPException:
        # Rate limiter not initialized, skip
        response = await call_next(request)
        return response
    
    # Get client identifier (IP address)
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not await rate_limiter.is_allowed(client_ip):
        wait_time = await rate_limiter.get_wait_time(client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {wait_time:.2f} seconds.",
            headers={"Retry-After": str(int(wait_time))}
        )
    
    response = await call_next(request)
    return response




