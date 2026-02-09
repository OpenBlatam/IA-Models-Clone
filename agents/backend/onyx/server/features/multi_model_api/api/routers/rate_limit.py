"""
Rate limiting router for Multi-Model API
Handles rate limit information endpoints
"""

from fastapi import APIRouter, Depends, Query, Request
from ...api.dependencies import check_rate_limit
from ...core.rate_limiter import get_rate_limiter

router = APIRouter(prefix="/multi-model/rate-limit", tags=["Rate Limiting"])


@router.get("/info")
async def get_rate_limit_info(
    request: Request,
    endpoint: str = Query("execute", description="Endpoint name")
):
    """
    Get current rate limit information for a client
    
    Args:
        request: FastAPI Request object
        endpoint: Endpoint name to check rate limit for
        
    Returns:
        Rate limit information including:
        - Current limit
        - Remaining requests
        - Reset time
        - Retry after (if limited)
    """
    rate_limiter = get_rate_limiter()
    rate_limit_info = await check_rate_limit(request, endpoint)
    
    return {
        "allowed": rate_limit_info.allowed,
        "limit": rate_limit_info.limit,
        "remaining": rate_limit_info.remaining,
        "reset_at": rate_limit_info.reset_at,
        "retry_after": rate_limit_info.retry_after
    }




