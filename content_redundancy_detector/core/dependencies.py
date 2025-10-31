"""
Core Dependencies - Dependency Injection container
Centralized dependency management for the application
"""

from typing import Optional
from fastapi import Request, Depends, HTTPException, status
from core.config import settings

# Cache for dependency instances
_dependency_cache = {}


def get_settings():
    """Get application settings"""
    from core.config import get_settings
    return get_settings()


def get_cache_manager():
    """Get cache manager instance"""
    if "cache_manager" not in _dependency_cache:
        from cache_manager import cache_manager
        _dependency_cache["cache_manager"] = cache_manager
    return _dependency_cache["cache_manager"]


def get_rate_limiter():
    """Get rate limiter instance"""
    if "rate_limiter" not in _dependency_cache:
        from rate_limiter_advanced import rate_limiter
        _dependency_cache["rate_limiter"] = rate_limiter
    return _dependency_cache["rate_limiter"]


def get_health_checker():
    """Get health checker instance"""
    if "health_checker" not in _dependency_cache:
        from health_checks_advanced import health_checker
        _dependency_cache["health_checker"] = health_checker
    return _dependency_cache["health_checker"]


def get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request (placeholder - implement based on your auth system)"""
    # This should be implemented based on your authentication system
    # For example, extract from JWT token or session
    return request.headers.get("X-User-ID")


def get_ip_address(request: Request) -> str:
    """Extract IP address from request"""
    # Check for forwarded IP first (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client host
    if request.client:
        return request.client.host
    
    return "unknown"


def get_user_context(request: Request) -> dict:
    """Get user context from request"""
    return {
        "user_id": get_user_id(request),
        "ip_address": get_ip_address(request),
        "user_agent": request.headers.get("User-Agent", "unknown")
    }


async def verify_rate_limit(
    request: Request,
    user_id: Optional[str] = Depends(get_user_id),
    ip_address: str = Depends(get_ip_address),
    rate_limiter_instance = Depends(get_rate_limiter)
):
    """Dependency to verify rate limiting"""
    allowed, info = rate_limiter_instance.is_allowed(
        user_id=user_id,
        ip_address=ip_address,
        config_name="default"
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": "Rate limit exceeded",
                "retry_after": info.get("retry_after", 60),
                "limit": info.get("limit", 60),
                "window": info.get("window", "minute")
            },
            headers={
                "Retry-After": str(info.get("retry_after", 60)),
                "X-RateLimit-Limit": str(info.get("limit", 60)),
                "X-RateLimit-Remaining": str(info.get("remaining", 0)),
                "X-RateLimit-Reset": str(int(info.get("reset_time", 0)))
            }
        )
    
    return info





