"""
Dependency injection for FastAPI endpoints.
Centralized management of dependencies like tutor instance, rate limiter, etc.
"""

from typing import Optional
from fastapi import Request, Depends
from functools import lru_cache

from ..core.maintenance_tutor import RobotMaintenanceTutor
from ..core.conversation_manager import ConversationManager
from ..config.maintenance_config import MaintenanceConfig
from ..utils.rate_limiter import RateLimiter


# Application-level singletons (initialized once)
_tutor_instance: Optional[RobotMaintenanceTutor] = None
_conversation_manager: Optional[ConversationManager] = None
_rate_limiter: Optional[RateLimiter] = None


@lru_cache()
def get_config() -> MaintenanceConfig:
    """Get cached configuration instance."""
    return MaintenanceConfig()


def get_tutor() -> RobotMaintenanceTutor:
    """
    Dependency to get tutor instance.
    Uses singleton pattern but allows for testing with dependency override.
    """
    global _tutor_instance
    if _tutor_instance is None:
        config = get_config()
        _tutor_instance = RobotMaintenanceTutor(config)
    return _tutor_instance


def get_conversation_manager() -> ConversationManager:
    """Dependency to get conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


def get_rate_limiter() -> RateLimiter:
    """Dependency to get rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
    return _rate_limiter


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    return client_ip


def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> str:
    """
    Dependency to check rate limit.
    Raises HTTPException if rate limit exceeded.
    """
    from fastapi import HTTPException, status
    
    identifier = get_client_identifier(request)
    allowed, error_msg = rate_limiter.is_allowed(identifier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_msg,
            headers={"Retry-After": "60"}
        )
    return identifier


# Functions to reset singletons (useful for testing)
def reset_tutor_instance():
    """Reset tutor instance (for testing)."""
    global _tutor_instance
    _tutor_instance = None


def reset_conversation_manager():
    """Reset conversation manager (for testing)."""
    global _conversation_manager
    _conversation_manager = None


def reset_rate_limiter():
    """Reset rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None






