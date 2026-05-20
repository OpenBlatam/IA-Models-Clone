"""
FastAPI dependencies for Facebook Posts API
Following dependency injection best practices
"""

from typing import Dict, Any, Optional, Generator
from contextlib import asynccontextmanager
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import time
import uuid
import logging
import structlog

from ..core.config import get_settings
from ..core.engine import FacebookPostsEngine

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Global engine instance
_engine: Optional[FacebookPostsEngine] = None


# ===== ENGINE DEPENDENCIES =====

async def get_facebook_engine() -> FacebookPostsEngine:
    """Get Facebook Posts engine instance"""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        _engine = FacebookPostsEngine(settings)
        await _engine.initialize()
    
    return _engine


@asynccontextmanager
async def get_service_lifespan():
    """Service lifespan manager for dependency injection"""
    global _engine
    
    try:
        # Initialize services
        if _engine is None:
            settings = get_settings()
            _engine = FacebookPostsEngine(settings)
            await _engine.initialize()
        
        yield _engine
        
    finally:
        # Cleanup services
        if _engine:
            await _engine.cleanup()
            _engine = None


# ===== AUTHENTICATION DEPENDENCIES =====

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    request: Request = None
) -> Dict[str, Any]:
    """
    Get current user from authentication credentials.
    In a real implementation, this would validate JWT tokens or API keys.
    """
    # Mock implementation - replace with actual authentication
    if not credentials:
        # Allow anonymous access for demo purposes
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "permissions": ["read", "write"],
            "rate_limit": 1000
        }
    
    # Mock token validation
    if credentials.credentials == "invalid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Mock user data
    return {
        "user_id": "user_123",
        "username": "demo_user",
        "permissions": ["read", "write", "admin"],
        "rate_limit": 5000
    }


async def require_permission(permission: str):
    """Dependency factory for permission requirements"""
    async def permission_checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return permission_checker


# ===== RATE LIMITING DEPENDENCIES =====

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
    
    async def check_rate_limit(self, user_id: str, limit: int, window: int) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Get user's request history
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        user_requests = self.requests[user_id]
        
        # Remove requests outside the window
        cutoff_time = current_time - window
        user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff_time]
        
        # Check if under limit
        if len(user_requests) >= limit:
            return False
        
        # Add current request
        user_requests.append(current_time)
        return True
    
    async def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limit entries"""
        cutoff_time = current_time - 3600  # 1 hour
        
        for user_id in list(self.requests.keys()):
            user_requests = self.requests[user_id]
            user_requests[:] = [req_time for req_time in user_requests if req_time > cutoff_time]
            
            if not user_requests:
                del self.requests[user_id]


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Check rate limits for the current user"""
    settings = get_settings()
    user_id = user.get("user_id", "anonymous")
    limit = user.get("rate_limit", settings.rate_limit_requests)
    window = settings.rate_limit_window
    
    is_allowed = await _rate_limiter.check_rate_limit(user_id, limit, window)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {limit} requests per {window} seconds"
        )
    
    return user


# ===== REQUEST TRACKING DEPENDENCIES =====

async def get_request_id(request: Request) -> str:
    """Get or generate request ID for tracking"""
    request_id = getattr(request.state, "request_id", None)
    
    if not request_id:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    
    return request_id


async def get_request_timing(request: Request) -> Dict[str, float]:
    """Get request timing information"""
    start_time = getattr(request.state, "start_time", time.time())
    current_time = time.time()
    
    return {
        "start_time": start_time,
        "current_time": current_time,
        "elapsed": current_time - start_time
    }


# ===== VALIDATION DEPENDENCIES =====

async def validate_post_id(post_id: str) -> str:
    """Validate post ID format"""
    if not post_id or len(post_id.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post ID cannot be empty"
        )
    
    if len(post_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Post ID is too long"
        )
    
    return post_id.strip()


async def validate_pagination_params(
    skip: int = 0,
    limit: int = 10
) -> Dict[str, int]:
    """Validate pagination parameters"""
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip parameter must be non-negative"
        )
    
    if limit <= 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 100"
        )
    
    return {"skip": skip, "limit": limit}


# ===== HEALTH CHECK DEPENDENCIES =====

async def get_health_status() -> Dict[str, Any]:
    """Get system health status"""
    try:
        engine = await get_facebook_engine()
        health_data = await engine.health_check()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": health_data.get("components", {}),
            "version": get_settings().api_version
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "version": get_settings().api_version
        }


# ===== CACHING DEPENDENCIES =====

class CacheManager:
    """Simple in-memory cache manager"""
    
    def __init__(self):
        self.cache = {}
        self.ttl = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            if time.time() < self.ttl.get(key, 0):
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.ttl[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache with TTL"""
        self.cache[key] = value
        self.ttl[key] = time.time() + ttl
    
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        self.cache.pop(key, None)
        self.ttl.pop(key, None)


# Global cache manager instance
_cache_manager = CacheManager()


async def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    return _cache_manager


# ===== UTILITY DEPENDENCIES =====

async def get_request_context(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
    request_id: str = Depends(get_request_id)
) -> Dict[str, Any]:
    """Get comprehensive request context"""
    return {
        "user": user,
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "timestamp": time.time()
    }


async def log_request(
    context: Dict[str, Any] = Depends(get_request_context)
) -> Dict[str, Any]:
    """Log request information"""
    logger.info(
        "Request received",
        method=context["method"],
        url=context["url"],
        user_id=context["user"]["user_id"],
        request_id=context["request_id"],
        client_ip=context["client_ip"]
    )
    
    return context


# ===== EXPORTS =====

__all__ = [
    # Engine dependencies
    'get_facebook_engine',
    'get_service_lifespan',
    
    # Authentication dependencies
    'get_current_user',
    'require_permission',
    
    # Rate limiting dependencies
    'check_rate_limit',
    
    # Request tracking dependencies
    'get_request_id',
    'get_request_timing',
    
    # Validation dependencies
    'validate_post_id',
    'validate_pagination_params',
    
    # Health check dependencies
    'get_health_status',
    
    # Caching dependencies
    'get_cache_manager',
    
    # Utility dependencies
    'get_request_context',
    'log_request',
]