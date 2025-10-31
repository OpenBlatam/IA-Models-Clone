from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer
import redis.asyncio as redis
from core import InstagramCaptionsEngine
from gmt_system import SimplifiedGMTSystem
from service import InstagramCaptionsService
    from config import config
from typing import Any, List, Dict, Optional
"""
FastAPI Dependencies for Instagram Captions API.

Optimized dependency injection with caching, resource management, and error handling.
"""




logger = logging.getLogger(__name__)

# Global instances cache
_instances_cache: Dict[str, Any] = {}
_redis_client: Optional[redis.Redis] = None

# Security
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_settings():
    """Get application settings with caching."""
    return config


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for caching with connection pooling."""
    global _redis_client
    
    if _redis_client is None:
        try:
            settings = get_settings()
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            
            _redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await _redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Continuing without cache.")
            _redis_client = None
    
    return _redis_client


async def get_captions_engine() -> InstagramCaptionsEngine:
    """Get Instagram Captions Engine with singleton pattern."""
    cache_key = "captions_engine"
    
    if cache_key not in _instances_cache:
        try:
            engine = InstagramCaptionsEngine()
            _instances_cache[cache_key] = engine
            logger.debug("Instagram Captions Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Captions Engine: {e}")
            raise HTTPException(
                status_code=503,
                detail="Caption generation service unavailable"
            )
    
    return _instances_cache[cache_key]


async def get_gmt_system() -> SimplifiedGMTSystem:
    """Get GMT System with singleton pattern."""
    cache_key = "gmt_system"
    
    if cache_key not in _instances_cache:
        try:
            gmt_system = SimplifiedGMTSystem()
            _instances_cache[cache_key] = gmt_system
            logger.debug("GMT System initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GMT System: {e}")
            raise HTTPException(
                status_code=503,
                detail="GMT system unavailable"
            )
    
    return _instances_cache[cache_key]


async def get_captions_service() -> InstagramCaptionsService:
    """Get Instagram Captions Service with dependency injection."""
    cache_key = "captions_service"
    
    if cache_key not in _instances_cache:
        try:
            service = InstagramCaptionsService()
            _instances_cache[cache_key] = service
            logger.debug("Instagram Captions Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Captions Service: {e}")
            raise HTTPException(
                status_code=503,
                detail="Caption service unavailable"
            )
    
    return _instances_cache[cache_key]


async async def get_request_context(request: Request) -> Dict[str, Any]:
    """Extract and validate request context."""
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown"),
        "request_id": request.headers.get("x-request-id"),
        "timestamp": asyncio.get_event_loop().time()
    }


class CacheManager:
    """Centralized cache management for API responses."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.default_ttl = 300  # 5 minutes
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value."""
        if not self.redis_client:
            return None
        
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set cached value."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.setex(
                key, 
                ttl or self.default_ttl, 
                value
            )
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False
    
    def generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate consistent cache key."""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return ":".join(key_parts)


async def get_cache_manager(
    redis_client: Optional[redis.Redis] = Depends(get_redis_client)
) -> CacheManager:
    """Get cache manager with Redis client."""
    return CacheManager(redis_client)


class RateLimiter:
    """Simple rate limiting for API endpoints."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.default_limit = 100  # requests per window
        self.default_window = 3600  # 1 hour
    
    async def is_allowed(self, key: str, limit: Optional[int] = None, 
                        window: Optional[int] = None) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        if not self.redis_client:
            return True, {"rate_limit_active": False}
        
        limit = limit or self.default_limit
        window = window or self.default_window
        
        try:
            current = await self.redis_client.get(key)
            
            if current is None:
                await self.redis_client.setex(key, window, 1)
                return True, {
                    "rate_limit_active": True,
                    "requests_remaining": limit - 1,
                    "reset_time": window
                }
            
            current_count = int(current)
            
            if current_count >= limit:
                ttl = await self.redis_client.ttl(key)
                return False, {
                    "rate_limit_active": True,
                    "requests_remaining": 0,
                    "reset_time": ttl
                }
            
            await self.redis_client.incr(key)
            return True, {
                "rate_limit_active": True,
                "requests_remaining": limit - current_count - 1,
                "reset_time": await self.redis_client.ttl(key)
            }
            
        except Exception as e:
            logger.warning(f"Rate limit check failed for key {key}: {e}")
            return True, {"rate_limit_active": False, "error": str(e)}


async def get_rate_limiter(
    redis_client: Optional[redis.Redis] = Depends(get_redis_client)
) -> RateLimiter:
    """Get rate limiter with Redis client."""
    return RateLimiter(redis_client)


async def check_rate_limit(
    request: Request,
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> Dict[str, Any]:
    """Check rate limit for current request."""
    if not request.client:
        return {"rate_limit_active": False}
    
    client_ip = request.client.host
    endpoint = request.url.path
    rate_key = f"rate_limit:{client_ip}:{endpoint}"
    
    is_allowed, rate_info = await rate_limiter.is_allowed(rate_key)
    
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "reset_time": rate_info.get("reset_time", 0)
            }
        )
    
    return rate_info


class HealthChecker:
    """System health checking for dependencies."""
    
    def __init__(self) -> Any:
        self.checks = {}
    
    async def check_engine_health(self, engine: InstagramCaptionsEngine) -> Dict[str, Any]:
        """Check Instagram Captions Engine health."""
        try:
            # Simple health check - try to analyze a test caption
            test_result = engine.analyze_quality("Test caption for health check")
            
            return {
                "status": "healthy",
                "response_time_ms": 1,  # Placeholder
                "last_check": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": asyncio.get_event_loop().time()
            }
    
    async def check_gmt_health(self, gmt_system: SimplifiedGMTSystem) -> Dict[str, Any]:
        """Check GMT System health."""
        try:
            # Simple health check - get timezone insights
            insights = gmt_system.get_timezone_insights("UTC")
            
            return {
                "status": "healthy",
                "timezones_supported": len(gmt_system.cultural_adapter.cultural_profiles),
                "last_check": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": asyncio.get_event_loop().time()
            }
    
    async def check_redis_health(self, redis_client: Optional[redis.Redis]) -> Dict[str, Any]:
        """Check Redis health."""
        if not redis_client:
            return {
                "status": "disabled",
                "message": "Redis not configured"
            }
        
        try:
            await redis_client.ping()
            return {
                "status": "healthy",
                "connection": "active",
                "last_check": asyncio.get_event_loop().time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": asyncio.get_event_loop().time()
            }


async def get_health_checker() -> HealthChecker:
    """Get health checker instance."""
    return HealthChecker()


async async def validate_request_size(request: Request) -> None:
    """Validate request size to prevent abuse."""
    content_length = request.headers.get("content-length")
    
    if content_length:
        size = int(content_length)
        max_size = 1024 * 1024  # 1MB
        
        if size > max_size:
            raise HTTPException(
                status_code=413,
                detail="Request too large"
            )


# Cleanup function for graceful shutdown
async def cleanup_dependencies():
    """Cleanup function for graceful shutdown."""
    global _redis_client, _instances_cache
    
    try:
        if _redis_client:
            await _redis_client.close()
            logger.info("Redis connection closed")
        
        _instances_cache.clear()
        logger.info("Dependencies cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Context manager for dependency lifecycle
@asynccontextmanager
async def dependency_lifespan():
    """Context manager for dependency lifecycle management."""
    try:
        # Initialize dependencies
        logger.info("Initializing API dependencies...")
        yield
    finally:
        # Cleanup on shutdown
        await cleanup_dependencies() 