"""
Advanced Rate Limiting with Multiple Strategies
Supports per-user, per-IP, and global rate limiting with Redis backend
"""

import time
import json
import hashlib
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    window_size: int = 60  # seconds
    cleanup_interval: int = 300  # seconds


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_client = None
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Default configurations
        self.configs["default"] = RateLimitConfig()
        self.configs["strict"] = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000
        )
        self.configs["premium"] = RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=2000,
            requests_per_day=20000
        )
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis rate limiter initialized")
            except RedisError as e:
                logger.warning(f"Redis not available for rate limiting: {e}")
                self.redis_client = None
    
    def _get_identifier(self, user_id: Optional[str] = None, ip_address: Optional[str] = None) -> str:
        """Generate unique identifier for rate limiting"""
        if user_id:
            return f"user:{user_id}"
        elif ip_address:
            return f"ip:{ip_address}"
        else:
            return "global"
    
    def _get_key(self, identifier: str, config_name: str, window_type: str) -> str:
        """Generate Redis/memory key for rate limiting"""
        current_time = int(time.time())
        if window_type == "minute":
            window = current_time // 60
        elif window_type == "hour":
            window = current_time // 3600
        elif window_type == "day":
            window = current_time // 86400
        else:
            window = current_time
        
        return f"rate_limit:{config_name}:{identifier}:{window_type}:{window}"
    
    def _is_allowed_fixed_window(self, identifier: str, config: RateLimitConfig, 
                                window_type: str) -> Tuple[bool, Dict[str, Any]]:
        """Fixed window rate limiting"""
        key = self._get_key(identifier, "default", window_type)
        current_time = time.time()
        
        # Get current count
        if self.redis_client:
            try:
                count = int(self.redis_client.get(key) or 0)
                ttl = self.redis_client.ttl(key)
            except RedisError:
                count = self.memory_store.get(key, {}).get("count", 0)
                ttl = self.memory_store.get(key, {}).get("expires_at", 0) - current_time
        else:
            entry = self.memory_store.get(key, {})
            count = entry.get("count", 0)
            ttl = entry.get("expires_at", 0) - current_time
        
        # Check limits
        if window_type == "minute":
            limit = config.requests_per_minute
            window_ttl = 60
        elif window_type == "hour":
            limit = config.requests_per_hour
            window_ttl = 3600
        elif window_type == "day":
            limit = config.requests_per_day
            window_ttl = 86400
        else:
            limit = config.requests_per_minute
            window_ttl = 60
        
        if count >= limit:
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset_time": current_time + window_ttl,
                "retry_after": max(1, int(ttl)) if ttl > 0 else window_ttl
            }
        
        # Increment counter
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_ttl)
                pipe.execute()
            except RedisError:
                # Fallback to memory
                if key not in self.memory_store:
                    self.memory_store[key] = {"count": 0, "expires_at": current_time + window_ttl}
                self.memory_store[key]["count"] += 1
        else:
            if key not in self.memory_store:
                self.memory_store[key] = {"count": 0, "expires_at": current_time + window_ttl}
            self.memory_store[key]["count"] += 1
        
        return True, {
            "limit": limit,
            "remaining": limit - count - 1,
            "reset_time": current_time + window_ttl,
            "retry_after": 0
        }
    
    def _is_allowed_sliding_window(self, identifier: str, config: RateLimitConfig, 
                                  window_type: str) -> Tuple[bool, Dict[str, Any]]:
        """Sliding window rate limiting"""
        current_time = time.time()
        window_size = config.window_size
        
        if window_type == "minute":
            limit = config.requests_per_minute
        elif window_type == "hour":
            limit = config.requests_per_hour
            window_size = 3600
        elif window_type == "day":
            limit = config.requests_per_day
            window_size = 86400
        else:
            limit = config.requests_per_minute
        
        # Use sorted set for sliding window
        key = f"rate_limit:sliding:{identifier}:{window_type}"
        
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                # Remove old entries
                pipe.zremrangebyscore(key, 0, current_time - window_size)
                # Count current entries
                pipe.zcard(key)
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                # Set expiry
                pipe.expire(key, window_size)
                results = pipe.execute()
                
                current_count = results[1]
                
                if current_count >= limit:
                    return False, {
                        "limit": limit,
                        "remaining": 0,
                        "reset_time": current_time + window_size,
                        "retry_after": window_size
                    }
                
                return True, {
                    "limit": limit,
                    "remaining": limit - current_count - 1,
                    "reset_time": current_time + window_size,
                    "retry_after": 0
                }
                
            except RedisError:
                # Fallback to memory-based sliding window
                pass
        
        # Memory-based sliding window
        if key not in self.memory_store:
            self.memory_store[key] = []
        
        # Remove old entries
        cutoff_time = current_time - window_size
        self.memory_store[key] = [
            timestamp for timestamp in self.memory_store[key]
            if timestamp > cutoff_time
        ]
        
        if len(self.memory_store[key]) >= limit:
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset_time": current_time + window_size,
                "retry_after": window_size
            }
        
        # Add current request
        self.memory_store[key].append(current_time)
        
        return True, {
            "limit": limit,
            "remaining": limit - len(self.memory_store[key]),
            "reset_time": current_time + window_size,
            "retry_after": 0
        }
    
    def is_allowed(self, user_id: Optional[str] = None, ip_address: Optional[str] = None,
                   config_name: str = "default") -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits"""
        identifier = self._get_identifier(user_id, ip_address)
        config = self.configs.get(config_name, self.configs["default"])
        
        # Check all time windows
        windows = ["minute", "hour", "day"]
        
        for window in windows:
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                allowed, info = self._is_allowed_fixed_window(identifier, config, window)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                allowed, info = self._is_allowed_sliding_window(identifier, config, window)
            else:
                # Default to fixed window
                allowed, info = self._is_allowed_fixed_window(identifier, config, window)
            
            if not allowed:
                return False, {
                    "allowed": False,
                    "window": window,
                    "limit": info["limit"],
                    "remaining": info["remaining"],
                    "reset_time": info["reset_time"],
                    "retry_after": info["retry_after"],
                    "strategy": config.strategy.value
                }
        
        return True, {
            "allowed": True,
            "strategy": config.strategy.value,
            "limits": {
                "minute": config.requests_per_minute,
                "hour": config.requests_per_hour,
                "day": config.requests_per_day
            }
        }
    
    def get_usage(self, user_id: Optional[str] = None, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Get current usage statistics"""
        identifier = self._get_identifier(user_id, ip_address)
        current_time = time.time()
        
        usage = {
            "identifier": identifier,
            "current_time": current_time,
            "windows": {}
        }
        
        for window_type in ["minute", "hour", "day"]:
            key = self._get_key(identifier, "default", window_type)
            
            if self.redis_client:
                try:
                    count = int(self.redis_client.get(key) or 0)
                    ttl = self.redis_client.ttl(key)
                except RedisError:
                    count = self.memory_store.get(key, {}).get("count", 0)
                    ttl = 0
            else:
                entry = self.memory_store.get(key, {})
                count = entry.get("count", 0)
                ttl = entry.get("expires_at", 0) - current_time
            
            usage["windows"][window_type] = {
                "count": count,
                "ttl": max(0, ttl)
            }
        
        return usage
    
    def reset_limits(self, user_id: Optional[str] = None, ip_address: Optional[str] = None) -> bool:
        """Reset rate limits for identifier"""
        identifier = self._get_identifier(user_id, ip_address)
        
        try:
            if self.redis_client:
                # Delete all keys for this identifier
                pattern = f"rate_limit:*:{identifier}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                # Remove from memory store
                keys_to_remove = [
                    key for key in self.memory_store.keys()
                    if identifier in key
                ]
                for key in keys_to_remove:
                    del self.memory_store[key]
            
            return True
        except Exception as e:
            logger.error(f"Error resetting limits: {e}")
            return False
    
    def add_config(self, name: str, config: RateLimitConfig) -> None:
        """Add custom rate limit configuration"""
        self.configs[name] = config
        logger.info(f"Added rate limit config: {name}")
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from memory store"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.memory_store.items():
            if isinstance(entry, dict) and "expires_at" in entry:
                if entry["expires_at"] < current_time:
                    expired_keys.append(key)
            elif isinstance(entry, list) and key.startswith("rate_limit:sliding:"):
                # For sliding window, remove old timestamps
                cutoff_time = current_time - 86400  # 24 hours
                self.memory_store[key] = [
                    timestamp for timestamp in entry
                    if timestamp > cutoff_time
                ]
                if not self.memory_store[key]:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_store[key]
        
        return len(expired_keys)


# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()


def rate_limit(config_name: str = "default"):
    """Decorator for rate limiting endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id and ip_address from request if available
            user_id = None
            ip_address = None
            
            # Try to get from FastAPI request object
            for arg in args:
                if hasattr(arg, 'client') and hasattr(arg.client, 'host'):
                    ip_address = arg.client.host
                if hasattr(arg, 'user') and hasattr(arg.user, 'id'):
                    user_id = arg.user.id
                break
            
            # Check rate limit
            allowed, info = rate_limiter.is_allowed(
                user_id=user_id,
                ip_address=ip_address,
                config_name=config_name
            )
            
            if not allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
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
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator





