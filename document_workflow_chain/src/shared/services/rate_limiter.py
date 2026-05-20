"""
Rate Limiter Service
====================

Advanced rate limiting service with multiple algorithms and backends.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import hashlib

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import log_execution


logger = logging.getLogger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithm enumeration"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitBackend(str, Enum):
    """Rate limiting backend enumeration"""
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    backend: RateLimitBackend = RateLimitBackend.MEMORY
    window_size: int = 60  # seconds
    cleanup_interval: int = 300  # seconds


@dataclass
class RateLimitResult:
    """Rate limit result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class TokenBucket:
    """Token bucket implementation"""
    capacity: int
    tokens: float
    last_refill: datetime
    refill_rate: float  # tokens per second
    
    def __post_init__(self):
        if self.last_refill is None:
            self.last_refill = DateTimeHelpers.now_utc()


@dataclass
class SlidingWindow:
    """Sliding window implementation"""
    window_size: int  # seconds
    requests: deque = field(default_factory=deque)
    
    def add_request(self, timestamp: datetime):
        """Add request to window"""
        self.requests.append(timestamp)
        self._cleanup_old_requests(timestamp)
    
    def _cleanup_old_requests(self, current_time: datetime):
        """Remove old requests outside window"""
        cutoff_time = current_time - timedelta(seconds=self.window_size)
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()
    
    def get_request_count(self, current_time: datetime) -> int:
        """Get request count in window"""
        self._cleanup_old_requests(current_time)
        return len(self.requests)


class RateLimiter:
    """Rate limiter service"""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindow] = {}
        self.fixed_windows: Dict[str, Dict[str, Any]] = {}
        self.leaky_buckets: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.is_running = False
        self.cleanup_task: Optional[asyncio.Task] = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize rate limiting backend"""
        if self.config.backend == RateLimitBackend.REDIS:
            # In a real implementation, you would initialize Redis connection
            logger.info("Redis backend initialized for rate limiting")
        elif self.config.backend == RateLimitBackend.DATABASE:
            # In a real implementation, you would initialize database connection
            logger.info("Database backend initialized for rate limiting")
        else:
            logger.info("Memory backend initialized for rate limiting")
    
    async def start(self):
        """Start the rate limiter"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        logger.info("Rate limiter started")
    
    async def stop(self):
        """Stop the rate limiter"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Rate limiter stopped")
    
    def check_rate_limit(
        self,
        identifier: str,
        limit: Optional[int] = None,
        window_size: Optional[int] = None
    ) -> RateLimitResult:
        """Check rate limit for identifier"""
        limit = limit or self.config.requests_per_minute
        window_size = window_size or self.config.window_size
        
        with self.lock:
            if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                return self._check_token_bucket(identifier, limit, window_size)
            elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                return self._check_sliding_window(identifier, limit, window_size)
            elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                return self._check_fixed_window(identifier, limit, window_size)
            elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                return self._check_leaky_bucket(identifier, limit, window_size)
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
    
    def _check_token_bucket(self, identifier: str, limit: int, window_size: int) -> RateLimitResult:
        """Check rate limit using token bucket algorithm"""
        now = DateTimeHelpers.now_utc()
        
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = TokenBucket(
                capacity=limit,
                tokens=float(limit),
                last_refill=now,
                refill_rate=limit / window_size
            )
        
        bucket = self.token_buckets[identifier]
        
        # Refill tokens based on time elapsed
        time_elapsed = (now - bucket.last_refill).total_seconds()
        tokens_to_add = time_elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now
        
        # Check if request is allowed
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            remaining = int(bucket.tokens)
            reset_time = now + timedelta(seconds=(1.0 - bucket.tokens) / bucket.refill_rate)
            
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_time=reset_time
            )
        else:
            retry_after = int((1.0 - bucket.tokens) / bucket.refill_rate)
            reset_time = now + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                reason="Rate limit exceeded"
            )
    
    def _check_sliding_window(self, identifier: str, limit: int, window_size: int) -> RateLimitResult:
        """Check rate limit using sliding window algorithm"""
        now = DateTimeHelpers.now_utc()
        
        if identifier not in self.sliding_windows:
            self.sliding_windows[identifier] = SlidingWindow(window_size=window_size)
        
        window = self.sliding_windows[identifier]
        request_count = window.get_request_count(now)
        
        if request_count < limit:
            window.add_request(now)
            remaining = limit - request_count - 1
            reset_time = now + timedelta(seconds=window_size)
            
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_time=reset_time
            )
        else:
            # Find oldest request to calculate retry after
            oldest_request = window.requests[0] if window.requests else now
            retry_after = int((oldest_request + timedelta(seconds=window_size) - now).total_seconds())
            reset_time = now + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(0, retry_after),
                reason="Rate limit exceeded"
            )
    
    def _check_fixed_window(self, identifier: str, limit: int, window_size: int) -> RateLimitResult:
        """Check rate limit using fixed window algorithm"""
        now = DateTimeHelpers.now_utc()
        window_start = now.replace(second=0, microsecond=0)
        window_key = f"{identifier}:{window_start.isoformat()}"
        
        if window_key not in self.fixed_windows:
            self.fixed_windows[window_key] = {
                "count": 0,
                "window_start": window_start,
                "window_end": window_start + timedelta(seconds=window_size)
            }
        
        window_data = self.fixed_windows[window_key]
        
        if window_data["count"] < limit:
            window_data["count"] += 1
            remaining = limit - window_data["count"]
            reset_time = window_data["window_end"]
            
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_time=reset_time
            )
        else:
            retry_after = int((window_data["window_end"] - now).total_seconds())
            
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=window_data["window_end"],
                retry_after=max(0, retry_after),
                reason="Rate limit exceeded"
            )
    
    def _check_leaky_bucket(self, identifier: str, limit: int, window_size: int) -> RateLimitResult:
        """Check rate limit using leaky bucket algorithm"""
        now = DateTimeHelpers.now_utc()
        
        if identifier not in self.leaky_buckets:
            self.leaky_buckets[identifier] = {
                "capacity": limit,
                "level": 0,
                "last_leak": now,
                "leak_rate": limit / window_size  # requests per second
            }
        
        bucket = self.leaky_buckets[identifier]
        
        # Leak requests based on time elapsed
        time_elapsed = (now - bucket["last_leak"]).total_seconds()
        leaked_requests = time_elapsed * bucket["leak_rate"]
        bucket["level"] = max(0, bucket["level"] - leaked_requests)
        bucket["last_leak"] = now
        
        # Check if request is allowed
        if bucket["level"] < bucket["capacity"]:
            bucket["level"] += 1
            remaining = int(bucket["capacity"] - bucket["level"])
            reset_time = now + timedelta(seconds=window_size)
            
            return RateLimitResult(
                allowed=True,
                limit=limit,
                remaining=remaining,
                reset_time=reset_time
            )
        else:
            retry_after = int((bucket["level"] - bucket["capacity"]) / bucket["leak_rate"])
            reset_time = now + timedelta(seconds=retry_after)
            
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(0, retry_after),
                reason="Rate limit exceeded"
            )
    
    async def _cleanup_worker(self):
        """Cleanup expired rate limit data"""
        while self.is_running:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
    
    async def _cleanup_expired_data(self):
        """Cleanup expired rate limit data"""
        now = DateTimeHelpers.now_utc()
        cutoff_time = now - timedelta(seconds=self.config.cleanup_interval * 2)
        
        with self.lock:
            # Cleanup token buckets
            expired_buckets = [
                key for key, bucket in self.token_buckets.items()
                if bucket.last_refill < cutoff_time
            ]
            for key in expired_buckets:
                del self.token_buckets[key]
            
            # Cleanup sliding windows
            expired_windows = [
                key for key, window in self.sliding_windows.items()
                if not window.requests or window.requests[-1] < cutoff_time
            ]
            for key in expired_windows:
                del self.sliding_windows[key]
            
            # Cleanup fixed windows
            expired_fixed_windows = [
                key for key, window_data in self.fixed_windows.items()
                if window_data["window_end"] < cutoff_time
            ]
            for key in expired_fixed_windows:
                del self.fixed_windows[key]
            
            # Cleanup leaky buckets
            expired_leaky_buckets = [
                key for key, bucket in self.leaky_buckets.items()
                if bucket["last_leak"] < cutoff_time
            ]
            for key in expired_leaky_buckets:
                del self.leaky_buckets[key]
            
            if expired_buckets or expired_windows or expired_fixed_windows or expired_leaky_buckets:
                logger.debug(f"Cleaned up {len(expired_buckets + expired_windows + expired_fixed_windows + expired_leaky_buckets)} expired rate limit entries")
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                "algorithm": self.config.algorithm.value,
                "backend": self.config.backend.value,
                "token_buckets": len(self.token_buckets),
                "sliding_windows": len(self.sliding_windows),
                "fixed_windows": len(self.fixed_windows),
                "leaky_buckets": len(self.leaky_buckets),
                "is_running": self.is_running,
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "burst_size": self.config.burst_size,
                    "window_size": self.config.window_size,
                    "cleanup_interval": self.config.cleanup_interval
                }
            }
    
    def reset_rate_limit(self, identifier: str):
        """Reset rate limit for identifier"""
        with self.lock:
            if identifier in self.token_buckets:
                del self.token_buckets[identifier]
            if identifier in self.sliding_windows:
                del self.sliding_windows[identifier]
            if identifier in self.leaky_buckets:
                del self.leaky_buckets[identifier]
            
            # Remove from fixed windows (more complex due to time-based keys)
            keys_to_remove = [key for key in self.fixed_windows.keys() if key.startswith(f"{identifier}:")]
            for key in keys_to_remove:
                del self.fixed_windows[key]
        
        logger.info(f"Rate limit reset for identifier: {identifier}")
    
    def clear_all_rate_limits(self):
        """Clear all rate limits"""
        with self.lock:
            self.token_buckets.clear()
            self.sliding_windows.clear()
            self.fixed_windows.clear()
            self.leaky_buckets.clear()
        
        logger.info("All rate limits cleared")


# Global rate limiter
rate_limiter = RateLimiter()


# Utility functions
async def start_rate_limiter():
    """Start the rate limiter"""
    await rate_limiter.start()


async def stop_rate_limiter():
    """Stop the rate limiter"""
    await rate_limiter.stop()


def check_rate_limit(
    identifier: str,
    limit: Optional[int] = None,
    window_size: Optional[int] = None
) -> RateLimitResult:
    """Check rate limit for identifier"""
    return rate_limiter.check_rate_limit(identifier, limit, window_size)


def get_rate_limit_stats() -> Dict[str, Any]:
    """Get rate limiter statistics"""
    return rate_limiter.get_rate_limit_stats()


def reset_rate_limit(identifier: str):
    """Reset rate limit for identifier"""
    rate_limiter.reset_rate_limit(identifier)


def clear_all_rate_limits():
    """Clear all rate limits"""
    rate_limiter.clear_all_rate_limits()


# Rate limiting decorators
def rate_limit_decorator(
    limit: int = 60,
    window_size: int = 60,
    identifier_func: Optional[Callable] = None
):
    """Rate limiting decorator"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Generate identifier
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                else:
                    identifier = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Check rate limit
                result = check_rate_limit(identifier, limit, window_size)
                
                if not result.allowed:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded. Retry after {result.retry_after} seconds",
                        headers={"Retry-After": str(result.retry_after)}
                    )
                
                return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Generate identifier
                if identifier_func:
                    identifier = identifier_func(*args, **kwargs)
                else:
                    identifier = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Check rate limit
                result = check_rate_limit(identifier, limit, window_size)
                
                if not result.allowed:
                    raise Exception(f"Rate limit exceeded. Retry after {result.retry_after} seconds")
                
                return func(*args, **kwargs)
            
            return sync_wrapper
    
    return decorator


# Common rate limiting functions
def get_client_identifier(request) -> str:
    """Get client identifier from request"""
    # Try to get user ID first
    if hasattr(request, 'state') and hasattr(request.state, 'user'):
        return f"user:{request.state.user}"
    
    # Fall back to IP address
    client_ip = getattr(request.client, 'host', 'unknown')
    return f"ip:{client_ip}"


def get_api_key_identifier(api_key: str) -> str:
    """Get identifier from API key"""
    return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"


def get_user_identifier(user_id: str) -> str:
    """Get identifier from user ID"""
    return f"user:{user_id}"




