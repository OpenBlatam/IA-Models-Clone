"""
Enhanced Rate Limiting Module for Blaze AI.

This module provides comprehensive rate limiting capabilities with multiple
algorithms, distributed support, and adaptive throttling.
"""

import asyncio
import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import redis.asyncio as redis
from pydantic import BaseModel, Field

from core.config import RateLimitingConfig
from core.exceptions import RateLimitExceededError
from core.logging import get_logger


# ============================================================================
# RATE LIMITING MODELS AND CONFIGURATION
# ============================================================================

class RateLimitContext(BaseModel):
    """Context information for rate limiting decisions."""
    user_id: Optional[str] = Field(None, description="User identifier")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    endpoint: Optional[str] = Field(None, description="API endpoint")
    method: Optional[str] = Field(None, description="HTTP method")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Session identifier")
    api_key: Optional[str] = Field(None, description="API key")
    priority: int = Field(default=0, description="Request priority (higher = more important)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")


class RateLimitRule(BaseModel):
    """Configuration rule for rate limiting."""
    name: str = Field(..., description="Rule name")
    context_keys: List[str] = Field(..., description="Context keys to use for limiting")
    limits: Dict[str, int] = Field(..., description="Rate limits by time window")
    algorithm: str = Field(default="sliding_window", description="Rate limiting algorithm")
    burst_limit: int = Field(default=0, description="Burst limit allowance")
    priority_boost: Dict[int, float] = Field(default_factory=dict, description="Priority-based rate boosts")
    enabled: bool = Field(default=True, description="Whether the rule is enabled")
    description: str = Field(default="", description="Rule description")


class RateLimitResult(BaseModel):
    """Result of a rate limiting check."""
    allowed: bool = Field(..., description="Whether the request is allowed")
    remaining: int = Field(..., description="Remaining requests allowed")
    reset_time: Optional[datetime] = Field(None, description="When the limit resets")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
    limit: int = Field(..., description="Total limit for the time window")
    context_key: str = Field(..., description="Context key used for limiting")
    rule_name: str = Field(..., description="Name of the rate limit rule")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result data")


class RateLimitStats(BaseModel):
    """Statistics for rate limiting operations."""
    total_requests: int = Field(0, description="Total requests processed")
    allowed_requests: int = Field(0, description="Requests allowed")
    blocked_requests: int = Field(0, description="Requests blocked")
    rate_limited_requests: int = Field(0, description="Requests rate limited")
    context_keys: Dict[str, int] = Field(default_factory=dict, description="Requests by context key")
    rules_used: Dict[str, int] = Field(default_factory=dict, description="Rules used for limiting")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last statistics update")


class ThrottleInfo(BaseModel):
    """Information about throttling for a request."""
    should_throttle: bool = Field(..., description="Whether the request should be throttled")
    delay_seconds: float = Field(0.0, description="Delay to apply in seconds")
    reason: Optional[str] = Field(None, description="Reason for throttling")
    adaptive_factor: float = Field(1.0, description="Adaptive throttling factor")


# ============================================================================
# RATE LIMITING INTERFACES AND BASE CLASSES
# ============================================================================

class RateLimitAlgorithm(ABC):
    """Abstract base class for rate limiting algorithms."""
    
    @abstractmethod
    async def check_limit(self, context_key: str, limit: int, window_seconds: int) -> Tuple[bool, int, Optional[datetime]]:
        """
        Check if a request is within the rate limit.
        
        Args:
            context_key: Unique identifier for the context
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        pass
    
    @abstractmethod
    async def record_request(self, context_key: str, timestamp: Optional[float] = None) -> None:
        """Record a request for rate limiting purposes."""
        pass
    
    @abstractmethod
    async def get_stats(self, context_key: str) -> Dict[str, Any]:
        """Get statistics for a context key."""
        pass


class RateLimitStorage(ABC):
    """Abstract base class for rate limit storage backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from storage."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in storage with optional TTL."""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a value in storage."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from storage."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        pass


class ThrottlingStrategy(ABC):
    """Abstract base class for throttling strategies."""
    
    @abstractmethod
    async def calculate_throttle(self, context: RateLimitContext, current_load: float) -> ThrottleInfo:
        """Calculate throttling for a request."""
        pass
    
    @abstractmethod
    async def update_strategy(self, context_key: str, success: bool, response_time: float) -> None:
        """Update throttling strategy based on request outcome."""
        pass


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

class FixedWindowAlgorithm(RateLimitAlgorithm):
    """Fixed time window rate limiting algorithm."""
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.logger = get_logger(__name__)
    
    async def check_limit(self, context_key: str, limit: int, window_seconds: int) -> Tuple[bool, int, Optional[datetime]]:
        """Check rate limit using fixed window algorithm."""
        try:
            current_window = int(time.time() / window_seconds)
            key = f"rate_limit:fixed:{context_key}:{current_window}"
            
            # Get current count
            current_count = await self.storage.get(key) or 0
            
            # Check if limit exceeded
            if current_count >= limit:
                reset_time = datetime.fromtimestamp((current_window + 1) * window_seconds)
                return False, 0, reset_time
            
            # Calculate remaining requests
            remaining = max(0, limit - current_count)
            reset_time = datetime.fromtimestamp((current_window + 1) * window_seconds)
            
            return True, remaining, reset_time
            
        except Exception as e:
            self.logger.error(f"Error in fixed window rate limiting: {e}")
            # Fail open on error
            return True, limit, None
    
    async def record_request(self, context_key: str, timestamp: Optional[float] = None) -> None:
        """Record a request in the current window."""
        try:
            current_time = timestamp or time.time()
            window_seconds = 60  # Default 1-minute window
            current_window = int(current_time / window_seconds)
            key = f"rate_limit:fixed:{context_key}:{current_window}"
            
            # Increment counter with TTL
            await self.storage.increment(key, 1, ttl=window_seconds * 2)
            
        except Exception as e:
            self.logger.error(f"Error recording request: {e}")
    
    async def get_stats(self, context_key: str) -> Dict[str, Any]:
        """Get statistics for the context key."""
        try:
            current_window = int(time.time() / 60)
            key = f"rate_limit:fixed:{context_key}:{current_window}"
            
            current_count = await self.storage.get(key) or 0
            previous_key = f"rate_limit:fixed:{context_key}:{current_window - 1}"
            previous_count = await self.storage.get(previous_key) or 0
            
            return {
                "current_window": current_window,
                "current_count": current_count,
                "previous_count": previous_count,
                "algorithm": "fixed_window"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


class SlidingWindowAlgorithm(RateLimitAlgorithm):
    """Sliding time window rate limiting algorithm."""
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.logger = get_logger(__name__)
    
    async def check_limit(self, context_key: str, limit: int, window_seconds: int) -> Tuple[bool, int, Optional[datetime]]:
        """Check rate limit using sliding window algorithm."""
        try:
            current_time = time.time()
            key = f"rate_limit:sliding:{context_key}"
            
            # Get existing requests
            requests_data = await self.storage.get(key) or []
            
            # Remove expired requests
            cutoff_time = current_time - window_seconds
            requests_data = [ts for ts in requests_data if ts > cutoff_time]
            
            # Check if limit exceeded
            if len(requests_data) >= limit:
                # Calculate reset time (when oldest request expires)
                oldest_request = min(requests_data) if requests_data else current_time
                reset_time = datetime.fromtimestamp(oldest_request + window_seconds)
                return False, 0, reset_time
            
            # Calculate remaining requests
            remaining = max(0, limit - len(requests_data))
            
            # Calculate reset time
            if requests_data:
                oldest_request = min(requests_data)
                reset_time = datetime.fromtimestamp(oldest_request + window_seconds)
            else:
                reset_time = datetime.fromtimestamp(current_time + window_seconds)
            
            return True, remaining, reset_time
            
        except Exception as e:
            self.logger.error(f"Error in sliding window rate limiting: {e}")
            # Fail open on error
            return True, limit, None
    
    async def record_request(self, context_key: str, timestamp: Optional[float] = None) -> None:
        """Record a request in the sliding window."""
        try:
            current_time = timestamp or time.time()
            key = f"rate_limit:sliding:{context_key}"
            
            # Get existing requests
            requests_data = await self.storage.get(key) or []
            
            # Add current request
            requests_data.append(current_time)
            
            # Store with TTL
            await self.storage.set(key, requests_data, ttl=3600)  # 1 hour TTL
            
        except Exception as e:
            self.logger.error(f"Error recording request: {e}")
    
    async def get_stats(self, context_key: str) -> Dict[str, Any]:
        """Get statistics for the context key."""
        try:
            key = f"rate_limit:sliding:{context_key}"
            requests_data = await self.storage.get(key) or []
            
            current_time = time.time()
            window_seconds = 60  # Default 1-minute window
            cutoff_time = current_time - window_seconds
            
            # Count requests in current window
            current_count = len([ts for ts in requests_data if ts > cutoff_time])
            
            return {
                "current_count": current_count,
                "total_requests": len(requests_data),
                "window_seconds": window_seconds,
                "algorithm": "sliding_window"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


class TokenBucketAlgorithm(RateLimitAlgorithm):
    """Token bucket rate limiting algorithm."""
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.logger = get_logger(__name__)
    
    async def check_limit(self, context_key: str, limit: int, window_seconds: int) -> Tuple[bool, int, Optional[datetime]]:
        """Check rate limit using token bucket algorithm."""
        try:
            current_time = time.time()
            key = f"rate_limit:token_bucket:{context_key}"
            
            # Get bucket state
            bucket_data = await self.storage.get(key) or {
                "tokens": limit,
                "last_refill": current_time,
                "capacity": limit,
                "refill_rate": limit / window_seconds
            }
            
            # Calculate tokens to add since last refill
            time_passed = current_time - bucket_data["last_refill"]
            tokens_to_add = time_passed * bucket_data["refill_rate"]
            
            # Refill bucket (but don't exceed capacity)
            bucket_data["tokens"] = min(
                bucket_data["capacity"],
                bucket_data["tokens"] + tokens_to_add
            )
            bucket_data["last_refill"] = current_time
            
            # Check if tokens available
            if bucket_data["tokens"] < 1:
                # Calculate when next token will be available
                tokens_needed = 1 - bucket_data["tokens"]
                time_to_wait = tokens_needed / bucket_data["refill_rate"]
                reset_time = datetime.fromtimestamp(current_time + time_to_wait)
                return False, 0, reset_time
            
            # Calculate remaining tokens
            remaining = int(bucket_data["tokens"])
            
            # Calculate reset time (when bucket will be full)
            tokens_to_full = bucket_data["capacity"] - bucket_data["tokens"]
            time_to_full = tokens_to_full / bucket_data["refill_rate"]
            reset_time = datetime.fromtimestamp(current_time + time_to_full)
            
            return True, remaining, reset_time
            
        except Exception as e:
            self.logger.error(f"Error in token bucket rate limiting: {e}")
            # Fail open on error
            return True, limit, None
    
    async def record_request(self, context_key: str, timestamp: Optional[float] = None) -> None:
        """Consume a token from the bucket."""
        try:
            current_time = timestamp or time.time()
            key = f"rate_limit:token_bucket:{context_key}"
            
            # Get bucket state
            bucket_data = await self.storage.get(key) or {
                "tokens": 0,
                "last_refill": current_time,
                "capacity": 100,  # Default capacity
                "refill_rate": 100 / 60  # Default refill rate
            }
            
            # Consume one token
            bucket_data["tokens"] = max(0, bucket_data["tokens"] - 1)
            
            # Store updated state
            await self.storage.set(key, bucket_data, ttl=3600)
            
        except Exception as e:
            self.logger.error(f"Error recording request: {e}")
    
    async def get_stats(self, context_key: str) -> Dict[str, Any]:
        """Get statistics for the context key."""
        try:
            key = f"rate_limit:token_bucket:{context_key}"
            bucket_data = await self.storage.get(key) or {}
            
            return {
                "tokens": bucket_data.get("tokens", 0),
                "capacity": bucket_data.get("capacity", 0),
                "refill_rate": bucket_data.get("refill_rate", 0),
                "last_refill": bucket_data.get("last_refill", 0),
                "algorithm": "token_bucket"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


class AdaptiveThrottlingStrategy(ThrottlingStrategy):
    """Adaptive throttling strategy based on system load and response times."""
    
    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.logger = get_logger(__name__)
        
        # Throttling parameters
        self.base_delay = 0.1  # Base delay in seconds
        self.max_delay = 5.0   # Maximum delay in seconds
        self.load_threshold = 0.8  # Load threshold for throttling
        self.response_time_threshold = 1.0  # Response time threshold in seconds
    
    async def calculate_throttle(self, context: RateLimitContext, current_load: float) -> ThrottleInfo:
        """Calculate throttling based on context and system load."""
        try:
            # Get historical data for this context
            context_key = self._get_context_key(context)
            historical_data = await self._get_historical_data(context_key)
            
            # Calculate adaptive factor
            adaptive_factor = await self._calculate_adaptive_factor(historical_data, current_load)
            
            # Determine if throttling is needed
            should_throttle = (
                current_load > self.load_threshold or
                historical_data.get("avg_response_time", 0) > self.response_time_threshold or
                historical_data.get("error_rate", 0) > 0.1
            )
            
            # Calculate delay
            delay_seconds = 0.0
            reason = None
            
            if should_throttle:
                delay_seconds = min(
                    self.max_delay,
                    self.base_delay * adaptive_factor
                )
                
                if current_load > self.load_threshold:
                    reason = "High system load"
                elif historical_data.get("avg_response_time", 0) > self.response_time_threshold:
                    reason = "High response time"
                elif historical_data.get("error_rate", 0) > 0.1:
                    reason = "High error rate"
            
            return ThrottleInfo(
                should_throttle=should_throttle,
                delay_seconds=delay_seconds,
                reason=reason,
                adaptive_factor=adaptive_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating throttle: {e}")
            # Return no throttling on error
            return ThrottleInfo(should_throttle=False, delay_seconds=0.0)
    
    async def update_strategy(self, context_key: str, success: bool, response_time: float) -> None:
        """Update throttling strategy based on request outcome."""
        try:
            key = f"throttle_strategy:{context_key}"
            
            # Get existing data
            data = await self.storage.get(key) or {
                "request_count": 0,
                "success_count": 0,
                "total_response_time": 0.0,
                "response_times": [],
                "last_updated": time.time()
            }
            
            # Update statistics
            data["request_count"] += 1
            if success:
                data["success_count"] += 1
            
            data["total_response_time"] += response_time
            data["response_times"].append(response_time)
            
            # Keep only recent response times
            if len(data["response_times"]) > 100:
                data["response_times"] = data["response_times"][-100:]
            
            data["last_updated"] = time.time()
            
            # Store updated data
            await self.storage.set(key, data, ttl=3600)
            
        except Exception as e:
            self.logger.error(f"Error updating strategy: {e}")
    
    async def _calculate_adaptive_factor(self, historical_data: Dict[str, Any], current_load: float) -> float:
        """Calculate adaptive throttling factor."""
        try:
            base_factor = 1.0
            
            # Adjust based on error rate
            error_rate = historical_data.get("error_rate", 0)
            if error_rate > 0.1:
                base_factor *= (1 + error_rate * 2)
            
            # Adjust based on response time
            avg_response_time = historical_data.get("avg_response_time", 0)
            if avg_response_time > self.response_time_threshold:
                base_factor *= (1 + (avg_response_time - self.response_time_threshold))
            
            # Adjust based on current load
            if current_load > self.load_threshold:
                load_factor = (current_load - self.load_threshold) / (1 - self.load_threshold)
                base_factor *= (1 + load_factor)
            
            return max(1.0, min(base_factor, 10.0))  # Clamp between 1.0 and 10.0
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive factor: {e}")
            return 1.0
    
    async def _get_historical_data(self, context_key: str) -> Dict[str, Any]:
        """Get historical data for a context key."""
        try:
            key = f"throttle_strategy:{context_key}"
            data = await self.storage.get(key) or {}
            
            if not data:
                return {}
            
            # Calculate derived metrics
            request_count = data.get("request_count", 0)
            success_count = data.get("success_count", 0)
            total_response_time = data.get("total_response_time", 0.0)
            response_times = data.get("response_times", [])
            
            error_rate = 0.0
            if request_count > 0:
                error_rate = (request_count - success_count) / request_count
            
            avg_response_time = 0.0
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
            
            return {
                "request_count": request_count,
                "success_count": success_count,
                "error_rate": error_rate,
                "avg_response_time": avg_response_time,
                "last_updated": data.get("last_updated", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return {}
    
    def _get_context_key(self, context: RateLimitContext) -> str:
        """Generate a context key for storage."""
        key_parts = []
        
        if context.user_id:
            key_parts.append(f"user:{context.user_id}")
        if context.ip_address:
            key_parts.append(f"ip:{context.ip_address}")
        if context.endpoint:
            key_parts.append(f"endpoint:{context.endpoint}")
        
        return ":".join(key_parts) if key_parts else "default"


class InMemoryStorage(RateLimitStorage):
    """In-memory storage backend for rate limiting."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._storage: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from storage."""
        async with self._lock:
            # Check if key exists and is not expired
            if key in self._storage:
                if key in self._expiry and time.time() > self._expiry[key]:
                    # Key expired, remove it
                    del self._storage[key]
                    del self._expiry[key]
                    return None
                return self._storage[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in storage with optional TTL."""
        async with self._lock:
            self._storage[key] = value
            if ttl:
                self._expiry[key] = time.time() + ttl
            elif key in self._expiry:
                del self._expiry[key]
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a value in storage."""
        async with self._lock:
            current_value = self._storage.get(key, 0)
            new_value = current_value + amount
            self._storage[key] = new_value
            
            if ttl:
                self._expiry[key] = time.time() + ttl
            elif key in self._expiry:
                del self._expiry[key]
            
            return new_value
    
    async def delete(self, key: str) -> None:
        """Delete a key from storage."""
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
            if key in self._expiry:
                del self._expiry[key]
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        async with self._lock:
            if key in self._storage:
                if key in self._expiry and time.time() > self._expiry[key]:
                    # Key expired, remove it
                    del self._storage[key]
                    del self._expiry[key]
                    return False
                return True
            return False
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired keys periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                async with self._lock:
                    current_time = time.time()
                    expired_keys = [
                        key for key, expiry in self._expiry.items()
                        if current_time > expiry
                    ]
                    
                    for key in expired_keys:
                        if key in self._storage:
                            del self._storage[key]
                        if key in self._expiry:
                            del self._expiry[key]
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired keys")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")


class RedisStorage(RateLimitStorage):
    """Redis storage backend for distributed rate limiting."""
    
    def __init__(self, redis_url: str):
        self.logger = get_logger(__name__)
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        try:
            redis_client = await self._get_redis()
            value = await redis_client.get(key)
            
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Return as string if not JSON
                return value.decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Error getting from Redis: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in Redis with optional TTL."""
        try:
            redis_client = await self._get_redis()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            if ttl:
                await redis_client.setex(key, ttl, serialized_value)
            else:
                await redis_client.set(key, serialized_value)
                
        except Exception as e:
            self.logger.error(f"Error setting in Redis: {e}")
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment a value in Redis."""
        try:
            redis_client = await self._get_redis()
            
            # Use pipeline for atomic operation
            async with redis_client.pipeline() as pipe:
                await pipe.incrby(key, amount)
                if ttl:
                    await pipe.expire(key, ttl)
                results = await pipe.execute()
            
            return results[0]
            
        except Exception as e:
            self.logger.error(f"Error incrementing in Redis: {e}")
            return 0
    
    async def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        try:
            redis_client = await self._get_redis()
            await redis_client.delete(key)
        except Exception as e:
            self.logger.error(f"Error deleting from Redis: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        try:
            redis_client = await self._get_redis()
            return bool(await redis_client.exists(key))
        except Exception as e:
            self.logger.error(f"Error checking existence in Redis: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# ============================================================================
# MAIN RATE LIMITER
# ============================================================================

class RateLimiter:
    """Main rate limiter that orchestrates all rate limiting features."""
    
    def __init__(self, config: RateLimitingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize storage
        if config.enable_distributed and config.redis_host:
            redis_url = f"redis://{config.redis_host}:{config.redis_port}/{config.redis_db}"
            self.storage = RedisStorage(redis_url)
        else:
            self.storage = InMemoryStorage()
        
        # Initialize algorithms
        self.algorithms: Dict[str, RateLimitAlgorithm] = {
            "fixed_window": FixedWindowAlgorithm(self.storage),
            "sliding_window": SlidingWindowAlgorithm(self.storage),
            "token_bucket": TokenBucketAlgorithm(self.storage)
        }
        
        # Initialize throttling strategy
        self.throttling_strategy = AdaptiveThrottlingStrategy(self.storage)
        
        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Statistics
        self.stats = RateLimitStats()
        
        # Initialize default rules
        self._initialize_default_rules()
        
        self.logger.info("Rate limiter initialized")
    
    async def check_rate_limit(self, context: RateLimitContext) -> RateLimitResult:
        """Check if a request is within rate limits."""
        try:
            # Update statistics
            self.stats.total_requests += 1
            self.stats.last_updated = datetime.utcnow()
            
            # Find applicable rule
            rule = await self._find_applicable_rule(context)
            if not rule:
                # No rule found, allow request
                self.stats.allowed_requests += 1
                return RateLimitResult(
                    allowed=True,
                    remaining=999999,
                    limit=999999,
                    context_key="default",
                    rule_name="default"
                )
            
            # Generate context key
            context_key = await self._generate_context_key(context, rule)
            
            # Check rate limit
            algorithm = self.algorithms.get(rule.algorithm, self.algorithms["sliding_window"])
            
            # Get primary limit (e.g., per minute)
            primary_limit = rule.limits.get("per_minute", self.config.requests_per_minute)
            is_allowed, remaining, reset_time = await algorithm.check_limit(
                context_key, primary_limit, 60
            )
            
            # Check burst limit if applicable
            if is_allowed and rule.burst_limit > 0:
                burst_key = f"{context_key}:burst"
                burst_count = await self.storage.get(burst_key) or 0
                if burst_count >= rule.burst_limit:
                    is_allowed = False
                    remaining = 0
            
            # Update statistics
            if is_allowed:
                self.stats.allowed_requests += 1
                await self._record_request(context_key, rule, algorithm)
            else:
                self.stats.blocked_requests += 1
                self.stats.rate_limited_requests += 1
            
            # Update context key statistics
            self.stats.context_keys[context_key] = self.stats.context_keys.get(context_key, 0) + 1
            self.stats.rules_used[rule.name] = self.stats.rules_used.get(rule.name, 0) + 1
            
            # Calculate retry after
            retry_after = None
            if not is_allowed and reset_time:
                retry_after = int((reset_time - datetime.utcnow()).total_seconds())
                retry_after = max(0, retry_after)
            
            return RateLimitResult(
                allowed=is_allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                limit=primary_limit,
                context_key=context_key,
                rule_name=rule.name
            )
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit: {e}")
            # Fail open on error
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                limit=999999,
                context_key="error",
                rule_name="error"
            )
    
    async def should_throttle(self, context: RateLimitContext, current_load: float = 0.5) -> ThrottleInfo:
        """Check if a request should be throttled."""
        try:
            return await self.throttling_strategy.calculate_throttle(context, current_load)
        except Exception as e:
            self.logger.error(f"Error checking throttling: {e}")
            return ThrottleInfo(should_throttle=False, delay_seconds=0.0)
    
    async def add_rule(self, rule: RateLimitRule) -> None:
        """Add a new rate limiting rule."""
        self.rules[rule.name] = rule
        self.logger.info(f"Added rate limiting rule: {rule.name}")
    
    async def remove_rule(self, rule_name: str) -> bool:
        """Remove a rate limiting rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed rate limiting rule: {rule_name}")
            return True
        return False
    
    async def get_stats(self) -> RateLimitStats:
        """Get rate limiting statistics."""
        return self.stats
    
    async def get_algorithm_stats(self, context_key: str, algorithm_name: str = "sliding_window") -> Dict[str, Any]:
        """Get statistics for a specific algorithm and context key."""
        try:
            algorithm = self.algorithms.get(algorithm_name)
            if algorithm:
                return await algorithm.get_stats(context_key)
            return {"error": "Algorithm not found"}
        except Exception as e:
            self.logger.error(f"Error getting algorithm stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Cleanup rate limiter resources."""
        try:
            if isinstance(self.storage, RedisStorage):
                await self.storage.close()
            
            self.logger.info("Rate limiter cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def _find_applicable_rule(self, context: RateLimitContext) -> Optional[RateLimitRule]:
        """Find the most applicable rate limiting rule for the context."""
        try:
            applicable_rules = []
            
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule applies to this context
                if await self._rule_applies_to_context(rule, context):
                    applicable_rules.append(rule)
            
            if not applicable_rules:
                return None
            
            # Return the most specific rule (most context keys)
            return max(applicable_rules, key=lambda r: len(r.context_keys))
            
        except Exception as e:
            self.logger.error(f"Error finding applicable rule: {e}")
            return None
    
    async def _rule_applies_to_context(self, rule: RateLimitRule, context: RateLimitContext) -> bool:
        """Check if a rule applies to the given context."""
        try:
            for context_key in rule.context_keys:
                if context_key == "user_id" and not context.user_id:
                    return False
                elif context_key == "ip_address" and not context.ip_address:
                    return False
                elif context_key == "endpoint" and not context.endpoint:
                    return False
                elif context_key == "method" and not context.method:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rule applicability: {e}")
            return False
    
    async def _generate_context_key(self, context: RateLimitContext, rule: RateLimitRule) -> str:
        """Generate a unique context key for rate limiting."""
        try:
            key_parts = []
            
            for context_key in rule.context_keys:
                if context_key == "user_id" and context.user_id:
                    key_parts.append(f"user:{context.user_id}")
                elif context_key == "ip_address" and context.ip_address:
                    key_parts.append(f"ip:{context.ip_address}")
                elif context_key == "endpoint" and context.endpoint:
                    key_parts.append(f"endpoint:{context.endpoint}")
                elif context_key == "method" and context.method:
                    key_parts.append(f"method:{context.method}")
            
            # Add rule name for uniqueness
            key_parts.append(f"rule:{rule.name}")
            
            # Generate hash for consistent key length
            context_string = ":".join(key_parts)
            return hashlib.md5(context_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating context key: {e}")
            return "default"
    
    async def _record_request(self, context_key: str, rule: RateLimitRule, algorithm: RateLimitAlgorithm) -> None:
        """Record a request for rate limiting purposes."""
        try:
            # Record in algorithm
            await algorithm.record_request(context_key)
            
            # Record burst if applicable
            if rule.burst_limit > 0:
                burst_key = f"{context_key}:burst"
                current_burst = await self.storage.get(burst_key) or 0
                await self.storage.set(burst_key, current_burst + 1, ttl=60)
                
        except Exception as e:
            self.logger.error(f"Error recording request: {e}")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default rate limiting rules."""
        default_rules = [
            RateLimitRule(
                name="global_default",
                context_keys=["ip_address"],
                limits={"per_minute": self.config.requests_per_minute},
                algorithm="sliding_window",
                burst_limit=self.config.burst_limit,
                description="Default global rate limiting rule"
            ),
            RateLimitRule(
                name="user_specific",
                context_keys=["user_id", "ip_address"],
                limits={"per_minute": self.config.requests_per_minute * 2},
                algorithm="sliding_window",
                burst_limit=self.config.burst_limit * 2,
                description="User-specific rate limiting rule"
            ),
            RateLimitRule(
                name="api_endpoint",
                context_keys=["endpoint", "ip_address"],
                limits={"per_minute": self.config.requests_per_minute // 2},
                algorithm="fixed_window",
                burst_limit=self.config.burst_limit // 2,
                description="API endpoint rate limiting rule"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_rate_limiter(config: RateLimitingConfig) -> RateLimiter:
    """Create and configure rate limiter."""
    return RateLimiter(config)


def create_storage(config: RateLimitingConfig) -> RateLimitStorage:
    """Create and configure storage backend."""
    if config.enable_distributed and config.redis_host:
        redis_url = f"redis://{config.redis_host}:{config.redis_port}/{config.redis_db}"
        return RedisStorage(redis_url)
    else:
        return InMemoryStorage()


def create_algorithm(algorithm_name: str, storage: RateLimitStorage) -> RateLimitAlgorithm:
    """Create and configure rate limiting algorithm."""
    algorithms = {
        "fixed_window": FixedWindowAlgorithm,
        "sliding_window": SlidingWindowAlgorithm,
        "token_bucket": TokenBucketAlgorithm
    }
    
    algorithm_class = algorithms.get(algorithm_name, SlidingWindowAlgorithm)
    return algorithm_class(storage)


def create_throttling_strategy(storage: RateLimitStorage) -> ThrottlingStrategy:
    """Create and configure throttling strategy."""
    return AdaptiveThrottlingStrategy(storage)
