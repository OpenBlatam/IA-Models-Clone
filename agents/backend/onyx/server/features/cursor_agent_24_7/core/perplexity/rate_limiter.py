"""
Rate Limiter - Advanced rate limiting for Perplexity
====================================================

Advanced rate limiting with multiple strategies and Redis support.
"""

import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 10.0,  # tokens per second
        redis_client: Optional[Any] = None
    ):
        """
        Initialize token bucket rate limiter.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            redis_client: Optional Redis client for distributed limiting
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.redis_client = redis_client
        
        # In-memory storage (fallback)
        self.buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {'tokens': capacity, 'last_refill': time.time()}
        )
    
    def is_allowed(
        self,
        key: str = "default",
        tokens: int = 1
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed and consume tokens.
        
        Args:
            key: Rate limit key
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        if self.redis_client:
            return self._check_redis(key, tokens)
        else:
            return self._check_memory(key, tokens)
    
    def _check_memory(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Check using in-memory storage."""
        now = time.time()
        bucket = self.buckets[key]
        
        # Refill tokens
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * self.refill_rate
        bucket['tokens'] = min(
            self.capacity,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_refill'] = now
        
        # Check if enough tokens
        if bucket['tokens'] >= tokens:
            bucket['tokens'] -= tokens
            return True, {
                'allowed': True,
                'remaining': int(bucket['tokens']),
                'reset_after': int((self.capacity - bucket['tokens']) / self.refill_rate)
            }
        else:
            return False, {
                'allowed': False,
                'remaining': int(bucket['tokens']),
                'reset_after': int((tokens - bucket['tokens']) / self.refill_rate)
            }
    
    def _check_redis(
        self,
        key: str,
        tokens: int
    ) -> tuple[bool, Dict[str, Any]]:
        """Check using Redis (if available)."""
        try:
            # Simplified Redis implementation
            # In production, use proper Lua script for atomicity
            redis_key = f"rate_limit:{key}"
            bucket_data = self.redis_client.get(redis_key)
            
            if bucket_data:
                data = eval(bucket_data)  # In production, use JSON
                now = time.time()
                time_passed = now - data['last_refill']
                tokens_to_add = time_passed * self.refill_rate
                current_tokens = min(
                    self.capacity,
                    data['tokens'] + tokens_to_add
                )
            else:
                current_tokens = self.capacity
                now = time.time()
            
            if current_tokens >= tokens:
                current_tokens -= tokens
                self.redis_client.setex(
                    redis_key,
                    int(self.capacity / self.refill_rate) + 10,
                    str({
                        'tokens': current_tokens,
                        'last_refill': now
                    })
                )
                return True, {
                    'allowed': True,
                    'remaining': int(current_tokens),
                    'reset_after': int((self.capacity - current_tokens) / self.refill_rate)
                }
            else:
                return False, {
                    'allowed': False,
                    'remaining': int(current_tokens),
                    'reset_after': int((tokens - current_tokens) / self.refill_rate)
                }
        except Exception as e:
            logger.warning(f"Redis rate limit check failed, falling back to memory: {e}")
            return self._check_memory(key, tokens)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_client: Optional[Any] = None
    ):
        """
        Initialize sliding window rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            redis_client: Optional Redis client
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_client = redis_client
        self.windows: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, key: str = "default") -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        now = time.time()
        window = self.windows[key]
        
        # Remove old requests outside window
        while window and window[0] < now - self.window_seconds:
            window.popleft()
        
        # Check limit
        if len(window) >= self.max_requests:
            oldest_request = window[0] if window else now
            reset_after = int(self.window_seconds - (now - oldest_request))
            return False, {
                'allowed': False,
                'remaining': 0,
                'reset_after': reset_after
            }
        
        # Record request
        window.append(now)
        return True, {
            'allowed': True,
            'remaining': self.max_requests - len(window),
            'reset_after': self.window_seconds
        }




