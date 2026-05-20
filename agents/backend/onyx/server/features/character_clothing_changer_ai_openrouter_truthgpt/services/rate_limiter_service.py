"""
Intelligent Rate Limiter Service
=================================
Service for rate limiting with multiple strategies
"""

import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: float
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # For token bucket


class RateLimiterService:
    """
    Intelligent rate limiter with multiple strategies
    """
    
    def __init__(self):
        self._limits: Dict[str, RateLimitConfig] = {}
        self._counters: Dict[str, Any] = {}
        self._stats = defaultdict(lambda: {
            'allowed': 0,
            'denied': 0,
            'total_requests': 0
        })
    
    def register_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: float,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
        burst_size: Optional[int] = None
    ):
        """Register rate limit for identifier"""
        self._limits[identifier] = RateLimitConfig(
            max_requests=max_requests,
            window_seconds=window_seconds,
            strategy=strategy,
            burst_size=burst_size or max_requests
        )
        
        # Initialize counter based on strategy
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._counters[identifier] = {
                'tokens': burst_size or max_requests,
                'last_refill': time.time(),
                'refill_rate': max_requests / window_seconds
            }
        elif strategy == RateLimitStrategy.LEAKY_BUCKET:
            self._counters[identifier] = {
                'queue': deque(),
                'last_leak': time.time(),
                'leak_rate': max_requests / window_seconds
            }
        else:
            self._counters[identifier] = {
                'requests': deque(),
                'window_start': time.time()
            }
    
    def check(
        self,
        identifier: str,
        count: int = 1
    ) -> RateLimitResult:
        """Check if request is allowed"""
        if identifier not in self._limits:
            # No limit configured, allow
            return RateLimitResult(
                allowed=True,
                remaining=float('inf'),
                reset_at=time.time() + 3600
            )
        
        config = self._limits[identifier]
        self._stats[identifier]['total_requests'] += count
        
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            result = self._check_fixed_window(identifier, config, count)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            result = self._check_sliding_window(identifier, config, count)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            result = self._check_token_bucket(identifier, config, count)
        elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            result = self._check_leaky_bucket(identifier, config, count)
        else:
            result = self._check_sliding_window(identifier, config, count)
        
        if result.allowed:
            self._stats[identifier]['allowed'] += count
        else:
            self._stats[identifier]['denied'] += count
        
        return result
    
    def _check_fixed_window(
        self,
        identifier: str,
        config: RateLimitConfig,
        count: int
    ) -> RateLimitResult:
        """Fixed window rate limiting"""
        counter = self._counters[identifier]
        now = time.time()
        
        # Reset window if expired
        if now - counter['window_start'] >= config.window_seconds:
            counter['window_start'] = now
            counter['requests'] = deque()
        
        # Check limit
        current_count = len(counter['requests'])
        if current_count + count > config.max_requests:
            reset_at = counter['window_start'] + config.window_seconds
            retry_after = max(0, reset_at - now)
            return RateLimitResult(
                allowed=False,
                remaining=config.max_requests - current_count,
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Add requests
        for _ in range(count):
            counter['requests'].append(now)
        
        reset_at = counter['window_start'] + config.window_seconds
        return RateLimitResult(
            allowed=True,
            remaining=config.max_requests - (current_count + count),
            reset_at=reset_at
        )
    
    def _check_sliding_window(
        self,
        identifier: str,
        config: RateLimitConfig,
        count: int
    ) -> RateLimitResult:
        """Sliding window rate limiting"""
        counter = self._counters[identifier]
        now = time.time()
        window_start = now - config.window_seconds
        
        # Remove old requests
        while counter['requests'] and counter['requests'][0] < window_start:
            counter['requests'].popleft()
        
        # Check limit
        current_count = len(counter['requests'])
        if current_count + count > config.max_requests:
            # Calculate retry after
            oldest_request = counter['requests'][0] if counter['requests'] else now
            reset_at = oldest_request + config.window_seconds
            retry_after = max(0, reset_at - now)
            return RateLimitResult(
                allowed=False,
                remaining=config.max_requests - current_count,
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Add requests
        for _ in range(count):
            counter['requests'].append(now)
        
        # Calculate reset time (oldest request + window)
        oldest_request = counter['requests'][0] if counter['requests'] else now
        reset_at = oldest_request + config.window_seconds
        
        return RateLimitResult(
            allowed=True,
            remaining=config.max_requests - (current_count + count),
            reset_at=reset_at
        )
    
    def _check_token_bucket(
        self,
        identifier: str,
        config: RateLimitConfig,
        count: int
    ) -> RateLimitResult:
        """Token bucket rate limiting"""
        counter = self._counters[identifier]
        now = time.time()
        
        # Refill tokens
        time_passed = now - counter['last_refill']
        tokens_to_add = time_passed * counter['refill_rate']
        counter['tokens'] = min(
            config.burst_size or config.max_requests,
            counter['tokens'] + tokens_to_add
        )
        counter['last_refill'] = now
        
        # Check if enough tokens
        if counter['tokens'] < count:
            # Calculate when enough tokens will be available
            tokens_needed = count - counter['tokens']
            retry_after = tokens_needed / counter['refill_rate']
            reset_at = now + retry_after
            return RateLimitResult(
                allowed=False,
                remaining=int(counter['tokens']),
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Consume tokens
        counter['tokens'] -= count
        
        # Calculate reset time (when bucket will be full)
        tokens_to_fill = (config.burst_size or config.max_requests) - counter['tokens']
        reset_at = now + (tokens_to_fill / counter['refill_rate'])
        
        return RateLimitResult(
            allowed=True,
            remaining=int(counter['tokens']),
            reset_at=reset_at
        )
    
    def _check_leaky_bucket(
        self,
        identifier: str,
        config: RateLimitConfig,
        count: int
    ) -> RateLimitResult:
        """Leaky bucket rate limiting"""
        counter = self._counters[identifier]
        now = time.time()
        
        # Leak requests
        time_passed = now - counter['last_leak']
        requests_to_leak = int(time_passed * counter['leak_rate'])
        for _ in range(min(requests_to_leak, len(counter['queue']))):
            counter['queue'].popleft()
        counter['last_leak'] = now
        
        # Check if bucket is full
        if len(counter['queue']) + count > config.max_requests:
            # Calculate when space will be available
            space_needed = (len(counter['queue']) + count) - config.max_requests
            retry_after = space_needed / counter['leak_rate']
            reset_at = now + retry_after
            return RateLimitResult(
                allowed=False,
                remaining=config.max_requests - len(counter['queue']),
                reset_at=reset_at,
                retry_after=retry_after
            )
        
        # Add to queue
        for _ in range(count):
            counter['queue'].append(now)
        
        # Calculate reset time (when queue will be empty)
        queue_size = len(counter['queue'])
        reset_at = now + (queue_size / counter['leak_rate'])
        
        return RateLimitResult(
            allowed=True,
            remaining=config.max_requests - queue_size,
            reset_at=reset_at
        )
    
    def get_statistics(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if identifier:
            stats = self._stats.get(identifier, {})
            total = stats['total_requests']
            return {
                'identifier': identifier,
                'allowed': stats['allowed'],
                'denied': stats['denied'],
                'total_requests': total,
                'denial_rate': stats['denied'] / total if total > 0 else 0
            }
        
        return {
            'limits': {
                id: {
                    'max_requests': limit.max_requests,
                    'window_seconds': limit.window_seconds,
                    'strategy': limit.strategy.value
                }
                for id, limit in self._limits.items()
            },
            'statistics': dict(self._stats)
        }


# Global rate limiter instance
rate_limiter_service = RateLimiterService()

