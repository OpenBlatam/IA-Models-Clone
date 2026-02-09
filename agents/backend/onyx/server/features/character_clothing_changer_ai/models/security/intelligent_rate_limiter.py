"""
Intelligent Rate Limiter
========================

Advanced rate limiting with adaptive algorithms.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limit strategy."""
    FIXED = "fixed"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window_seconds: float
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


@dataclass
class RateLimitResult:
    """Rate limit check result."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None


class IntelligentRateLimiter:
    """Intelligent rate limiter with adaptive algorithms."""
    
    def __init__(
        self,
        default_limit: RateLimit,
        strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE,
    ):
        """
        Initialize intelligent rate limiter.
        
        Args:
            default_limit: Default rate limit
            strategy: Rate limit strategy
        """
        self.default_limit = default_limit
        self.strategy = strategy
        
        # Per-identifier tracking
        self.requests: Dict[str, deque] = {}
        self.tokens: Dict[str, float] = {}  # For token bucket
        self.last_refill: Dict[str, float] = {}
        
        # Adaptive tracking
        self.request_history: Dict[str, List[float]] = {}
        self.adaptive_limits: Dict[str, RateLimit] = {}
    
    def check_rate_limit(
        self,
        identifier: str,
        limit: Optional[RateLimit] = None,
    ) -> RateLimitResult:
        """
        Check rate limit for identifier.
        
        Args:
            identifier: Client identifier (IP, user ID, etc.)
            limit: Optional custom limit
            
        Returns:
            Rate limit result
        """
        limit = limit or self.default_limit
        current_time = time.time()
        
        if self.strategy == RateLimitStrategy.ADAPTIVE:
            return self._check_adaptive(identifier, limit, current_time)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(identifier, limit, current_time)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(identifier, limit, current_time)
        else:  # FIXED
            return self._check_fixed(identifier, limit, current_time)
    
    def _check_sliding_window(
        self,
        identifier: str,
        limit: RateLimit,
        current_time: float,
    ) -> RateLimitResult:
        """Check using sliding window algorithm."""
        if identifier not in self.requests:
            self.requests[identifier] = deque()
        
        requests = self.requests[identifier]
        window_start = current_time - limit.window_seconds
        
        # Remove old requests
        while requests and requests[0] < window_start:
            requests.popleft()
        
        allowed = len(requests) < limit.requests
        
        if allowed:
            requests.append(current_time)
        
        remaining = max(0, limit.requests - len(requests))
        reset_time = current_time + limit.window_seconds
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=limit.window_seconds if not allowed else None,
        )
    
    def _check_token_bucket(
        self,
        identifier: str,
        limit: RateLimit,
        current_time: float,
    ) -> RateLimitResult:
        """Check using token bucket algorithm."""
        if identifier not in self.tokens:
            self.tokens[identifier] = float(limit.requests)
            self.last_refill[identifier] = current_time
        
        # Refill tokens
        time_passed = current_time - self.last_refill[identifier]
        tokens_to_add = (time_passed / limit.window_seconds) * limit.requests
        self.tokens[identifier] = min(
            limit.requests,
            self.tokens[identifier] + tokens_to_add
        )
        self.last_refill[identifier] = current_time
        
        allowed = self.tokens[identifier] >= 1.0
        
        if allowed:
            self.tokens[identifier] -= 1.0
        
        remaining = int(self.tokens[identifier])
        reset_time = current_time + limit.window_seconds
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=limit.window_seconds if not allowed else None,
        )
    
    def _check_fixed(
        self,
        identifier: str,
        limit: RateLimit,
        current_time: float,
    ) -> RateLimitResult:
        """Check using fixed window algorithm."""
        if identifier not in self.requests:
            self.requests[identifier] = deque()
        
        requests = self.requests[identifier]
        
        # Remove requests outside window
        window_start = current_time - limit.window_seconds
        while requests and requests[0] < window_start:
            requests.popleft()
        
        allowed = len(requests) < limit.requests
        
        if allowed:
            requests.append(current_time)
        
        remaining = max(0, limit.requests - len(requests))
        reset_time = current_time + limit.window_seconds
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=limit.window_seconds if not allowed else None,
        )
    
    def _check_adaptive(
        self,
        identifier: str,
        limit: RateLimit,
        current_time: float,
    ) -> RateLimitResult:
        """Check using adaptive algorithm."""
        # Track request history
        if identifier not in self.request_history:
            self.request_history[identifier] = []
        
        history = self.request_history[identifier]
        history.append(current_time)
        
        # Keep only last hour
        hour_ago = current_time - 3600
        history[:] = [t for t in history if t > hour_ago]
        
        # Calculate adaptive limit based on recent behavior
        if len(history) > 10:
            recent_requests = [t for t in history if t > current_time - limit.window_seconds]
            avg_rate = len(recent_requests) / limit.window_seconds
            
            # Adjust limit based on average rate
            if avg_rate > limit.requests / limit.window_seconds * 0.8:
                # High usage, be more restrictive
                adaptive_limit = RateLimit(
                    requests=int(limit.requests * 0.9),
                    window_seconds=limit.window_seconds,
                )
            else:
                # Low usage, allow more
                adaptive_limit = limit
        else:
            adaptive_limit = limit
        
        # Use sliding window with adaptive limit
        return self._check_sliding_window(identifier, adaptive_limit, current_time)
    
    def reset(self, identifier: str) -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Client identifier
        """
        self.requests.pop(identifier, None)
        self.tokens.pop(identifier, None)
        self.last_refill.pop(identifier, None)
        self.request_history.pop(identifier, None)
        self.adaptive_limits.pop(identifier, None)
    
    def get_statistics(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Args:
            identifier: Optional identifier filter
            
        Returns:
            Statistics dictionary
        """
        if identifier:
            requests_count = len(self.requests.get(identifier, deque()))
            tokens = self.tokens.get(identifier, 0)
            history_count = len(self.request_history.get(identifier, []))
            
            return {
                "identifier": identifier,
                "requests_in_window": requests_count,
                "tokens": tokens,
                "history_count": history_count,
            }
        
        return {
            "total_identifiers": len(self.requests),
            "strategy": self.strategy.value,
            "default_limit": {
                "requests": self.default_limit.requests,
                "window_seconds": self.default_limit.window_seconds,
            },
        }

