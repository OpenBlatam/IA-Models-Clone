"""
Rate Limiter for Color Grading AI
==================================

Advanced rate limiting with multiple algorithms.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests: int = 100
    window_seconds: float = 60.0
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    refill_rate: float = 1.0  # Tokens per second
    burst_size: Optional[int] = None


class RateLimiter:
    """
    Advanced rate limiter.
    
    Features:
    - Multiple algorithms
    - Per-key limiting
    - Burst support
    - Statistics
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Optional rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._windows: Dict[str, deque] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
    
    def is_allowed(
        self,
        key: str = "default",
        cost: int = 1
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key
            cost: Request cost
            
        Returns:
            True if allowed
        """
        if self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._token_bucket_check(key, cost)
        elif self.config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._sliding_window_check(key, cost)
        elif self.config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._fixed_window_check(key, cost)
        elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return self._leaky_bucket_check(key, cost)
        
        return True
    
    def _token_bucket_check(self, key: str, cost: int) -> bool:
        """Token bucket algorithm."""
        now = time.time()
        
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": self.config.burst_size or self.config.max_requests,
                "last_refill": now,
            }
        
        bucket = self._buckets[key]
        
        # Refill tokens
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * self.config.refill_rate
        bucket["tokens"] = min(
            self.config.burst_size or self.config.max_requests,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now
        
        # Check if enough tokens
        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            self._record_allowed(key)
            return True
        
        self._record_denied(key)
        return False
    
    def _sliding_window_check(self, key: str, cost: int) -> bool:
        """Sliding window algorithm."""
        now = time.time()
        window_start = now - self.config.window_seconds
        
        if key not in self._windows:
            self._windows[key] = deque()
        
        window = self._windows[key]
        
        # Remove old entries
        while window and window[0] < window_start:
            window.popleft()
        
        # Check capacity
        if len(window) + cost <= self.config.max_requests:
            # Add current request
            for _ in range(cost):
                window.append(now)
            self._record_allowed(key)
            return True
        
        self._record_denied(key)
        return False
    
    def _fixed_window_check(self, key: str, cost: int) -> bool:
        """Fixed window algorithm."""
        now = time.time()
        window_start = int(now / self.config.window_seconds) * self.config.window_seconds
        
        if key not in self._buckets:
            self._buckets[key] = {
                "count": 0,
                "window_start": window_start,
            }
        
        bucket = self._buckets[key]
        
        # Reset if new window
        if bucket["window_start"] < window_start:
            bucket["count"] = 0
            bucket["window_start"] = window_start
        
        # Check capacity
        if bucket["count"] + cost <= self.config.max_requests:
            bucket["count"] += cost
            self._record_allowed(key)
            return True
        
        self._record_denied(key)
        return False
    
    def _leaky_bucket_check(self, key: str, cost: int) -> bool:
        """Leaky bucket algorithm."""
        now = time.time()
        
        if key not in self._buckets:
            self._buckets[key] = {
                "level": 0,
                "last_leak": now,
            }
        
        bucket = self._buckets[key]
        
        # Leak tokens
        elapsed = now - bucket["last_leak"]
        leaked = elapsed * self.config.refill_rate
        bucket["level"] = max(0, bucket["level"] - leaked)
        bucket["last_leak"] = now
        
        # Check capacity
        max_level = self.config.burst_size or self.config.max_requests
        if bucket["level"] + cost <= max_level:
            bucket["level"] += cost
            self._record_allowed(key)
            return True
        
        self._record_denied(key)
        return False
    
    def _record_allowed(self, key: str):
        """Record allowed request."""
        if key not in self._stats:
            self._stats[key] = {
                "allowed": 0,
                "denied": 0,
                "total": 0,
            }
        
        self._stats[key]["allowed"] += 1
        self._stats[key]["total"] += 1
    
    def _record_denied(self, key: str):
        """Record denied request."""
        if key not in self._stats:
            self._stats[key] = {
                "allowed": 0,
                "denied": 0,
                "total": 0,
            }
        
        self._stats[key]["denied"] += 1
        self._stats[key]["total"] += 1
    
    def get_stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limit statistics."""
        if key:
            return self._stats.get(key, {})
        return self._stats.copy()
    
    def reset(self, key: Optional[str] = None):
        """Reset rate limiter for key or all."""
        if key:
            if key in self._buckets:
                del self._buckets[key]
            if key in self._windows:
                del self._windows[key]
            if key in self._stats:
                del self._stats[key]
        else:
            self._buckets.clear()
            self._windows.clear()
            self._stats.clear()




