"""
Advanced Rate Limiting
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import logging
import threading

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float
    ):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens
        
        Args:
            tokens: Number of tokens to consume
        
        Returns:
            True if successful
        """
        with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_available_tokens(self) -> float:
        """Get available tokens"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            available = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            return available


class SlidingWindowLimiter:
    """Sliding window rate limiter"""
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: float
    ):
        """
        Initialize sliding window limiter
        
        Args:
            max_requests: Maximum requests
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, float]:
        """
        Check if request is allowed
        
        Args:
            identifier: Request identifier
        
        Returns:
            Tuple of (allowed, wait_time)
        """
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests
            requests = self.requests[identifier]
            while requests and requests[0] < window_start:
                requests.popleft()
            
            if len(requests) < self.max_requests:
                requests.append(now)
                return True, 0.0
            
            # Calculate wait time
            wait_time = requests[0] + self.window_seconds - now
            return False, max(0.0, wait_time)


class AdvancedRateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(
        self,
        strategy: str = "token_bucket",
        **kwargs
    ):
        """
        Initialize advanced rate limiter
        
        Args:
            strategy: Limiting strategy (token_bucket, sliding_window)
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        
        if strategy == "token_bucket":
            self.limiter = TokenBucket(
                capacity=kwargs.get("capacity", 100),
                refill_rate=kwargs.get("refill_rate", 10.0)
            )
        elif strategy == "sliding_window":
            self.limiter = SlidingWindowLimiter(
                max_requests=kwargs.get("max_requests", 100),
                window_seconds=kwargs.get("window_seconds", 60.0)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"AdvancedRateLimiter initialized: {strategy}")
    
    def is_allowed(self, identifier: str = "default") -> Tuple[bool, float]:
        """
        Check if request is allowed
        
        Args:
            identifier: Request identifier
        
        Returns:
            Tuple of (allowed, wait_time)
        """
        if self.strategy == "token_bucket":
            allowed = self.limiter.consume()
            wait_time = 0.0 if allowed else 1.0 / self.limiter.refill_rate
            return allowed, wait_time
        else:
            return self.limiter.is_allowed(identifier)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limiter statistics"""
        if self.strategy == "token_bucket":
            return {
                "strategy": self.strategy,
                "available_tokens": self.limiter.get_available_tokens(),
                "capacity": self.limiter.capacity
            }
        else:
            return {
                "strategy": self.strategy,
                "max_requests": self.limiter.max_requests,
                "window_seconds": self.limiter.window_seconds
            }

