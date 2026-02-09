"""
Advanced Throttler for Piel Mejorador AI SAM3
=============================================

Advanced throttling with multiple strategies.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ThrottleStrategy(Enum):
    """Throttling strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class ThrottleConfig:
    """Throttle configuration."""
    strategy: ThrottleStrategy = ThrottleStrategy.SLIDING_WINDOW
    max_requests: int = 100
    window_seconds: float = 60.0
    tokens_per_second: float = 10.0
    burst_size: int = 20


class AdvancedThrottler:
    """
    Advanced throttler with multiple strategies.
    
    Features:
    - Multiple throttling strategies
    - Per-client throttling
    - Dynamic rate adjustment
    - Burst handling
    """
    
    def __init__(self, config: Optional[ThrottleConfig] = None):
        """
        Initialize throttler.
        
        Args:
            config: Throttle configuration
        """
        self.config = config or ThrottleConfig()
        self._clients: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        
        self._stats = {
            "total_requests": 0,
            "throttled_requests": 0,
            "allowed_requests": 0,
        }
    
    async def is_allowed(
        self,
        client_id: str,
        tokens: float = 1.0
    ) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier
            tokens: Number of tokens to consume
            
        Returns:
            True if allowed
        """
        async with self._lock:
            self._stats["total_requests"] += 1
            
            if client_id not in self._clients:
                self._clients[client_id] = self._create_client_state()
            
            client_state = self._clients[client_id]
            allowed = self._check_throttle(client_state, tokens)
            
            if allowed:
                self._stats["allowed_requests"] += 1
            else:
                self._stats["throttled_requests"] += 1
            
            return allowed
    
    def _create_client_state(self) -> Dict[str, Any]:
        """Create client state based on strategy."""
        if self.config.strategy == ThrottleStrategy.FIXED_WINDOW:
            return {
                "window_start": time.time(),
                "count": 0,
            }
        elif self.config.strategy == ThrottleStrategy.SLIDING_WINDOW:
            return {
                "requests": deque(),
            }
        elif self.config.strategy == ThrottleStrategy.TOKEN_BUCKET:
            return {
                "tokens": float(self.config.burst_size),
                "last_refill": time.time(),
            }
        elif self.config.strategy == ThrottleStrategy.LEAKY_BUCKET:
            return {
                "queue_size": 0,
                "last_leak": time.time(),
            }
        return {}
    
    def _check_throttle(self, state: Dict[str, Any], tokens: float) -> bool:
        """Check throttle based on strategy."""
        if self.config.strategy == ThrottleStrategy.FIXED_WINDOW:
            return self._check_fixed_window(state, tokens)
        elif self.config.strategy == ThrottleStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(state, tokens)
        elif self.config.strategy == ThrottleStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(state, tokens)
        elif self.config.strategy == ThrottleStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(state, tokens)
        return True
    
    def _check_fixed_window(self, state: Dict[str, Any], tokens: float) -> bool:
        """Fixed window throttling."""
        now = time.time()
        window_start = state["window_start"]
        
        if now - window_start >= self.config.window_seconds:
            # New window
            state["window_start"] = now
            state["count"] = 0
        
        if state["count"] + tokens <= self.config.max_requests:
            state["count"] += tokens
            return True
        return False
    
    def _check_sliding_window(self, state: Dict[str, Any], tokens: float) -> bool:
        """Sliding window throttling."""
        now = time.time()
        requests = state["requests"]
        
        # Remove old requests
        cutoff = now - self.config.window_seconds
        while requests and requests[0] < cutoff:
            requests.popleft()
        
        if len(requests) + tokens <= self.config.max_requests:
            for _ in range(int(tokens)):
                requests.append(now)
            return True
        return False
    
    def _check_token_bucket(self, state: Dict[str, Any], tokens: float) -> bool:
        """Token bucket throttling."""
        now = time.time()
        elapsed = now - state["last_refill"]
        
        # Refill tokens
        state["tokens"] = min(
            self.config.burst_size,
            state["tokens"] + elapsed * self.config.tokens_per_second
        )
        state["last_refill"] = now
        
        if state["tokens"] >= tokens:
            state["tokens"] -= tokens
            return True
        return False
    
    def _check_leaky_bucket(self, state: Dict[str, Any], tokens: float) -> bool:
        """Leaky bucket throttling."""
        now = time.time()
        elapsed = now - state["last_leak"]
        
        # Leak tokens
        leak_rate = self.config.tokens_per_second
        leaked = elapsed * leak_rate
        state["queue_size"] = max(0, state["queue_size"] - leaked)
        state["last_leak"] = now
        
        max_queue = self.config.burst_size
        if state["queue_size"] + tokens <= max_queue:
            state["queue_size"] += tokens
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get throttler statistics."""
        throttle_rate = (
            self._stats["throttled_requests"] / self._stats["total_requests"]
            if self._stats["total_requests"] > 0 else 0
        )
        
        return {
            **self._stats,
            "throttle_rate": throttle_rate,
            "active_clients": len(self._clients),
            "strategy": self.config.strategy.value,
        }




