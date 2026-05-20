"""
Advanced Throttling System
==========================

Advanced throttling system with multiple strategies and per-user limits.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ThrottleStrategy(Enum):
    """Throttling strategy."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class ThrottleConfig:
    """Throttle configuration."""
    strategy: ThrottleStrategy = ThrottleStrategy.SLIDING_WINDOW
    rate: float = 10.0  # Requests per time window
    window_seconds: float = 60.0  # Time window in seconds
    burst: int = 20  # Burst capacity (for token bucket)
    per_user: bool = False  # Apply per user or globally


class Throttler:
    """Advanced throttler with multiple strategies."""
    
    def __init__(self, config: ThrottleConfig):
        """
        Initialize throttler.
        
        Args:
            config: Throttle configuration
        """
        self.config = config
        self.requests: Dict[str, deque] = {}  # Per-user or global
        self.tokens: Dict[str, float] = {}  # For token bucket
        self.last_update: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _get_key(self, user_id: Optional[str] = None) -> str:
        """Get throttle key for user."""
        if self.config.per_user and user_id:
            return f"user:{user_id}"
        return "global"
    
    async def check(
        self,
        user_id: Optional[str] = None
    ) -> tuple[bool, Optional[float]]:
        """
        Check if request should be throttled.
        
        Args:
            user_id: Optional user ID
            
        Returns:
            Tuple of (is_allowed, wait_time_seconds)
        """
        async with self._lock:
            key = self._get_key(user_id)
            
            if self.config.strategy == ThrottleStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key)
            elif self.config.strategy == ThrottleStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key)
            elif self.config.strategy == ThrottleStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key)
            else:
                return True, None
    
    async def _check_sliding_window(self, key: str) -> tuple[bool, Optional[float]]:
        """Check sliding window throttle."""
        now = time.time()
        
        if key not in self.requests:
            self.requests[key] = deque()
        
        requests = self.requests[key]
        
        # Remove old requests outside window
        while requests and requests[0] < now - self.config.window_seconds:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < self.config.rate:
            requests.append(now)
            return True, None
        
        # Calculate wait time
        oldest_request = requests[0]
        wait_time = (oldest_request + self.config.window_seconds) - now
        return False, max(0.0, wait_time)
    
    async def _check_fixed_window(self, key: str) -> tuple[bool, Optional[float]]:
        """Check fixed window throttle."""
        now = time.time()
        window_start = int(now / self.config.window_seconds) * self.config.window_seconds
        
        if key not in self.requests:
            self.requests[key] = deque()
        
        requests = self.requests[key]
        
        # Remove requests from previous windows
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check if under limit
        if len(requests) < self.config.rate:
            requests.append(now)
            return True, None
        
        # Calculate wait time until next window
        wait_time = (window_start + self.config.window_seconds) - now
        return False, max(0.0, wait_time)
    
    async def _check_token_bucket(self, key: str) -> tuple[bool, Optional[float]]:
        """Check token bucket throttle."""
        now = time.time()
        
        if key not in self.tokens:
            self.tokens[key] = float(self.config.burst)
            self.last_update[key] = now
        
        tokens = self.tokens[key]
        last_update = self.last_update[key]
        
        # Add tokens based on elapsed time
        elapsed = now - last_update
        tokens = min(
            self.config.burst,
            tokens + elapsed * (self.config.rate / self.config.window_seconds)
        )
        
        # Check if we have tokens
        if tokens >= 1.0:
            tokens -= 1.0
            self.tokens[key] = tokens
            self.last_update[key] = now
            return True, None
        
        # Calculate wait time
        needed = 1.0 - tokens
        wait_time = needed / (self.config.rate / self.config.window_seconds)
        return False, wait_time
    
    async def wait(
        self,
        user_id: Optional[str] = None
    ):
        """
        Wait until request is allowed.
        
        Args:
            user_id: Optional user ID
        """
        is_allowed, wait_time = await self.check(user_id)
        
        if not is_allowed and wait_time:
            await asyncio.sleep(wait_time)
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get throttler statistics.
        
        Args:
            user_id: Optional user ID
            
        Returns:
            Statistics dictionary
        """
        key = self._get_key(user_id)
        
        stats = {
            "key": key,
            "strategy": self.config.strategy.value,
            "rate": self.config.rate,
            "window_seconds": self.config.window_seconds
        }
        
        if key in self.requests:
            stats["current_requests"] = len(self.requests[key])
        
        if key in self.tokens:
            stats["current_tokens"] = self.tokens[key]
        
        return stats

