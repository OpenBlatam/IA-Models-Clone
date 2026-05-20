"""
Adaptive Rate Limiter
Intelligent rate limiting based on system load and performance metrics
"""

import logging
import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SystemLoad(Enum):
    """System load levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: float
    burst_size: int
    window_seconds: int = 60


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter
    
    Features:
    - Load-based rate limiting
    - Automatic rate adjustment
    - Burst handling
    - Priority-based limiting
    - Performance-aware throttling
    """
    
    def __init__(
        self,
        base_rate: float = 100.0,  # requests per second
        min_rate: float = 10.0,
        max_rate: float = 1000.0,
        window_size: int = 60
    ):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.window_size = window_size
        
        self._current_rate = base_rate
        self._request_times: deque = deque(maxlen=10000)
        self._response_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._success_count = 0
        
        self._load_history: deque = deque(maxlen=100)
        self._adjustment_lock = asyncio.Lock()
        
        logger.info(f"✅ Adaptive rate limiter initialized (base_rate: {base_rate}/s)")
    
    async def acquire(
        self,
        priority: int = 5,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire rate limit token
        
        Args:
            priority: Request priority (1-10, higher = more important)
            timeout: Maximum time to wait
            
        Returns:
            True if token acquired, False if rate limited
        """
        current_time = time.time()
        
        # Clean old request times
        cutoff_time = current_time - self.window_size
        while self._request_times and self._request_times[0] < cutoff_time:
            self._request_times.popleft()
        
        # Check current rate
        current_count = len(self._request_times)
        allowed_count = int(self._current_rate * self.window_size)
        
        # Priority adjustment (higher priority = more lenient)
        priority_multiplier = 1.0 + (priority - 5) * 0.1
        adjusted_allowed = int(allowed_count * priority_multiplier)
        
        if current_count >= adjusted_allowed:
            logger.debug(f"Rate limit exceeded: {current_count}/{adjusted_allowed}")
            return False
        
        # Acquire token
        self._request_times.append(current_time)
        return True
    
    def record_response_time(self, response_time: float):
        """Record response time for adaptive adjustment"""
        self._response_times.append(response_time)
        
        # Keep only recent response times
        if len(self._response_times) > 1000:
            self._response_times.popleft()
    
    def record_error(self):
        """Record error for adaptive adjustment"""
        self._error_count += 1
    
    def record_success(self):
        """Record success for adaptive adjustment"""
        self._success_count += 1
    
    async def adjust_rate(self, system_load: SystemLoad, cpu_usage: float, memory_usage: float):
        """
        Adjust rate limit based on system metrics
        
        Args:
            system_load: Current system load level
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage percentage (0-100)
        """
        async with self._adjustment_lock:
            # Calculate adjustment factor
            adjustment = 1.0
            
            # System load adjustment
            if system_load == SystemLoad.LOW:
                adjustment *= 1.2  # Increase rate
            elif system_load == SystemLoad.NORMAL:
                adjustment *= 1.0  # No change
            elif system_load == SystemLoad.HIGH:
                adjustment *= 0.8  # Decrease rate
            elif system_load == SystemLoad.CRITICAL:
                adjustment *= 0.5  # Significantly decrease
            
            # CPU usage adjustment
            if cpu_usage > 80:
                adjustment *= 0.7
            elif cpu_usage > 60:
                adjustment *= 0.9
            elif cpu_usage < 30:
                adjustment *= 1.1
            
            # Memory usage adjustment
            if memory_usage > 80:
                adjustment *= 0.8
            elif memory_usage < 30:
                adjustment *= 1.1
            
            # Response time adjustment
            if self._response_times:
                avg_response_time = sum(self._response_times) / len(self._response_times)
                if avg_response_time > 1.0:  # > 1 second
                    adjustment *= 0.8
                elif avg_response_time < 0.1:  # < 100ms
                    adjustment *= 1.1
            
            # Error rate adjustment
            total_requests = self._error_count + self._success_count
            if total_requests > 0:
                error_rate = self._error_count / total_requests
                if error_rate > 0.1:  # > 10% errors
                    adjustment *= 0.7
                elif error_rate < 0.01:  # < 1% errors
                    adjustment *= 1.05
            
            # Apply adjustment
            new_rate = self._current_rate * adjustment
            new_rate = max(self.min_rate, min(self.max_rate, new_rate))
            
            if abs(new_rate - self._current_rate) > 1.0:
                logger.info(f"Rate limit adjusted: {self._current_rate:.1f} -> {new_rate:.1f}/s")
                self._current_rate = new_rate
    
    def get_current_rate(self) -> float:
        """Get current rate limit"""
        return self._current_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        total_requests = self._error_count + self._success_count
        error_rate = (self._error_count / total_requests * 100) if total_requests > 0 else 0
        
        avg_response_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0
        )
        
        return {
            "current_rate": self._current_rate,
            "base_rate": self.base_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "current_requests": len(self._request_times),
            "error_count": self._error_count,
            "success_count": self._success_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time
        }


# Global limiter instance
_limiter: Optional[AdaptiveRateLimiter] = None


def get_adaptive_limiter() -> AdaptiveRateLimiter:
    """Get global adaptive rate limiter instance"""
    global _limiter
    if _limiter is None:
        _limiter = AdaptiveRateLimiter()
    return _limiter















