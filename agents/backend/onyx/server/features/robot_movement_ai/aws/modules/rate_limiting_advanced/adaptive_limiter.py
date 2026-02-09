"""
Adaptive Limiter
================

Adaptive rate limiting based on system load.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Adaptive limiter configuration."""
    base_rate: float
    min_rate: float
    max_rate: float
    load_threshold: float = 0.8
    adjustment_factor: float = 0.1


class AdaptiveLimiter:
    """Adaptive rate limiter."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self._current_rate = config.base_rate
        self._request_times: list[float] = []
        self._system_load: float = 0.0
    
    def record_request(self):
        """Record request timestamp."""
        self._request_times.append(time.time())
        
        # Keep only recent requests
        cutoff = time.time() - 60.0  # Last minute
        self._request_times = [t for t in self._request_times if t > cutoff]
    
    def update_system_load(self, load: float):
        """Update system load metric."""
        self._system_load = load
        self._adjust_rate()
    
    def _adjust_rate(self):
        """Adjust rate based on system load."""
        if self._system_load > self.config.load_threshold:
            # Reduce rate
            self._current_rate = max(
                self.config.min_rate,
                self._current_rate * (1 - self.config.adjustment_factor)
            )
        else:
            # Increase rate
            self._current_rate = min(
                self.config.max_rate,
                self._current_rate * (1 + self.config.adjustment_factor)
            )
    
    async def acquire(self, key: str) -> bool:
        """Acquire request slot."""
        self.record_request()
        
        # Calculate current rate
        if len(self._request_times) >= 2:
            time_span = self._request_times[-1] - self._request_times[0]
            if time_span > 0:
                current_rate = len(self._request_times) / time_span
                
                if current_rate >= self._current_rate:
                    return False
        
        return True
    
    def get_current_rate(self) -> float:
        """Get current rate limit."""
        return self._current_rate
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive limiter statistics."""
        return {
            "current_rate": self._current_rate,
            "base_rate": self.config.base_rate,
            "min_rate": self.config.min_rate,
            "max_rate": self.config.max_rate,
            "system_load": self._system_load,
            "recent_requests": len(self._request_times)
        }















