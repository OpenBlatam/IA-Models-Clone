"""
Rate Limiter - Request rate limiting for webhook endpoints
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int = 100
    window_seconds: int = 60  # 1 minute window
    burst_allowance: int = 10  # Allow burst of 10 requests


class RateLimiter:
    """
    Sliding window rate limiter for webhook endpoints
    Uses sliding window algorithm for accurate rate limiting
    """
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter
        
        Args:
            default_config: Default rate limit configuration
        """
        self._configs: Dict[str, RateLimitConfig] = {}
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
        
        self.default_config = default_config or RateLimitConfig()
    
    def configure_endpoint(
        self,
        endpoint_id: str,
        max_requests: int,
        window_seconds: int = 60,
        burst_allowance: int = 10
    ) -> None:
        """
        Configure rate limit for an endpoint
        
        Args:
            endpoint_id: Endpoint identifier
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds
            burst_allowance: Burst allowance
        """
        self._configs[endpoint_id] = RateLimitConfig(
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst_allowance=burst_allowance
        )
        logger.debug(f"Rate limit configured for {endpoint_id}: {max_requests}/{window_seconds}s")
    
    async def is_allowed(self, endpoint_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        async with self._lock:
            config = self._configs.get(endpoint_id, self.default_config)
            window = self._windows[endpoint_id]
            
            current_time = time.time()
            window_start = current_time - config.window_seconds
            
            # Clean old entries from window
            while window and window[0] < window_start:
                window.popleft()
            
            # Check if limit exceeded
            current_count = len(window)
            
            if current_count >= config.max_requests:
                # Calculate time until next request is allowed
                oldest_request = window[0] if window else current_time
                retry_after = window_start + config.window_seconds - oldest_request
                
                return False, f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"
            
            # Record this request
            window.append(current_time)
            
            return True, None
    
    async def record_request(self, endpoint_id: str) -> None:
        """
        Record a request (same as is_allowed but for logging)
        
        Args:
            endpoint_id: Endpoint identifier
        """
        await self.is_allowed(endpoint_id)
    
    def get_rate_limit_status(self, endpoint_id: str) -> Dict:
        """
        Get current rate limit status for endpoint
        
        Args:
            endpoint_id: Endpoint identifier
            
        Returns:
            Status dictionary
        """
        config = self._configs.get(endpoint_id, self.default_config)
        window = self._windows[endpoint_id]
        
        current_time = time.time()
        window_start = current_time - config.window_seconds
        
        # Clean old entries
        while window and window[0] < window_start:
            window.popleft()
        
        current_count = len(window)
        remaining = max(0, config.max_requests - current_count)
        
        # Calculate requests per second
        if config.window_seconds > 0:
            rps = current_count / config.window_seconds
        else:
            rps = 0
        
        return {
            "endpoint_id": endpoint_id,
            "current_requests": current_count,
            "max_requests": config.max_requests,
            "remaining": remaining,
            "window_seconds": config.window_seconds,
            "requests_per_second": round(rps, 2),
            "limit_exceeded": current_count >= config.max_requests,
            "burst_allowance": config.burst_allowance
        }
    
    def reset_endpoint(self, endpoint_id: str) -> None:
        """
        Reset rate limit for an endpoint
        
        Args:
            endpoint_id: Endpoint identifier
        """
        self._windows[endpoint_id].clear()
        logger.debug(f"Rate limit reset for {endpoint_id}")
    
    def reset_all(self) -> None:
        """Reset all rate limits"""
        self._windows.clear()
        logger.debug("All rate limits reset")






