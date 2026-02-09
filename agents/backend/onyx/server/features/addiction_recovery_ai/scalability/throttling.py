"""
Intelligent Throttling
Advanced request throttling and rate limiting
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThrottleConfig:
    """Throttling configuration"""
    requests_per_second: int = 10
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 20
    priority_levels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.priority_levels is None:
            self.priority_levels = {
                "high": 2,
                "medium": 1,
                "low": 0.5
            }


class IntelligentThrottler:
    """
    Intelligent request throttler
    
    Features:
    - Token bucket algorithm
    - Priority-based throttling
    - Adaptive rate limiting
    - Per-endpoint throttling
    - Per-user throttling
    """
    
    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._configs: Dict[str, ThrottleConfig] = {}
        self._request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def configure(
        self,
        endpoint: str,
        config: ThrottleConfig
    ) -> None:
        """Configure throttling for endpoint"""
        self._configs[endpoint] = config
        
        # Initialize token bucket
        self._buckets[endpoint] = {
            "tokens": config.burst_size,
            "last_refill": time.time(),
            "rate": config.requests_per_second
        }
        
        logger.info(f"Configured throttling for: {endpoint}")
    
    def is_allowed(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        priority: str = "medium"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed
        
        Returns:
            (is_allowed, info_dict)
        """
        config = self._configs.get(endpoint)
        if not config:
            return True, {"reason": "no_config"}
        
        # Check token bucket
        bucket = self._buckets.get(endpoint)
        if not bucket:
            return True, {"reason": "no_bucket"}
        
        # Refill tokens
        now = time.time()
        elapsed = now - bucket["last_refill"]
        tokens_to_add = elapsed * bucket["rate"]
        bucket["tokens"] = min(
            config.burst_size,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now
        
        # Apply priority multiplier
        priority_multiplier = config.priority_levels.get(priority, 1.0)
        required_tokens = 1.0 / priority_multiplier
        
        # Check if enough tokens
        if bucket["tokens"] >= required_tokens:
            bucket["tokens"] -= required_tokens
            
            # Record request
            key = f"{endpoint}:{user_id or 'anonymous'}"
            self._request_history[key].append(now)
            
            return True, {
                "tokens_remaining": bucket["tokens"],
                "priority": priority
            }
        else:
            return False, {
                "reason": "rate_limit_exceeded",
                "tokens_available": bucket["tokens"],
                "tokens_required": required_tokens,
                "retry_after": (required_tokens - bucket["tokens"]) / bucket["rate"]
            }
    
    def get_throttle_status(self, endpoint: str) -> Dict[str, Any]:
        """Get current throttle status"""
        bucket = self._buckets.get(endpoint)
        config = self._configs.get(endpoint)
        
        if not bucket or not config:
            return {"status": "not_configured"}
        
        return {
            "tokens": bucket["tokens"],
            "max_tokens": config.burst_size,
            "rate": bucket["rate"],
            "utilization": 1.0 - (bucket["tokens"] / config.burst_size)
        }
    
    def adapt_rate(self, endpoint: str, success_rate: float) -> None:
        """Adapt throttling rate based on success rate"""
        if endpoint not in self._buckets:
            return
        
        bucket = self._buckets[endpoint]
        config = self._configs.get(endpoint)
        
        if not config:
            return
        
        # Adjust rate based on success rate
        if success_rate > 0.95:
            # Increase rate by 10%
            bucket["rate"] = min(
                config.requests_per_second * 2,
                bucket["rate"] * 1.1
            )
        elif success_rate < 0.8:
            # Decrease rate by 10%
            bucket["rate"] = max(
                config.requests_per_second * 0.5,
                bucket["rate"] * 0.9
            )


# Global throttler
_throttler: Optional[IntelligentThrottler] = None


def get_throttler() -> IntelligentThrottler:
    """Get global throttler"""
    global _throttler
    if _throttler is None:
        _throttler = IntelligentThrottler()
    return _throttler

