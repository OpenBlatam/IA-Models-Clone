"""
Rate Limiter
============

Token bucket rate limiting implementation.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Features:
    - Per-client rate limiting
    - Configurable rate limits
    - Burst support
    """
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str, config: Optional[RateLimitConfig] = None) -> bool:
        """
        Check if request is allowed.
        
        Args:
            client_id: Client identifier
            config: Optional rate limit config override
            
        Returns:
            True if allowed, False otherwise
        """
        config = config or self.default_config
        
        async with self._lock:
            now = time.time()
            
            # Get or create bucket
            if client_id not in self._buckets:
                self._buckets[client_id] = {
                    "tokens": config.burst_size,
                    "last_update": now
                }
            
            bucket = self._buckets[client_id]
            
            # Add tokens based on elapsed time
            elapsed = now - bucket["last_update"]
            tokens_to_add = elapsed * config.requests_per_second
            bucket["tokens"] = min(
                config.burst_size,
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_update"] = now
            
            # Check if we have tokens
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            
            return False
    
    async def get_wait_time(self, client_id: str, config: Optional[RateLimitConfig] = None) -> float:
        """
        Get wait time until next request is allowed.
        
        Args:
            client_id: Client identifier
            config: Optional rate limit config override
            
        Returns:
            Wait time in seconds
        """
        config = config or self.default_config
        
        async with self._lock:
            if client_id not in self._buckets:
                return 0.0
            
            bucket = self._buckets[client_id]
            tokens_needed = 1.0 - bucket["tokens"]
            
            if tokens_needed <= 0:
                return 0.0
            
            return tokens_needed / config.requests_per_second
    
    async def reset(self, client_id: str):
        """Reset rate limit for a client."""
        async with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]
    
    async def cleanup_old_buckets(self, max_age_seconds: int = 3600):
        """Clean up old buckets."""
        async with self._lock:
            now = time.time()
            to_remove = []
            
            for client_id, bucket in self._buckets.items():
                age = now - bucket["last_update"]
                if age > max_age_seconds:
                    to_remove.append(client_id)
            
            for client_id in to_remove:
                del self._buckets[client_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old rate limit buckets")

