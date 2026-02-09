"""
Token Bucket
============

Token bucket rate limiting algorithm.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BucketState:
    """Token bucket state."""
    tokens: float
    last_refill: float
    capacity: float
    refill_rate: float  # tokens per second


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: Dict[str, BucketState] = {}
    
    def _get_bucket(self, key: str) -> BucketState:
        """Get or create bucket for key."""
        if key not in self._buckets:
            self._buckets[key] = BucketState(
                tokens=float(self.capacity),
                last_refill=time.time(),
                capacity=float(self.capacity),
                refill_rate=float(self.refill_rate)
            )
        
        return self._buckets[key]
    
    def _refill_tokens(self, bucket: BucketState):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - bucket.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now
    
    async def acquire(self, key: str, tokens: float = 1.0) -> bool:
        """Acquire tokens from bucket."""
        bucket = self._get_bucket(key)
        self._refill_tokens(bucket)
        
        if bucket.tokens >= tokens:
            bucket.tokens -= tokens
            return True
        
        return False
    
    def get_available_tokens(self, key: str) -> float:
        """Get available tokens for key."""
        bucket = self._get_bucket(key)
        self._refill_tokens(bucket)
        return bucket.tokens
    
    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        return {
            "total_buckets": len(self._buckets),
            "buckets": {
                key: {
                    "tokens": self.get_available_tokens(key),
                    "capacity": bucket.capacity,
                    "refill_rate": bucket.refill_rate
                }
                for key, bucket in self._buckets.items()
            }
        }















