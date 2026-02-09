"""
Token Bucket

Token bucket algorithm for rate limiting.
"""

import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 1.0
    ):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens
            refill_rate: Tokens per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = Lock()
    
    def consume(
        self,
        tokens: int = 1
    ) -> bool:
        """
        Consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens available, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        
        self.last_refill = now
    
    def get_tokens(self) -> float:
        """Get current token count."""
        with self.lock:
            self._refill()
            return self.tokens
    
    def wait_for_tokens(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
        """
        while not self.consume(tokens):
            time.sleep(0.1)


def create_token_bucket(
    capacity: int = 100,
    refill_rate: float = 1.0
) -> TokenBucket:
    """Create token bucket."""
    return TokenBucket(capacity, refill_rate)



