"""
LLM Middleware - Circuit Breaker and Rate Limiting
"""

from typing import List, Optional
import logging

from ..utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError
from ..utils.rate_limiter import RateLimiter, RateLimitConfig, RateLimitExceededError
from .base import LLMMessage

logger = logging.getLogger(__name__)


class LLMMiddleware:
    """Middleware for LLM service with circuit breaker and rate limiting"""
    
    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
    
    async def before_generate(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None
    ) -> None:
        """Execute before generation"""
        # Rate limiting
        if self.rate_limiter:
            key = f"llm:{model or 'default'}"
            if not await self.rate_limiter.is_allowed(key):
                remaining = await self.rate_limiter.get_remaining(key)
                raise RateLimitExceededError(
                    f"Rate limit exceeded for model {model}",
                    retry_after=60
                )
    
    async def wrap_generate(self, func, *args, **kwargs):
        """Wrap generate function with middleware"""
        # Check rate limit
        if self.rate_limiter and 'messages' in kwargs:
            await self.before_generate(kwargs['messages'], kwargs.get('model'))
        
        # Circuit breaker
        if self.circuit_breaker:
            try:
                return await self.circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError as e:
                logger.error(f"Circuit breaker open: {e}")
                raise
        else:
            return await func(*args, **kwargs)
    
    def get_remaining_requests(self, model: Optional[str] = None) -> Optional[int]:
        """Get remaining requests in rate limit window"""
        if self.rate_limiter:
            key = f"llm:{model or 'default'}"
            return self.rate_limiter.get_remaining(key)
        return None

