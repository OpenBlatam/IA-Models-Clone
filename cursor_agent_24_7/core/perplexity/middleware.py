"""
Perplexity Middleware - Middleware for request/response processing
==================================================================

Middleware components for logging, rate limiting, and monitoring.
"""

import time
import logging
from typing import Callable, Any, Dict
from functools import wraps
from .metrics import PerplexityMetrics
from .exceptions import PerplexityError

logger = logging.getLogger(__name__)


def timing_middleware(func: Callable) -> Callable:
    """
    Middleware to measure execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
            return result
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"{func.__name__} failed after {elapsed:.2f}ms: {e}")
            raise
    return wrapper


def error_handling_middleware(func: Callable) -> Callable:
    """
    Middleware for error handling and logging.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except PerplexityError as e:
            logger.error(f"Perplexity error in {func.__name__}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise PerplexityError(f"Error in {func.__name__}: {str(e)}") from e
    return wrapper


def metrics_middleware(metrics: PerplexityMetrics) -> Callable:
    """
    Create middleware that records metrics.
    
    Args:
        metrics: Metrics collector instance
        
    Returns:
        Middleware decorator
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                
                # Try to extract query type from result
                query_type = "general"
                if isinstance(result, dict):
                    query_type = result.get('query_type', 'general')
                
                # Record metrics if available
                if metrics:
                    try:
                        # Record query processing metrics
                        metrics.record_query_processing(
                            query_type=query_type,
                            duration_ms=elapsed,
                            success=True
                        )
                    except Exception as metric_error:
                        # Don't fail the request if metrics recording fails
                        logger.warning(f"Failed to record metrics: {metric_error}")
                
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"{func.__name__} error after {elapsed:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for query processing."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., user ID, IP)
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < self.window_seconds
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str = "default") -> int:
        """Get remaining requests in current window."""
        if key not in self.requests:
            return self.max_requests
        
        now = time.time()
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < self.window_seconds
        ]
        
        return max(0, self.max_requests - len(self.requests[key]))




