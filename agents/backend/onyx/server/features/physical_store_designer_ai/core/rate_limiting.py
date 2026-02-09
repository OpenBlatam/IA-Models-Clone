"""
Advanced rate limiting utilities and decorators
"""

from functools import wraps
from typing import Callable, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

from .logging_config import get_logger
from .exceptions import RateLimitError

logger = get_logger(__name__)


class EndpointRateLimiter:
    """Rate limiter for specific endpoints"""
    
    def __init__(self):
        self.limits: Dict[str, Dict[str, Any]] = {}
        self.requests: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque()))
        self._lock = asyncio.Lock()
    
    def set_limit(
        self,
        endpoint: str,
        requests_per_minute: int,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None
    ):
        """Set rate limit for an endpoint"""
        self.limits[endpoint] = {
            "per_minute": requests_per_minute,
            "per_hour": requests_per_hour,
            "per_day": requests_per_day
        }
    
    def get_limit(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get rate limit for an endpoint"""
        return self.limits.get(endpoint)
    
    async def check_limit(
        self,
        endpoint: str,
        client_id: str,
        now: Optional[datetime] = None
    ) -> tuple[bool, Optional[int]]:
        """Check if request is within rate limit"""
        if now is None:
            now = datetime.now()
        
        limit_config = self.limits.get(endpoint)
        if not limit_config:
            return True, None  # No limit set
        
        async with self._lock:
            client_requests = self.requests[endpoint][client_id]
            
            # Clean up old requests
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1) if limit_config.get("per_hour") else None
            cutoff_day = now - timedelta(days=1) if limit_config.get("per_day") else None
            
            # Remove expired entries
            while client_requests and client_requests[0] < cutoff_minute:
                client_requests.popleft()
            
            # Check per-minute limit
            per_minute = limit_config.get("per_minute")
            if per_minute:
                recent_minute = sum(1 for req_time in client_requests if req_time >= cutoff_minute)
                if recent_minute >= per_minute:
                    retry_after = 60 - (now - client_requests[0]).seconds if client_requests else 60
                    return False, retry_after
            
            # Check per-hour limit
            if cutoff_hour and limit_config.get("per_hour"):
                recent_hour = sum(1 for req_time in client_requests if req_time >= cutoff_hour)
                if recent_hour >= limit_config["per_hour"]:
                    retry_after = 3600 - (now - client_requests[0]).seconds if client_requests else 3600
                    return False, retry_after
            
            # Check per-day limit
            if cutoff_day and limit_config.get("per_day"):
                recent_day = sum(1 for req_time in client_requests if req_time >= cutoff_day)
                if recent_day >= limit_config["per_day"]:
                    retry_after = 86400 - (now - client_requests[0]).seconds if client_requests else 86400
                    return False, retry_after
            
            # Record request
            client_requests.append(now)
            return True, None
    
    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        if endpoint:
            return {
                "endpoint": endpoint,
                "limit": self.limits.get(endpoint),
                "active_clients": len(self.requests.get(endpoint, {}))
            }
        else:
            return {
                "endpoints": list(self.limits.keys()),
                "total_endpoints": len(self.limits),
                "total_clients": sum(len(clients) for clients in self.requests.values())
            }


# Global endpoint rate limiter
_endpoint_limiter: Optional[EndpointRateLimiter] = None


def get_endpoint_rate_limiter() -> EndpointRateLimiter:
    """Get global endpoint rate limiter"""
    global _endpoint_limiter
    if _endpoint_limiter is None:
        _endpoint_limiter = EndpointRateLimiter()
    return _endpoint_limiter


def rate_limit_endpoint(
    requests_per_minute: int,
    requests_per_hour: Optional[int] = None,
    requests_per_day: Optional[int] = None
):
    """
    Decorator to add rate limiting to a specific endpoint
    
    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_hour: Maximum requests per hour (optional)
        requests_per_day: Maximum requests per day (optional)
    """
    def decorator(func: Callable) -> Callable:
        endpoint_name = f"{func.__module__}.{func.__name__}"
        limiter = get_endpoint_rate_limiter()
        limiter.set_limit(endpoint_name, requests_per_minute, requests_per_hour, requests_per_day)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get client ID from request (if available)
            client_id = "unknown"
            for arg in args:
                if hasattr(arg, 'client') and arg.client:
                    client_id = arg.client.host
                    break
            
            # Check rate limit
            allowed, retry_after = await limiter.check_limit(endpoint_name, client_id)
            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {endpoint_name} from {client_id}",
                    extra={"endpoint": endpoint_name, "client_id": client_id}
                )
                raise RateLimitError(retry_after=retry_after)
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'd need to run in executor
            # This is a simplified version
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

