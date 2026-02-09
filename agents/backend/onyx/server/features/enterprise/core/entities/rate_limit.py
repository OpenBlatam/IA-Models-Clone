from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from typing import Optional
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Rate Limit Info Entity
======================

Domain entity for rate limiting information.
"""



@dataclass
class RateLimitInfo:
    """Rate limiting information for a request."""
    
    allowed: bool
    requests_remaining: int
    window_size: int
    current_requests: int
    retry_after: Optional[int] = None
    rate_limit_active: bool = True
    
    @classmethod
    def create_allowed(cls, remaining: int, window_size: int, current: int) -> "RateLimitInfo":
        """Create info for allowed request."""
        return cls(
            allowed=True,
            requests_remaining=remaining,
            window_size=window_size,
            current_requests=current
        )
    
    @classmethod
    def create_denied(cls, window_size: int, current: int, retry_after: int) -> "RateLimitInfo":
        """Create info for denied request."""
        return cls(
            allowed=False,
            requests_remaining=0,
            window_size=window_size,
            current_requests=current,
            retry_after=retry_after
        )
    
    @classmethod
    def create_inactive(cls) -> "RateLimitInfo":
        """Create info when rate limiting is inactive."""
        return cls(
            allowed=True,
            requests_remaining=0,
            window_size=0,
            current_requests=0,
            rate_limit_active=False
        )
    
    def get_headers(self) -> dict:
        """Get HTTP headers for rate limit info."""
        headers = {}
        
        if self.rate_limit_active:
            headers["X-RateLimit-Remaining"] = str(self.requests_remaining)
            headers["X-RateLimit-Window"] = str(self.window_size)
            
            if not self.allowed and self.retry_after:
                headers["Retry-After"] = str(self.retry_after)
        
        return headers 