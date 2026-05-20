from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from ..entities.rate_limit import RateLimitInfo
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Rate Limit Service Interface
============================

Abstract interface for rate limiting operations.
"""



class IRateLimitService(ABC):
    """Abstract interface for rate limiting operations."""
    
    @abstractmethod
    async def is_allowed(self, identifier: str) -> RateLimitInfo:
        """Check if request is allowed under rate limit."""
        pass
    
    @abstractmethod
    async def reset_limit(self, identifier: str) -> bool:
        """Reset rate limit for identifier."""
        pass
    
    @abstractmethod
    async async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        pass
    
    @abstractmethod
    async def get_window_info(self, identifier: str) -> dict:
        """Get window information for identifier."""
        pass 