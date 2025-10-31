from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Any, Optional
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Cache Service Interface
======================

Abstract interface for caching operations following dependency inversion principle.
"""



class ICacheService(ABC):
    """Abstract interface for cache operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        pass 