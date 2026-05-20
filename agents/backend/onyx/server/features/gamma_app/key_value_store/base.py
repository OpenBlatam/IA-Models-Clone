"""
Key-Value Store Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import timedelta


class KVStore:
    """Key-value store definition"""
    
    def __init__(self, backend: str = "redis"):
        self.backend = backend


class KVStoreBase(ABC):
    """Base interface for key-value store"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value by key"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

