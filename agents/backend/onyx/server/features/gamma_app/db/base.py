"""
Database Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from contextlib import asynccontextmanager


class Transaction:
    """Transaction context"""
    
    def __init__(self, connection: Any):
        self.connection = connection
        self.committed = False
        self.rolled_back = False


class DatabaseConnection:
    """Database connection"""
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool: Optional[Any] = None


class DatabaseBase(ABC):
    """Base interface for database"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from database"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute query"""
        pass
    
    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager"""
        pass
    
    @abstractmethod
    async def migrate(self) -> bool:
        """Run migrations"""
        pass

