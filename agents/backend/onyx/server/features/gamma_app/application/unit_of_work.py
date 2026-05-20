"""
Unit of Work Pattern
Manages transactions and coordinates repository operations
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
from contextlib import asynccontextmanager

from ..infrastructure.database.session import DatabaseSessionManager, get_db_manager
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class IUnitOfWork(ABC):
    """Unit of Work interface"""
    
    @abstractmethod
    async def __aenter__(self):
        """Enter async context"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context"""
        pass
    
    @abstractmethod
    async def commit(self):
        """Commit the transaction"""
        pass
    
    @abstractmethod
    async def rollback(self):
        """Rollback the transaction"""
        pass

class UnitOfWork(IUnitOfWork):
    """Unit of Work implementation"""
    
    def __init__(self, db_manager: Optional[DatabaseSessionManager] = None):
        """Initialize Unit of Work"""
        self.db_manager = db_manager or get_db_manager()
        self.session: Optional[Session] = None
        self._repositories: dict = {}
    
    async def __aenter__(self):
        """Enter async context and start transaction"""
        self.session = self.db_manager.get_scoped_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and handle transaction"""
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()
        
        if self.session:
            self.session.close()
    
    async def commit(self):
        """Commit the transaction"""
        if self.session:
            try:
                self.session.commit()
                logger.debug("Transaction committed")
            except Exception as e:
                logger.error(f"Error committing transaction: {e}")
                await self.rollback()
                raise
    
    async def rollback(self):
        """Rollback the transaction"""
        if self.session:
            try:
                self.session.rollback()
                logger.debug("Transaction rolled back")
            except Exception as e:
                logger.error(f"Error rolling back transaction: {e}")
    
    def get_repository(self, repository_class: type):
        """Get or create a repository instance"""
        if repository_class not in self._repositories:
            self._repositories[repository_class] = repository_class(self.session)
        return self._repositories[repository_class]
    
    @property
    def session_context(self) -> Session:
        """Get the current session"""
        if self.session is None:
            raise RuntimeError("Unit of Work not initialized. Use as async context manager.")
        return self.session

@asynccontextmanager
async def unit_of_work():
    """Context manager for Unit of Work"""
    async with UnitOfWork() as uow:
        yield uow







