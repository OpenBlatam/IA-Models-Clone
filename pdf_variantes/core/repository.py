"""
PDF Variantes - Repository Pattern
Abstract repository interfaces for data access
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any
from uuid import UUID

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Base repository interface"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination"""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create new entity"""
        pass
    
    @abstractmethod
    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        pass


class DocumentRepository(BaseRepository):
    """Repository for PDF documents"""
    
    @abstractmethod
    async def get_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List:
        """Get documents by user ID"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List:
        """Get documents by status"""
        pass
    
    @abstractmethod
    async def search(self, query: str, user_id: Optional[str] = None) -> List:
        """Search documents"""
        pass


class VariantRepository(BaseRepository):
    """Repository for PDF variants"""
    
    @abstractmethod
    async def get_by_document(self, document_id: str, skip: int = 0, limit: int = 100) -> List:
        """Get variants by document ID"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str, document_id: Optional[str] = None) -> List:
        """Get variants by status"""
        pass


class TopicRepository(BaseRepository):
    """Repository for topics"""
    
    @abstractmethod
    async def get_by_document(self, document_id: str, min_relevance: float = 0.0) -> List:
        """Get topics by document ID"""
        pass
    
    @abstractmethod
    async def get_by_relevance(self, min_relevance: float, document_id: Optional[str] = None) -> List:
        """Get topics by minimum relevance score"""
        pass






