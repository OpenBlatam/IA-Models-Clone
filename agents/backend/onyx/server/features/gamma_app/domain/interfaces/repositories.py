"""
Repository Interfaces
Abstract base classes for repository pattern
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any
from uuid import UUID

T = TypeVar('T')

class IRepository(Generic[T], ABC):
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
        """Create a new entity"""
        pass
    
    @abstractmethod
    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update an existing entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID"""
        pass
    
    @abstractmethod
    async def find_by(self, **filters: Any) -> List[T]:
        """Find entities by filters"""
        pass

class IUserRepository(IRepository, ABC):
    """User repository interface"""
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[Any]:
        """Get user by email"""
        pass
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[Any]:
        """Get user by username"""
        pass

class IContentRepository(IRepository, ABC):
    """Content repository interface"""
    
    @abstractmethod
    async def get_by_user_id(self, user_id: str, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get content by user ID"""
        pass
    
    @abstractmethod
    async def get_by_project_id(self, project_id: str) -> List[Any]:
        """Get content by project ID"""
        pass
    
    @abstractmethod
    async def get_by_content_type(self, content_type: str, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get content by type"""
        pass

class IProjectRepository(IRepository, ABC):
    """Project repository interface"""
    
    @abstractmethod
    async def get_by_owner_id(self, owner_id: str) -> List[Any]:
        """Get projects by owner ID"""
        pass
    
    @abstractmethod
    async def get_public_projects(self, skip: int = 0, limit: int = 100) -> List[Any]:
        """Get public projects"""
        pass

class IAnalyticsRepository(IRepository, ABC):
    """Analytics repository interface"""
    
    @abstractmethod
    async def track_event(self, event_type: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """Track an analytics event"""
        pass
    
    @abstractmethod
    async def get_user_metrics(self, user_id: str, start_date: Optional[Any], end_date: Optional[Any]) -> Dict[str, Any]:
        """Get user metrics"""
        pass







