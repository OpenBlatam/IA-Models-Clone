"""
Base Repository Implementation
Common repository functionality
"""

from typing import Generic, TypeVar, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ...domain.interfaces.repositories import IRepository

T = TypeVar('T')

class BaseRepository(Generic[T], IRepository[T]):
    """Base repository implementation"""
    
    def __init__(self, session: Session, model_class: type):
        """Initialize repository with session and model class"""
        self.session = session
        self.model_class = model_class
    
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination"""
        return self.session.query(self.model_class).offset(skip).limit(limit).all()
    
    async def create(self, entity: T) -> T:
        """Create a new entity"""
        self.session.add(entity)
        self.session.flush()
        return entity
    
    async def update(self, id: str, entity: T) -> Optional[T]:
        """Update an existing entity"""
        existing = await self.get_by_id(id)
        if existing:
            for key, value in entity.__dict__.items():
                if not key.startswith('_') and hasattr(existing, key):
                    setattr(existing, key, value)
            self.session.flush()
            return existing
        return None
    
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID"""
        entity = await self.get_by_id(id)
        if entity:
            self.session.delete(entity)
            self.session.flush()
            return True
        return False
    
    async def find_by(self, **filters: Any) -> List[T]:
        """Find entities by filters"""
        query = self.session.query(self.model_class)
        conditions = []
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                conditions.append(getattr(self.model_class, key) == value)
        
        if conditions:
            query = query.filter(and_(*conditions))
        
        return query.all()







