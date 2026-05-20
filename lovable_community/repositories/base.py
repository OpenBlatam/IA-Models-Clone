"""
Base Repository

Abstract base class for all repositories following Repository Pattern.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
import logging

from .query_helpers import apply_ordering, execute_query_with_pagination
from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Base repository class providing common CRUD operations.
    
    Follows Repository Pattern to abstract data access layer.
    """
    
    def __init__(self, db: Session, model_class: type[T]):
        """
        Initialize repository.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
        """
        self.db = db
        self.model_class = model_class
    
    def create(self, **kwargs) -> T:
        """
        Create a new entity.
        
        Args:
            **kwargs: Entity attributes
            
        Returns:
            Created entity
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            entity = self.model_class(**kwargs)
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating {self.model_class.__name__}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create {self.model_class.__name__}: {str(e)}") from e
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity or None if not found
        """
        return self.db.query(self.model_class).filter(
            self.model_class.id == entity_id
        ).first()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        order_direction: str = "desc"
    ) -> List[T]:
        """
        Get all entities with pagination and ordering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field to order by
            order_direction: Order direction (asc/desc)
            
        Returns:
            List of entities
        """
        query = self.db.query(self.model_class)
        return execute_query_with_pagination(
            query, skip, limit, order_by, order_direction, self.model_class
        )
    
    def update(self, entity_id: str, **kwargs) -> Optional[T]:
        """
        Update entity by ID.
        
        Args:
            entity_id: Entity ID
            **kwargs: Attributes to update
            
        Returns:
            Updated entity or None if not found
            
        Raises:
            DatabaseError: If update fails
        """
        entity = self.get_by_id(entity_id)
        if not entity:
            return None
        
        try:
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating {self.model_class.__name__} {entity_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update {self.model_class.__name__}: {str(e)}") from e
    
    def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseError: If deletion fails
        """
        entity = self.get_by_id(entity_id)
        if not entity:
            return False
        
        try:
            self.db.delete(entity)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting {self.model_class.__name__} {entity_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete {self.model_class.__name__}: {str(e)}") from e
    
    def count(self, **filters) -> int:
        """
        Count entities matching filters.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            Count of matching entities
        """
        query = self.db.query(self.model_class)
        
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.filter(getattr(self.model_class, key) == value)
        
        return query.count()
    
    def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists (optimized).
        
        Uses count query instead of fetching full entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            True if exists, False otherwise
        """
        return self.db.query(self.model_class).filter(
            self.model_class.id == entity_id
        ).count() > 0



