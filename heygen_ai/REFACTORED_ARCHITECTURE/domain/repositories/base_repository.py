"""
Base Repository Interface

This module provides the base repository interface that all repositories inherit from.
It defines the common CRUD operations that repositories should implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID

from ..entities.base_entity import BaseEntity

T = TypeVar('T', bound=BaseEntity)


class BaseRepository(ABC, Generic[T]):
    """
    Base repository interface that defines common CRUD operations.
    
    This interface provides:
    - Create operations
    - Read operations (get by ID, get all, search)
    - Update operations
    - Delete operations
    - Count operations
    """
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to retrieve
            
        Returns:
            The entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all entities with pagination.
        
        Args:
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            
        Returns:
            List of entities
        """
        pass
    
    @abstractmethod
    async def search(self, filters: Dict[str, Any], skip: int = 0, limit: int = 100) -> List[T]:
        """
        Search for entities based on filters.
        
        Args:
            filters: Dictionary of filters to apply
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            
        Returns:
            List of entities matching the filters
        """
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: The entity to create
            
        Returns:
            The created entity
        """
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: The entity to update
            
        Returns:
            The updated entity
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """
        Check if an entity exists by its ID.
        
        Args:
            entity_id: The ID of the entity to check
            
        Returns:
            True if the entity exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities based on optional filters.
        
        Args:
            filters: Optional dictionary of filters to apply
            
        Returns:
            Number of entities matching the filters
        """
        pass
    
    @abstractmethod
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in a single operation.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
        """
        pass
    
    @abstractmethod
    async def bulk_update(self, entities: List[T]) -> List[T]:
        """
        Update multiple entities in a single operation.
        
        Args:
            entities: List of entities to update
            
        Returns:
            List of updated entities
        """
        pass
    
    @abstractmethod
    async def bulk_delete(self, entity_ids: List[UUID]) -> int:
        """
        Delete multiple entities by their IDs.
        
        Args:
            entity_ids: List of entity IDs to delete
            
        Returns:
            Number of entities deleted
        """
        pass