from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID
from pydantic import BaseModel
from ..entities.base import Entity
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Repository Interface
========================

Base repository interface defining common operations for all repositories.
"""




T = TypeVar("T", bound=Entity)


class BaseRepository(ABC, Generic[T]):
    """
    Base repository interface for all entities.
    
    Defines common CRUD operations and query methods that all
    repositories must implement.
    """
    
    def __init__(self) -> None:
        """Initialize repository."""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: Entity to create
            
        Returns:
            Created entity with ID
            
        Raises:
            RepositoryError: If creation fails
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity if found, None otherwise
            
        Raises:
            RepositoryError: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: Entity to update
            
        Returns:
            Updated entity
            
        Raises:
            RepositoryError: If update fails
            EntityNotFoundError: If entity doesn't exist
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: Entity ID to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            RepositoryError: If deletion fails
        """
        pass
    
    @abstractmethod
    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> List[T]:
        """
        List entities with pagination and filtering.
        
        Args:
            skip: Number of entities to skip
            limit: Maximum number of entities to return
            filters: Filter criteria
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            List of entities
            
        Raises:
            RepositoryError: If listing fails
        """
        pass
    
    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching filters.
        
        Args:
            filters: Filter criteria
            
        Returns:
            Number of matching entities
            
        Raises:
            RepositoryError: If count fails
        """
        pass
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Entity ID to check
            
        Returns:
            True if entity exists, False otherwise
            
        Raises:
            RepositoryError: If check fails
        """
        pass
    
    async def get_by_ids(self, entity_ids: List[UUID]) -> List[T]:
        """
        Get multiple entities by IDs.
        
        Args:
            entity_ids: List of entity IDs
            
        Returns:
            List of found entities
            
        Raises:
            RepositoryError: If retrieval fails
        """
        entities = []
        for entity_id in entity_ids:
            entity = await self.get_by_id(entity_id)
            if entity:
                entities.append(entity)
        return entities
    
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in bulk.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
            
        Raises:
            RepositoryError: If bulk creation fails
        """
        created_entities = []
        for entity in entities:
            created_entity = await self.create(entity)
            created_entities.append(created_entity)
        return created_entities
    
    async def bulk_update(self, entities: List[T]) -> List[T]:
        """
        Update multiple entities in bulk.
        
        Args:
            entities: List of entities to update
            
        Returns:
            List of updated entities
            
        Raises:
            RepositoryError: If bulk update fails
        """
        updated_entities = []
        for entity in entities:
            updated_entity = await self.update(entity)
            updated_entities.append(updated_entity)
        return updated_entities
    
    async def bulk_delete(self, entity_ids: List[UUID]) -> int:
        """
        Delete multiple entities in bulk.
        
        Args:
            entity_ids: List of entity IDs to delete
            
        Returns:
            Number of deleted entities
            
        Raises:
            RepositoryError: If bulk deletion fails
        """
        deleted_count = 0
        for entity_id in entity_ids:
            if await self.delete(entity_id):
                deleted_count += 1
        return deleted_count
    
    async def find_one(self, filters: Dict[str, Any]) -> Optional[T]:
        """
        Find a single entity matching filters.
        
        Args:
            filters: Filter criteria
            
        Returns:
            First matching entity or None
            
        Raises:
            RepositoryError: If search fails
        """
        entities = await self.list(skip=0, limit=1, filters=filters)
        return entities[0] if entities else None
    
    async def find_many(self, filters: Dict[str, Any], limit: int = 100) -> List[T]:
        """
        Find multiple entities matching filters.
        
        Args:
            filters: Filter criteria
            limit: Maximum number of entities to return
            
        Returns:
            List of matching entities
            
        Raises:
            RepositoryError: If search fails
        """
        return await self.list(skip=0, limit=limit, filters=filters)
    
    async def save(self, entity: T) -> T:
        """
        Save entity (create or update).
        
        Args:
            entity: Entity to save
            
        Returns:
            Saved entity
            
        Raises:
            RepositoryError: If save fails
        """
        if await self.exists(entity.id):
            return await self.update(entity)
        else:
            return await self.create(entity)
    
    async def get_or_create(
        self,
        filters: Dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None
    ) -> tuple[T, bool]:
        """
        Get existing entity or create new one.
        
        Args:
            filters: Filter criteria to find existing entity
            defaults: Default values for new entity
            
        Returns:
            Tuple of (entity, created) where created is True if new entity
            
        Raises:
            RepositoryError: If operation fails
        """
        existing = await self.find_one(filters)
        if existing:
            return existing, False
        
        # Create new entity with defaults
        if defaults:
            # This is a simplified version - actual implementation would need
            # to know how to create the specific entity type
            raise NotImplementedError("get_or_create with defaults not implemented")
        
        raise ValueError("Cannot create entity without defaults")
    
    async def update_or_create(
        self,
        filters: Dict[str, Any],
        defaults: Dict[str, Any]
    ) -> tuple[T, bool]:
        """
        Update existing entity or create new one.
        
        Args:
            filters: Filter criteria to find existing entity
            defaults: Values to update/create
            
        Returns:
            Tuple of (entity, created) where created is True if new entity
            
        Raises:
            RepositoryError: If operation fails
        """
        existing = await self.find_one(filters)
        if existing:
            # Update existing entity
            for key, value in defaults.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            updated = await self.update(existing)
            return updated, False
        else:
            # Create new entity
            # This is a simplified version - actual implementation would need
            # to know how to create the specific entity type
            raise NotImplementedError("update_or_create not implemented")
    
    async def transaction(self) -> Any:
        """
        Get transaction context manager.
        
        Returns:
            Transaction context manager
            
        Raises:
            RepositoryError: If transaction not supported
        """
        raise NotImplementedError("Transactions not supported by this repository")
    
    async def close(self) -> None:
        """
        Close repository connections.
        
        Should be called when repository is no longer needed.
        """
        pass
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        await self.close()


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class EntityNotFoundError(RepositoryError):
    """Exception raised when entity is not found."""
    pass


class DuplicateEntityError(RepositoryError):
    """Exception raised when trying to create duplicate entity."""
    pass


class ValidationError(RepositoryError):
    """Exception raised when entity validation fails."""
    pass 