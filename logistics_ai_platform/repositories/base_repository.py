"""
Base repository with common CRUD operations

This module provides a base repository class that implements common
CRUD operations to reduce code duplication across repositories.
"""

import logging
from typing import TypeVar, Generic, Optional, List, Dict, Callable, Any
from datetime import datetime

T = TypeVar('T')
ID_FIELD = "id"

logger = logging.getLogger(__name__)


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations
    
    Provides standard methods for save, find_by_id, delete, count, etc.
    Subclasses can extend with domain-specific methods.
    """
    
    def __init__(self, entity_name: str = "Entity"):
        """
        Initialize base repository
        
        Args:
            entity_name: Name of entity type (for logging)
        """
        self._storage: Dict[str, T] = {}
        self.entity_name = entity_name
        logger.info(f"{entity_name}Repository initialized")
    
    def _get_id(self, entity: T) -> Optional[str]:
        """
        Extract ID from entity
        
        Args:
            entity: Entity instance
            
        Returns:
            Entity ID or None
        """
        if hasattr(entity, 'id'):
            return str(getattr(entity, 'id'))
        for attr in ['quote_id', 'booking_id', 'shipment_id', 'container_id']:
            if hasattr(entity, attr):
                return str(getattr(entity, attr))
        return None
    
    async def save(self, entity: T) -> T:
        """
        Save an entity
        
        Args:
            entity: Entity to save
            
        Returns:
            Saved entity
            
        Raises:
            ValueError: If entity or ID is missing
        """
        entity_id = self._get_id(entity)
        if not entity or not entity_id:
            raise ValueError(f"{self.entity_name} and ID are required")
        
        if hasattr(entity, 'updated_at'):
            setattr(entity, 'updated_at', datetime.now())
        
        self._storage[entity_id] = entity
        logger.debug(f"{self.entity_name} saved: {entity_id}")
        return entity
    
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """
        Find entity by ID
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity if found, None otherwise
        """
        if not entity_id:
            return None
        
        entity = self._storage.get(entity_id)
        if entity:
            logger.debug(f"{self.entity_name} found: {entity_id}")
        else:
            logger.debug(f"{self.entity_name} not found: {entity_id}")
        return entity
    
    async def find_all(
        self,
        filter_func: Optional[Callable[[T], bool]] = None,
        sort_key: Optional[Callable[[T], Any]] = None,
        reverse: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[T]:
        """
        Find all entities with optional filtering and sorting
        
        Args:
            filter_func: Optional filter function
            sort_key: Optional sort key function
            reverse: Sort in reverse order
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of entities
        """
        entities = list(self._storage.values())
        
        if filter_func:
            entities = [e for e in entities if filter_func(e)]
        
        if sort_key:
            entities.sort(key=sort_key, reverse=reverse)
        
        result = entities[offset:offset + limit]
        logger.debug(f"Found {len(result)} {self.entity_name.lower()}s")
        return result
    
    async def find_by_field(
        self,
        field_name: str,
        value: Any,
        case_sensitive: bool = False
    ) -> Optional[T]:
        """
        Find entity by field value
        
        Args:
            field_name: Field name to search
            value: Value to match
            case_sensitive: Whether comparison is case-sensitive
            
        Returns:
            Entity if found, None otherwise
        """
        if not value:
            return None
        
        search_value = value if case_sensitive else str(value).upper()
        
        for entity in self._storage.values():
            field_value = getattr(entity, field_name, None)
            if field_value is None:
                continue
            
            compare_value = field_value if case_sensitive else str(field_value).upper()
            if compare_value == search_value:
                logger.debug(f"{self.entity_name} found by {field_name}: {value}")
                return entity
        
        logger.debug(f"{self.entity_name} not found by {field_name}: {value}")
        return None
    
    async def find_all_by_field(
        self,
        field_name: str,
        value: Any
    ) -> List[T]:
        """
        Find all entities by field value
        
        Args:
            field_name: Field name to search
            value: Value to match
            
        Returns:
            List of matching entities
        """
        if not value:
            return []
        
        entities = [
            entity for entity in self._storage.values()
            if getattr(entity, field_name, None) == value
        ]
        
        logger.debug(f"Found {len(entities)} {self.entity_name.lower()}s by {field_name}: {value}")
        return entities
    
    async def count(self) -> int:
        """
        Get total number of entities
        
        Returns:
            Total count
        """
        count = len(self._storage)
        logger.debug(f"Total {self.entity_name.lower()}s: {count}")
        return count
    
    async def count_by_field(
        self,
        field_name: str,
        value: Any
    ) -> int:
        """
        Count entities by field value
        
        Args:
            field_name: Field name to search
            value: Value to match
            
        Returns:
            Count of matching entities
        """
        count = sum(
            1 for entity in self._storage.values()
            if getattr(entity, field_name, None) == value
        )
        logger.debug(f"{self.entity_name}s with {field_name}={value}: {count}")
        return count
    
    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if deleted, False if not found
        """
        if entity_id in self._storage:
            del self._storage[entity_id]
            logger.info(f"{self.entity_name} deleted: {entity_id}")
            return True
        
        logger.debug(f"{self.entity_name} not found for deletion: {entity_id}")
        return False
    
    async def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if exists, False otherwise
        """
        exists = entity_id in self._storage
        logger.debug(f"{self.entity_name} {entity_id} exists: {exists}")
        return exists







