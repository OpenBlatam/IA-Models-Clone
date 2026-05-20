"""
Repository Pattern - Abstract data access
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class IRepository(ABC):
    """
    Interface for repositories
    """
    
    @abstractmethod
    def get(self, id: str) -> Optional[Any]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    def save(self, entity: Any) -> Any:
        """Save entity"""
        pass
    
    @abstractmethod
    def delete(self, id: str) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Find all entities matching filters"""
        pass


class BaseRepository(IRepository):
    """
    Base repository implementation
    """
    
    def __init__(self, name: str = "BaseRepository"):
        self.name = name
        self._storage: Dict[str, Any] = {}
    
    def get(self, id: str) -> Optional[Any]:
        """Get entity by ID"""
        return self._storage.get(id)
    
    def save(self, entity: Any) -> Any:
        """Save entity"""
        entity_id = self._get_entity_id(entity)
        self._storage[entity_id] = entity
        logger.debug(f"Saved entity {entity_id} in {self.name}")
        return entity
    
    def delete(self, id: str) -> bool:
        """Delete entity by ID"""
        if id in self._storage:
            del self._storage[id]
            logger.debug(f"Deleted entity {id} from {self.name}")
            return True
        return False
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Find all entities matching filters"""
        results = list(self._storage.values())
        
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    def _get_entity_id(self, entity: Any) -> str:
        """Extract ID from entity - override in subclasses"""
        if hasattr(entity, 'id'):
            return str(entity.id)
        elif isinstance(entity, dict):
            return str(entity.get('id', id(entity)))
        else:
            return str(id(entity))
    
    def _apply_filters(self, entities: List[Any], filters: Dict[str, Any]) -> List[Any]:
        """Apply filters to entities - override in subclasses"""
        filtered = []
        for entity in entities:
            match = True
            for key, value in filters.items():
                if hasattr(entity, key):
                    if getattr(entity, key) != value:
                        match = False
                        break
                elif isinstance(entity, dict):
                    if entity.get(key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered.append(entity)
        
        return filtered
    
    def clear(self):
        """Clear all entities"""
        self._storage.clear()
        logger.info(f"Cleared {self.name}")








