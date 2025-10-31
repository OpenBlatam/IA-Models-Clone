"""
Base Entity for Domain Layer

This module provides the base entity class that all domain entities inherit from.
It provides common attributes and methods that are shared across all entities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4, UUID


class BaseEntity(ABC):
    """
    Base entity class that provides common functionality for all domain entities.
    
    This class provides:
    - Unique identifier generation
    - Creation and update timestamps
    - Common validation methods
    - Dictionary conversion methods
    """
    
    def __init__(self, entity_id: Optional[UUID] = None):
        """
        Initialize the base entity.
        
        Args:
            entity_id: Optional UUID for the entity. If not provided, a new UUID will be generated.
        """
        self._id = entity_id or uuid4()
        self._created_at = datetime.utcnow()
        self._updated_at = datetime.utcnow()
        self._version = 1
    
    @property
    def id(self) -> UUID:
        """Get the entity ID."""
        return self._id
    
    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get the last update timestamp."""
        return self._updated_at
    
    @property
    def version(self) -> int:
        """Get the entity version."""
        return self._version
    
    def touch(self) -> None:
        """Update the last modified timestamp and increment version."""
        self._updated_at = datetime.utcnow()
        self._version += 1
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary representation.
        
        Returns:
            Dictionary representation of the entity
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEntity':
        """
        Create an entity from a dictionary representation.
        
        Args:
            data: Dictionary containing entity data
            
        Returns:
            Entity instance created from the data
        """
        pass
    
    def __eq__(self, other: Any) -> bool:
        """Check if two entities are equal based on their ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Get the hash of the entity based on its ID."""
        return hash(self._id)
    
    def __str__(self) -> str:
        """Get string representation of the entity."""
        return f"{self.__class__.__name__}(id={self._id})"
    
    def __repr__(self) -> str:
        """Get detailed string representation of the entity."""
        return f"{self.__class__.__name__}(id={self._id}, created_at={self._created_at}, updated_at={self._updated_at})"