from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Entity
==========

Base class for all domain entities with common functionality.
"""




class Entity(BaseModel, ABC):
    """
    Base entity class with common functionality.
    
    All domain entities inherit from this class and provide:
    - Unique identifier
    - Creation and modification timestamps
    - Domain event tracking
    - Validation and business rules
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique entity identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    version: int = Field(default=1, description="Entity version for optimistic locking")
    
    # Domain events
    _domain_events: list = Field(default_factory=list, exclude=True)
    _is_dirty: bool = Field(default=False, exclude=True)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def __init__(self, **data: Any) -> None:
        """Initialize entity with validation."""
        super().__init__(**data)
        self._validate_entity()
    
    @abstractmethod
    def _validate_entity(self) -> None:
        """Validate entity business rules."""
        pass
    
    def add_domain_event(self, event: Any) -> None:
        """Add a domain event to the entity."""
        self._domain_events.append(event)
        self._is_dirty = True
    
    def clear_domain_events(self) -> list:
        """Clear and return domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def mark_as_dirty(self) -> None:
        """Mark entity as modified."""
        self._is_dirty = True
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def is_dirty(self) -> bool:
        """Check if entity has been modified."""
        return self._is_dirty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return self.model_dump(exclude={"_domain_events", "_is_dirty"})
    
    def __eq__(self, other: Any) -> bool:
        """Compare entities by ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash entity by ID."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.id})"


class AggregateRoot(Entity):
    """
    Aggregate root entity.
    
    Aggregate roots are the entry points to aggregates and ensure
    consistency boundaries within the domain.
    """
    
    def __init__(self, **data: Any) -> None:
        """Initialize aggregate root."""
        super().__init__(**data)
    
    def _validate_entity(self) -> None:
        """Validate aggregate root business rules."""
        # Base validation - can be overridden by subclasses
        pass
    
    def ensure_consistency(self) -> None:
        """Ensure aggregate consistency."""
        self._validate_entity()
    
    def get_aggregate_id(self) -> UUID:
        """Get aggregate identifier."""
        return self.id 