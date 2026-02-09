from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Generic
from uuid import UUID, uuid4
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Entity Class

Contains common functionality for all domain entities.
"""


logger = structlog.get_logger()

# Type for entity IDs
EntityID = TypeVar('EntityID')


class BaseEntity(ABC, Generic[EntityID]):
    """
    Base class for all domain entities.
    
    Provides common functionality like:
    - Unique identity
    - Audit fields (created_at, updated_at)
    - Domain events
    - Equality comparison
    """
    
    def __init__(
        self,
        id: Optional[EntityID] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        
    """__init__ function."""
self._id = id or self._generate_id()
        self._created_at = created_at or self._utc_now()
        self._updated_at = updated_at or self._utc_now()
        self._domain_events: List[Any] = []
        self._is_deleted = False
    
    @property
    def id(self) -> EntityID:
        """Get entity ID."""
        return self._id
    
    @property
    def created_at(self) -> datetime:
        """Get creation timestamp."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Get last update timestamp."""
        return self._updated_at
    
    @property
    def is_deleted(self) -> bool:
        """Check if entity is soft deleted."""
        return self._is_deleted
    
    @property
    def domain_events(self) -> List[Any]:
        """Get pending domain events."""
        return self._domain_events.copy()
    
    def mark_updated(self) -> None:
        """Mark entity as updated."""
        self._updated_at = self._utc_now()
    
    def mark_deleted(self) -> None:
        """Mark entity as soft deleted."""
        self._is_deleted = True
        self.mark_updated()
    
    def add_domain_event(self, event: Any) -> None:
        """Add a domain event."""
        self._domain_events.append(event)
        logger.debug("Domain event added", entity_id=str(self._id), event_type=type(event).__name__)
    
    def clear_domain_events(self) -> None:
        """Clear all domain events."""
        self._domain_events.clear()
    
    def __eq__(self, other: object) -> bool:
        """Compare entities by ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Hash by entity ID."""
        return hash(self._id)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self._id})"
    
    @abstractmethod
    def _generate_id(self) -> EntityID:
        """Generate a new entity ID."""
        pass
    
    @staticmethod
    def _utc_now() -> datetime:
        """Get current UTC datetime."""
        return datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": str(self._id),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "is_deleted": self._is_deleted
        }


class EntityID(ABC):
    """
    Base class for strongly-typed entity IDs.
    """
    
    def __init__(self, value: UUID):
        
    """__init__ function."""
if not isinstance(value, UUID):
            raise ValueError("Entity ID must be a UUID")
        self._value = value
    
    @property
    def value(self) -> UUID:
        """Get UUID value."""
        return self._value
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._value)
    
    def __eq__(self, other: object) -> bool:
        """Compare by UUID value."""
        if not isinstance(other, EntityID):
            return False
        return self._value == other._value
    
    def __hash__(self) -> int:
        """Hash by UUID value."""
        return hash(self._value)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self._value})"
    
    @classmethod
    def generate(cls) -> 'EntityID':
        """Generate a new entity ID."""
        return cls(uuid4())
    
    @classmethod
    def from_string(cls, value: str) -> 'EntityID':
        """Create entity ID from string."""
        return cls(UUID(value)) 