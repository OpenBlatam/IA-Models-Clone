"""
Shared Value Objects
Value objects used across modules
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict
from uuid import UUID, uuid4


@dataclass(frozen=True)
class EntityId:
    """Universal entity ID"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("Entity ID cannot be empty")
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def generate(cls) -> 'EntityId':
        """Generate new entity ID"""
        return cls(str(uuid4()))


@dataclass(frozen=True)
class Timestamp:
    """Timestamp value object"""
    value: datetime
    
    def __post_init__(self):
        if not isinstance(self.value, datetime):
            raise ValueError("Timestamp must be a datetime object")
    
    def iso_string(self) -> str:
        """Get ISO string representation"""
        return self.value.isoformat()
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """Create timestamp for now"""
        return cls(datetime.utcnow())


@dataclass(frozen=True)
class Metadata:
    """Metadata value object"""
    data: Dict[str, Any]
    
    def __post_init__(self):
        if not isinstance(self.data, dict):
            raise ValueError("Metadata must be a dictionary")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.data.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.data
    
    def merge(self, other: Dict[str, Any]) -> 'Metadata':
        """Merge with other metadata"""
        merged = {**self.data, **other}
        return Metadata(merged)






