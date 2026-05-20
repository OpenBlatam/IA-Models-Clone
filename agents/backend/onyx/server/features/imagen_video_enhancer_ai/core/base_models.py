"""
Base Models
===========

Base classes for common data models.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict, field
from abc import ABC


@dataclass
class BaseModel:
    """Base class for data models with common functionality."""
    
    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        
        Args:
            exclude_none: Whether to exclude None values
            
        Returns:
            Dictionary representation
        """
        data = asdict(self)
        
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        
        # Convert enums to values
        for key, value in data.items():
            if hasattr(value, 'value'):
                data[key] = value.value
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create model from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Model instance
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement from_dict")
    
    def update(self, **kwargs):
        """
        Update model fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get field value.
        
        Args:
            key: Field name
            default: Default value if field doesn't exist
            
        Returns:
            Field value or default
        """
        return getattr(self, key, default)


@dataclass
class TimestampedModel(BaseModel):
    """Base model with timestamps."""
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def touch(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


@dataclass
class IdentifiedModel(BaseModel):
    """Base model with ID."""
    
    id: str
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a new ID."""
        import uuid
        return str(uuid.uuid4())


@dataclass
class StatusModel(BaseModel, ABC):
    """Base model with status."""
    
    status: Any  # Can be string or enum
    
    def is_pending(self) -> bool:
        """Check if status is pending."""
        status_value = self.status.value if hasattr(self.status, 'value') else self.status
        return status_value == "pending"
    
    def is_completed(self) -> bool:
        """Check if status is completed."""
        status_value = self.status.value if hasattr(self.status, 'value') else self.status
        return status_value == "completed"
    
    def is_failed(self) -> bool:
        """Check if status is failed."""
        status_value = self.status.value if hasattr(self.status, 'value') else self.status
        return status_value == "failed"
    
    def is_processing(self) -> bool:
        """Check if status is processing."""
        status_value = self.status.value if hasattr(self.status, 'value') else self.status
        return status_value == "processing"

