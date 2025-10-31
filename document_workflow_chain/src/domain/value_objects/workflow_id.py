"""
Workflow ID Value Object
=======================

Immutable value object representing a workflow identifier.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True)
class WorkflowId:
    """
    Workflow ID Value Object
    
    Immutable value object that represents a workflow identifier.
    Provides type safety and validation.
    """
    
    value: UUID
    
    def __post_init__(self) -> None:
        """Validate the workflow ID"""
        if not isinstance(self.value, UUID):
            raise ValueError("WorkflowId must be a UUID")
    
    @classmethod
    def generate(cls) -> WorkflowId:
        """Generate a new workflow ID"""
        return cls(uuid4())
    
    @classmethod
    def from_string(cls, value: str) -> WorkflowId:
        """Create WorkflowId from string"""
        try:
            uuid_value = UUID(value)
            return cls(uuid_value)
        except ValueError as e:
            raise ValueError(f"Invalid UUID string: {value}") from e
    
    def to_string(self) -> str:
        """Convert to string"""
        return str(self.value)
    
    def __str__(self) -> str:
        """String representation"""
        return str(self.value)
    
    def __repr__(self) -> str:
        """Representation"""
        return f"WorkflowId({self.value})"
    
    def __eq__(self, other: Any) -> bool:
        """Check equality"""
        if not isinstance(other, WorkflowId):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Get hash"""
        return hash(self.value)




