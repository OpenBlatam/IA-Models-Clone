"""
Types Module - Common type definitions and aliases.

Provides:
- Type aliases
- Common type definitions
- Union types
- Protocol definitions
"""

from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass
from datetime import datetime

# Type aliases
ConfigDict = Dict[str, Any]
ResultDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
PayloadDict = Dict[str, Any]

# Common return types
MaybeResult = Optional[ResultDict]
MaybeList = Optional[List[Any]]
MaybeDict = Optional[Dict[str, Any]]

# Function types
HandlerFunc = Callable[[Any], Any]
ValidatorFunc = Callable[[Any], bool]
TransformerFunc = Callable[[Any], Any]

# Protocol definitions
class Configurable(Protocol):
    """Protocol for configurable objects."""
    def configure(self, config: ConfigDict) -> None:
        """Configure object."""
        ...
    
    def get_config(self) -> ConfigDict:
        """Get current configuration."""
        ...


class Validatable(Protocol):
    """Protocol for validatable objects."""
    def validate(self) -> bool:
        """Validate object."""
        ...
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        ...


class Serializable(Protocol):
    """Protocol for serializable objects."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create from dictionary."""
        ...


# Common data structures
@dataclass
class Timestamped:
    """Base class for timestamped objects."""
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        """Initialize timestamps."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()
    
    def touch(self) -> None:
        """Update timestamp."""
        self.updated_at = datetime.now().isoformat()


@dataclass
class Identifiable:
    """Base class for identifiable objects."""
    id: str = ""
    
    def generate_id(self, prefix: str = "id") -> str:
        """Generate unique ID."""
        import uuid
        self.id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        return self.id












