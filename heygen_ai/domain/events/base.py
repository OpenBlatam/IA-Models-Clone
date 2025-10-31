from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Base Domain Event

Base class for all domain events in the system.
"""



@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for all domain events.
    
    Domain events represent significant business occurrences that other
    parts of the system might be interested in.
    """
    
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_version: int = field(default=1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.__class__.__name__,
            "occurred_at": self.occurred_at.isoformat(),
            "event_version": self.event_version,
            "data": self._event_data()
        }
    
    def _event_data(self) -> Dict[str, Any]:
        """Get event-specific data. Override in subclasses."""
        # Get all fields except the base ones
        base_fields = {"event_id", "occurred_at", "event_version"}
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__.keys()
            if field_name not in base_fields
        } 