"""
Event Data Structure Module

Event data structure for event system.
"""

from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """
    Event data structure.
    
    Args:
        name: Event name.
        data: Event data.
        timestamp: Event timestamp.
        source: Optional event source.
    """
    name: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None



