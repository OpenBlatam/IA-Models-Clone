"""
Messaging Port
==============

Port for messaging operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable


class MessagingPort(ABC):
    """Port for messaging operations."""
    
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        """Publish message to topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to topic."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from topic."""
        pass















