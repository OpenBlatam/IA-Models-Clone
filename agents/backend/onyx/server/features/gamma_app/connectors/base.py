"""
Connectors Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4


class ConnectorType(str, Enum):
    """Connector types"""
    API = "api"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"
    AUTH = "auth"


class Connector:
    """Connector definition"""
    
    def __init__(
        self,
        name: str,
        connector_type: ConnectorType,
        config: Dict[str, Any],
        enabled: bool = True
    ):
        self.id = str(uuid4())
        self.name = name
        self.connector_type = connector_type
        self.config = config
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.last_used: Optional[datetime] = None


class ConnectorBase(ABC):
    """Base interface for connectors"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to external service"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from external service"""
        pass
    
    @abstractmethod
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """Execute operation"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check connector health"""
        pass

