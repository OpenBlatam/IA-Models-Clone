"""
Federated Connectors Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import uuid4


class FederatedNode:
    """Federated node definition"""
    
    def __init__(
        self,
        node_id: str,
        endpoint: str,
        capabilities: List[str],
        auth_config: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.auth_config = auth_config or {}
        self.connected = False
        self.last_sync: Optional[datetime] = None


class FederatedConnection:
    """Federated connection"""
    
    def __init__(self, from_node: str, to_node: str):
        self.id = str(uuid4())
        self.from_node = from_node
        self.to_node = to_node
        self.created_at = datetime.utcnow()
        self.active = True


class FederatedConnectorBase(ABC):
    """Base interface for federated connectors"""
    
    @abstractmethod
    async def connect_node(self, node: FederatedNode) -> bool:
        """Connect to federated node"""
        pass
    
    @abstractmethod
    async def sync_data(
        self,
        node_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Sync data with federated node"""
        pass
    
    @abstractmethod
    async def resolve_conflict(
        self,
        conflict_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve data conflict"""
        pass

