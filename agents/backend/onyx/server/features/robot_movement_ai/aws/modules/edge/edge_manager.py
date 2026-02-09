"""
Edge Manager
============

Edge computing management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EdgeNode:
    """Edge node definition."""
    id: str
    name: str
    location: str
    endpoint: str
    status: str = "active"  # active, offline, maintenance
    capabilities: List[str] = None
    last_sync: Optional[datetime] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class EdgeManager:
    """Edge computing manager."""
    
    def __init__(self):
        self._nodes: Dict[str, EdgeNode] = {}
        self._deployments: Dict[str, List[str]] = {}  # service -> node_ids
    
    def register_node(
        self,
        node_id: str,
        name: str,
        location: str,
        endpoint: str,
        capabilities: Optional[List[str]] = None
    ) -> EdgeNode:
        """Register edge node."""
        node = EdgeNode(
            id=node_id,
            name=name,
            location=location,
            endpoint=endpoint,
            capabilities=capabilities or []
        )
        
        self._nodes[node_id] = node
        logger.info(f"Registered edge node: {node_id} at {location}")
        return node
    
    def deploy_to_edge(self, service_name: str, node_id: str) -> bool:
        """Deploy service to edge node."""
        if node_id not in self._nodes:
            return False
        
        if service_name not in self._deployments:
            self._deployments[service_name] = []
        
        if node_id not in self._deployments[service_name]:
            self._deployments[service_name].append(node_id)
            logger.info(f"Deployed {service_name} to edge node {node_id}")
        
        return True
    
    def get_nodes_by_location(self, location: str) -> List[EdgeNode]:
        """Get nodes by location."""
        return [node for node in self._nodes.values() if node.location == location]
    
    def get_nodes_by_capability(self, capability: str) -> List[EdgeNode]:
        """Get nodes by capability."""
        return [
            node for node in self._nodes.values()
            if capability in node.capabilities
        ]
    
    def update_node_sync(self, node_id: str):
        """Update node sync timestamp."""
        if node_id in self._nodes:
            self._nodes[node_id].last_sync = datetime.now()
    
    def get_edge_stats(self) -> Dict[str, Any]:
        """Get edge computing statistics."""
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": sum(1 for n in self._nodes.values() if n.status == "active"),
            "by_location": {
                location: sum(1 for n in self._nodes.values() if n.location == location)
                for location in set(n.location for n in self._nodes.values())
            },
            "deployments": len(self._deployments)
        }















