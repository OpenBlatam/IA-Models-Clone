"""
Load Balancer for Flux2 Clothing Changer
==========================================

Intelligent load balancing and distribution.
"""

import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"


@dataclass
class ServerNode:
    """Server node information."""
    node_id: str
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    last_response_time: float = 0.0
    is_healthy: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.response_times: deque = deque(maxlen=100)


class LoadBalancer:
    """Intelligent load balancer."""
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS,
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.nodes: Dict[str, ServerNode] = {}
        self.current_index = 0  # For round robin
        self.request_history: deque = deque(maxlen=1000)
    
    def add_node(
        self,
        node_id: str,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a server node.
        
        Args:
            node_id: Node identifier
            weight: Node weight (for weighted strategies)
            metadata: Optional metadata
        """
        self.nodes[node_id] = ServerNode(
            node_id=node_id,
            weight=weight,
            metadata=metadata or {},
        )
        logger.info(f"Added node: {node_id} with weight {weight}")
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a server node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if removed
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node: {node_id}")
            return True
        return False
    
    def select_node(self) -> Optional[str]:
        """
        Select a node based on strategy.
        
        Returns:
            Selected node ID or None
        """
        # Filter healthy nodes
        healthy_nodes = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.is_healthy
        }
        
        if not healthy_nodes:
            logger.warning("No healthy nodes available")
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_nodes)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return self._random(healthy_nodes)
        else:
            return self._round_robin(healthy_nodes)
    
    def _round_robin(self, nodes: Dict[str, ServerNode]) -> str:
        """Round robin selection."""
        node_list = list(nodes.keys())
        if not node_list:
            return None
        
        node_id = node_list[self.current_index % len(node_list)]
        self.current_index += 1
        return node_id
    
    def _least_connections(self, nodes: Dict[str, ServerNode]) -> str:
        """Least connections selection."""
        return min(
            nodes.items(),
            key=lambda x: x[1].active_connections
        )[0]
    
    def _weighted_round_robin(self, nodes: Dict[str, ServerNode]) -> str:
        """Weighted round robin selection."""
        # Simple weighted selection
        total_weight = sum(node.weight for node in nodes.values())
        if total_weight == 0:
            return self._round_robin(nodes)
        
        # Select based on weights
        random_value = random.uniform(0, total_weight)
        current = 0.0
        
        for node_id, node in nodes.items():
            current += node.weight
            if random_value <= current:
                return node_id
        
        return list(nodes.keys())[0]
    
    def _least_response_time(self, nodes: Dict[str, ServerNode]) -> str:
        """Least response time selection."""
        return min(
            nodes.items(),
            key=lambda x: x[1].avg_response_time if x[1].avg_response_time > 0 else float('inf')
        )[0]
    
    def _random(self, nodes: Dict[str, ServerNode]) -> str:
        """Random selection."""
        return random.choice(list(nodes.keys()))
    
    def record_request(
        self,
        node_id: str,
        response_time: float,
        success: bool = True,
    ) -> None:
        """
        Record request metrics.
        
        Args:
            node_id: Node that handled request
            response_time: Response time in seconds
            success: Whether request was successful
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.active_connections = max(0, node.active_connections - 1)
        node.total_requests += 1
        node.last_response_time = response_time
        node.response_times.append(response_time)
        
        # Update average
        if node.response_times:
            node.avg_response_time = sum(node.response_times) / len(node.response_times)
        
        if not success:
            node.error_count += 1
        
        # Record in history
        self.request_history.append({
            "node_id": node_id,
            "response_time": response_time,
            "success": success,
            "timestamp": time.time(),
        })
    
    def start_request(self, node_id: str) -> None:
        """Mark request start."""
        if node_id in self.nodes:
            self.nodes[node_id].active_connections += 1
    
    def set_node_health(self, node_id: str, is_healthy: bool) -> None:
        """
        Set node health status.
        
        Args:
            node_id: Node identifier
            is_healthy: Health status
        """
        if node_id in self.nodes:
            self.nodes[node_id].is_healthy = is_healthy
            logger.info(f"Node {node_id} health set to {is_healthy}")
    
    def get_node_statistics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a node."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        error_rate = (
            node.error_count / node.total_requests
            if node.total_requests > 0 else 0.0
        )
        
        return {
            "node_id": node_id,
            "weight": node.weight,
            "active_connections": node.active_connections,
            "total_requests": node.total_requests,
            "avg_response_time": node.avg_response_time,
            "error_count": node.error_count,
            "error_rate": error_rate,
            "is_healthy": node.is_healthy,
            "metadata": node.metadata,
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all nodes."""
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.nodes),
            "healthy_nodes": sum(1 for n in self.nodes.values() if n.is_healthy),
            "nodes": {
                node_id: self.get_node_statistics(node_id)
                for node_id in self.nodes.keys()
            },
            "total_requests": sum(n.total_requests for n in self.nodes.values()),
        }


