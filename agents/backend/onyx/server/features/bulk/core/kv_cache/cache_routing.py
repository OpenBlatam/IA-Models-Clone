"""
Cache routing and load balancing.

Provides routing capabilities for distributed cache.
"""
from __future__ import annotations

import logging
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    CONSISTENT_HASH = "consistent_hash"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


@dataclass
class CacheNode:
    """Cache node information."""
    id: str
    cache: Any
    weight: float = 1.0
    connections: int = 0
    health: bool = True


class CacheRouter:
    """
    Cache router.
    
    Routes requests to appropriate cache nodes.
    """
    
    def __init__(
        self,
        nodes: List[CacheNode],
        strategy: RoutingStrategy = RoutingStrategy.CONSISTENT_HASH
    ):
        """
        Initialize router.
        
        Args:
            nodes: List of cache nodes
            strategy: Routing strategy
        """
        self.nodes = nodes
        self.strategy = strategy
        self.current_index = 0
        self.ring: Dict[int, CacheNode] = {}
        
        if strategy == RoutingStrategy.CONSISTENT_HASH:
            self._build_ring()
    
    def _build_ring(self) -> None:
        """Build consistent hash ring."""
        for node in self.nodes:
            for i in range(int(node.weight * 100)):
                node_key = f"{node.id}_{i}"
                hash_value = int(hashlib.md5(node_key.encode()).hexdigest(), 16)
                self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def route(self, key: Any) -> Optional[CacheNode]:
        """
        Route key to node.
        
        Args:
            key: Cache key
            
        Returns:
            Cache node or None
        """
        if not self.nodes:
            return None
        
        # Filter healthy nodes
        healthy_nodes = [n for n in self.nodes if n.health]
        if not healthy_nodes:
            return None
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            node = healthy_nodes[self.current_index % len(healthy_nodes)]
            self.current_index += 1
            return node
        
        elif self.strategy == RoutingStrategy.CONSISTENT_HASH:
            key_str = str(key)
            key_hash = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
            
            for node_hash in self.sorted_keys:
                if node_hash >= key_hash:
                    node = self.ring[node_hash]
                    if node.health:
                        return node
            
            # Wrap around
            node = self.ring[self.sorted_keys[0]]
            return node if node.health else None
        
        elif self.strategy == RoutingStrategy.WEIGHTED:
            total_weight = sum(n.weight for n in healthy_nodes)
            import random
            r = random.uniform(0, total_weight)
            
            current = 0
            for node in healthy_nodes:
                current += node.weight
                if r <= current:
                    return node
        
        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(healthy_nodes, key=lambda n: n.connections)
        
        elif self.strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(healthy_nodes)
        
        return healthy_nodes[0]
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from routed node.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        node = self.route(key)
        if node:
            node.connections += 1
            try:
                return node.cache.get(key)
            finally:
                node.connections -= 1
        return None
    
    def put(self, key: Any, value: Any) -> bool:
        """
        Put value to routed node.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        node = self.route(key)
        if node:
            node.connections += 1
            try:
                node.cache.put(key, value)
                return True
            finally:
                node.connections -= 1
        return False
    
    def add_node(self, node: CacheNode) -> None:
        """
        Add node to router.
        
        Args:
            node: Cache node
        """
        self.nodes.append(node)
        if self.strategy == RoutingStrategy.CONSISTENT_HASH:
            self._build_ring()
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node from router.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if removed
        """
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                del self.nodes[i]
                if self.strategy == RoutingStrategy.CONSISTENT_HASH:
                    self._build_ring()
                return True
        return False
    
    def set_node_health(self, node_id: str, health: bool) -> None:
        """
        Set node health status.
        
        Args:
            node_id: Node ID
            health: Health status
        """
        for node in self.nodes:
            if node.id == node_id:
                node.health = health
                break


class CacheLoadBalancer:
    """
    Cache load balancer.
    
    Balances load across cache nodes.
    """
    
    def __init__(self, router: CacheRouter):
        """
        Initialize load balancer.
        
        Args:
            router: Cache router
        """
        self.router = router
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get load balancing metrics.
        
        Returns:
            Metrics dictionary
        """
        return {
            "nodes": len(self.router.nodes),
            "healthy_nodes": sum(1 for n in self.router.nodes if n.health),
            "total_connections": sum(n.connections for n in self.router.nodes),
            "node_metrics": [
                {
                    "id": node.id,
                    "connections": node.connections,
                    "weight": node.weight,
                    "health": node.health
                }
                for node in self.router.nodes
            ]
        }
    
    def rebalance(self) -> None:
        """Rebalance load across nodes."""
        # Simple rebalancing: adjust weights based on connections
        total_connections = sum(n.connections for n in self.router.nodes)
        
        if total_connections > 0:
            for node in self.router.nodes:
                if node.health:
                    # Adjust weight based on current load
                    load_ratio = node.connections / total_connections
                    # Inverse relationship: more connections = lower weight
                    node.weight = max(0.1, 1.0 - load_ratio)
        
        # Rebuild ring if using consistent hash
        if self.router.strategy == RoutingStrategy.CONSISTENT_HASH:
            self.router._build_ring()

