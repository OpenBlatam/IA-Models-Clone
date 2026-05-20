"""
Load balancing system for KV cache.

This module provides load balancing capabilities for distributing
cache operations across multiple cache instances.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    LEAST_LOADED = "least_loaded"


@dataclass
class CacheNode:
    """A cache node in the load balancer."""
    node_id: str
    cache: Any
    weight: int = 1
    active: bool = True
    connection_count: int = 0
    load_score: float = 0.0
    last_used: float = field(default_factory=time.time)


@dataclass
class LoadBalanceStats:
    """Load balancing statistics."""
    total_requests: int
    requests_per_node: Dict[str, int]
    node_health: Dict[str, bool]
    average_load: float


class CacheLoadBalancer:
    """Load balancer for cache operations."""
    
    def __init__(
        self,
        nodes: List[CacheNode],
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    ):
        self.nodes = {node.node_id: node for node in nodes}
        self.strategy = strategy
        self._current_index = 0
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def get_node(self, key: Optional[str] = None) -> Optional[CacheNode]:
        """Get a node based on load balancing strategy."""
        active_nodes = [n for n in self.nodes.values() if n.active]
        
        if not active_nodes:
            return None
            
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(active_nodes)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return self._random(active_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(active_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(active_nodes)
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._consistent_hash(active_nodes, key)
        elif self.strategy == LoadBalanceStrategy.LEAST_LOADED:
            return self._least_loaded(active_nodes)
        else:
            return active_nodes[0]
            
    def _round_robin(self, nodes: List[CacheNode]) -> CacheNode:
        """Round robin selection."""
        with self._lock:
            node = nodes[self._current_index % len(nodes)]
            self._current_index += 1
            node.connection_count += 1
            node.last_used = time.time()
            self._request_counts[node.node_id] += 1
            return node
            
    def _random(self, nodes: List[CacheNode]) -> CacheNode:
        """Random selection."""
        import random
        node = random.choice(nodes)
        node.connection_count += 1
        node.last_used = time.time()
        self._request_counts[node.node_id] += 1
        return node
        
    def _least_connections(self, nodes: List[CacheNode]) -> CacheNode:
        """Select node with least connections."""
        node = min(nodes, key=lambda n: n.connection_count)
        node.connection_count += 1
        node.last_used = time.time()
        self._request_counts[node.node_id] += 1
        return node
        
    def _weighted_round_robin(self, nodes: List[CacheNode]) -> CacheNode:
        """Weighted round robin selection."""
        total_weight = sum(n.weight for n in nodes)
        if total_weight == 0:
            return self._round_robin(nodes)
            
        # Select based on weight
        import random
        r = random.randint(1, total_weight)
        current = 0
        for node in nodes:
            current += node.weight
            if r <= current:
                node.connection_count += 1
                node.last_used = time.time()
                self._request_counts[node.node_id] += 1
                return node
                
        return nodes[0]
        
    def _consistent_hash(self, nodes: List[CacheNode], key: Optional[str]) -> CacheNode:
        """Consistent hash selection."""
        if not key:
            return self._random(nodes)
            
        import hashlib
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        node_index = key_hash % len(nodes)
        node = nodes[node_index]
        node.connection_count += 1
        node.last_used = time.time()
        self._request_counts[node.node_id] += 1
        return node
        
    def _least_loaded(self, nodes: List[CacheNode]) -> CacheNode:
        """Select least loaded node."""
        node = min(nodes, key=lambda n: n.load_score)
        node.connection_count += 1
        node.last_used = time.time()
        self._request_counts[node.node_id] += 1
        return node
        
    def get(self, key: str) -> Any:
        """Get value from balanced cache."""
        node = self.get_node(key)
        if node:
            return node.cache.get(key)
        return None
        
    def put(self, key: str, value: Any) -> bool:
        """Put value to balanced cache."""
        node = self.get_node(key)
        if node:
            return node.cache.put(key, value)
        return False
        
    def delete(self, key: str) -> bool:
        """Delete value from balanced cache."""
        node = self.get_node(key)
        if node:
            return node.cache.delete(key)
        return False
        
    def get_stats(self) -> LoadBalanceStats:
        """Get load balancing statistics."""
        total = sum(self._request_counts.values())
        node_health = {node_id: node.active for node_id, node in self.nodes.items()}
        
        avg_load = 0.0
        if self.nodes:
            avg_load = sum(n.load_score for n in self.nodes.values()) / len(self.nodes)
            
        return LoadBalanceStats(
            total_requests=total,
            requests_per_node=dict(self._request_counts),
            node_health=node_health,
            average_load=avg_load
        )
        
    def add_node(self, node: CacheNode) -> None:
        """Add a node to the load balancer."""
        self.nodes[node.node_id] = node
        
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._request_counts.pop(node_id, None)
            return True
        return False
        
    def set_node_active(self, node_id: str, active: bool) -> None:
        """Set node active/inactive."""
        if node_id in self.nodes:
            self.nodes[node_id].active = active


