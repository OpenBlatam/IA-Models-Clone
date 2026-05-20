"""
Load Balancer
=============

Advanced load balancing with multiple algorithms and health checking.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class BackendServer:
    """Backend server definition."""
    server_id: str
    host: str
    port: int
    weight: int = 1
    is_healthy: bool = True
    active_connections: int = 0
    response_time: float = 0.0
    last_check: Optional[datetime] = None
    metadata: Dict[str, Any] = None

class LoadBalancer:
    """Advanced load balancer."""
    
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN):
        self.algorithm = algorithm
        self.backends: Dict[str, BackendServer] = {}
        self.round_robin_index: Dict[str, int] = {}
        self.consistent_hash_ring: Dict[int, str] = {}
        self.is_running = False
    
    def add_backend(
        self,
        server_id: str,
        host: str,
        port: int,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BackendServer:
        """Add a backend server."""
        backend = BackendServer(
            server_id=server_id,
            host=host,
            port=port,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.backends[server_id] = backend
        self.round_robin_index[server_id] = 0
        
        # Update consistent hash ring
        if self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            self._update_consistent_hash_ring()
        
        logger.info(f"Backend added: {server_id} ({host}:{port})")
        
        return backend
    
    def remove_backend(self, server_id: str):
        """Remove a backend server."""
        if server_id in self.backends:
            del self.backends[server_id]
            if server_id in self.round_robin_index:
                del self.round_robin_index[server_id]
            
            if self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                self._update_consistent_hash_ring()
            
            logger.info(f"Backend removed: {server_id}")
    
    def get_backend(self, pool_name: str = "default", client_ip: Optional[str] = None) -> Optional[BackendServer]:
        """Get a backend server using load balancing algorithm."""
        healthy_backends = [
            b for b in self.backends.values()
            if b.is_healthy
        ]
        
        if not healthy_backends:
            logger.warning("No healthy backends available")
            return None
        
        if len(healthy_backends) == 1:
            return healthy_backends[0]
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin(healthy_backends, pool_name)
        
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(healthy_backends)
        
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_backends, pool_name)
        
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)
        
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return min(healthy_backends, key=lambda b: b.response_time)
        
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            if client_ip:
                hash_value = hash(client_ip) % len(healthy_backends)
                return healthy_backends[hash_value]
            return healthy_backends[0]
        
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            if client_ip:
                return self._consistent_hash(client_ip)
            return healthy_backends[0]
        
        return healthy_backends[0]
    
    def _round_robin(self, backends: List[BackendServer], pool_name: str) -> BackendServer:
        """Round robin selection."""
        if pool_name not in self.round_robin_index:
            self.round_robin_index[pool_name] = 0
        
        index = self.round_robin_index[pool_name]
        backend = backends[index % len(backends)]
        self.round_robin_index[pool_name] = (index + 1) % len(backends)
        
        return backend
    
    def _weighted_round_robin(self, backends: List[BackendServer], pool_name: str) -> BackendServer:
        """Weighted round robin selection."""
        total_weight = sum(b.weight for b in backends)
        
        if pool_name not in self.round_robin_index:
            self.round_robin_index[pool_name] = 0
        
        current_weight = self.round_robin_index[pool_name]
        
        for backend in backends:
            if current_weight < backend.weight:
                self.round_robin_index[pool_name] = (current_weight + 1) % total_weight
                return backend
            current_weight -= backend.weight
        
        # Fallback
        backend = backends[0]
        self.round_robin_index[pool_name] = (self.round_robin_index[pool_name] + 1) % total_weight
        return backend
    
    def _consistent_hash(self, key: str) -> Optional[BackendServer]:
        """Consistent hash selection."""
        if not self.consistent_hash_ring:
            return None
        
        hash_value = hash(key)
        sorted_keys = sorted(self.consistent_hash_ring.keys())
        
        for ring_key in sorted_keys:
            if hash_value <= ring_key:
                server_id = self.consistent_hash_ring[ring_key]
                return self.backends.get(server_id)
        
        # Wrap around
        if sorted_keys:
            server_id = self.consistent_hash_ring[sorted_keys[0]]
            return self.backends.get(server_id)
        
        return None
    
    def _update_consistent_hash_ring(self):
        """Update consistent hash ring."""
        self.consistent_hash_ring = {}
        for server_id, backend in self.backends.items():
            if backend.is_healthy:
                # Create multiple virtual nodes
                for i in range(backend.weight * 100):
                    virtual_key = hash(f"{server_id}:{i}")
                    self.consistent_hash_ring[virtual_key] = server_id
    
    def increment_connections(self, server_id: str):
        """Increment connection count."""
        if server_id in self.backends:
            self.backends[server_id].active_connections += 1
    
    def decrement_connections(self, server_id: str):
        """Decrement connection count."""
        if server_id in self.backends:
            self.backends[server_id].active_connections = max(
                0,
                self.backends[server_id].active_connections - 1
            )
    
    def update_response_time(self, server_id: str, response_time: float):
        """Update response time for a server."""
        if server_id in self.backends:
            self.backends[server_id].response_time = response_time
            self.backends[server_id].last_check = datetime.now()
    
    def set_health(self, server_id: str, is_healthy: bool):
        """Set health status for a server."""
        if server_id in self.backends:
            self.backends[server_id].is_healthy = is_healthy
            if self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                self._update_consistent_hash_ring()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_connections = sum(b.active_connections for b in self.backends.values())
        healthy_count = sum(1 for b in self.backends.values() if b.is_healthy)
        
        return {
            "algorithm": self.algorithm.value,
            "total_backends": len(self.backends),
            "healthy_backends": healthy_count,
            "total_connections": total_connections,
            "backends": {
                server_id: {
                    "host": b.host,
                    "port": b.port,
                    "weight": b.weight,
                    "is_healthy": b.is_healthy,
                    "active_connections": b.active_connections,
                    "response_time": b.response_time,
                    "last_check": b.last_check.isoformat() if b.last_check else None
                }
                for server_id, b in self.backends.items()
            }
        }

# Global instance
load_balancer = LoadBalancer()

















