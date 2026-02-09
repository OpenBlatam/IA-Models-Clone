"""
Intelligent Load Balancer
=========================

AI-powered intelligent load balancing.
"""

import logging
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_LOAD = "least_load"


@dataclass
class BackendServer:
    """Backend server definition."""
    id: str
    url: str
    weight: int = 1
    health: bool = True
    connections: int = 0
    response_time: float = 0.0
    load: float = 0.0


class IntelligentLoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOAD):
        self.strategy = strategy
        self._backends: Dict[str, BackendServer] = {}
        self._round_robin_index = 0
        self._stats: Dict[str, Dict[str, Any]] = {}
    
    def add_backend(self, server: BackendServer):
        """Add backend server."""
        self._backends[server.id] = server
        self._stats[server.id] = {
            "requests": 0,
            "successes": 0,
            "failures": 0
        }
        logger.info(f"Added backend: {server.id} ({server.url})")
    
    def remove_backend(self, server_id: str):
        """Remove backend server."""
        if server_id in self._backends:
            del self._backends[server_id]
            logger.info(f"Removed backend: {server_id}")
    
    def get_backend(self, client_ip: Optional[str] = None) -> Optional[BackendServer]:
        """Get backend server based on strategy."""
        healthy_backends = [
            server for server in self._backends.values()
            if server.health
        ]
        
        if not healthy_backends:
            logger.warning("No healthy backends available")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda s: s.connections)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(healthy_backends, key=lambda s: s.response_time)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_backends)
        
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(healthy_backends, client_ip)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_LOAD:
            return min(healthy_backends, key=lambda s: s.load)
        
        return healthy_backends[0]
    
    def _round_robin(self, backends: List[BackendServer]) -> BackendServer:
        """Round robin selection."""
        backend = backends[self._round_robin_index % len(backends)]
        self._round_robin_index += 1
        return backend
    
    def _weighted_round_robin(self, backends: List[BackendServer]) -> BackendServer:
        """Weighted round robin selection."""
        total_weight = sum(s.weight for s in backends)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for server in backends:
            cumulative += server.weight
            if r <= cumulative:
                return server
        
        return backends[0]
    
    def _ip_hash(self, backends: List[BackendServer], client_ip: Optional[str]) -> BackendServer:
        """IP hash selection."""
        if not client_ip:
            return backends[0]
        
        hash_value = hash(client_ip)
        index = hash_value % len(backends)
        return backends[index]
    
    def update_backend_health(self, server_id: str, health: bool):
        """Update backend health status."""
        if server_id in self._backends:
            self._backends[server_id].health = health
            logger.info(f"Backend {server_id} health: {health}")
    
    def update_backend_metrics(
        self,
        server_id: str,
        connections: Optional[int] = None,
        response_time: Optional[float] = None,
        load: Optional[float] = None
    ):
        """Update backend metrics."""
        if server_id in self._backends:
            server = self._backends[server_id]
            if connections is not None:
                server.connections = connections
            if response_time is not None:
                server.response_time = response_time
            if load is not None:
                server.load = load
    
    def record_request(self, server_id: str, success: bool = True):
        """Record request statistics."""
        if server_id in self._stats:
            self._stats[server_id]["requests"] += 1
            if success:
                self._stats[server_id]["successes"] += 1
            else:
                self._stats[server_id]["failures"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            "strategy": self.strategy.value,
            "backends": len(self._backends),
            "healthy_backends": sum(1 for s in self._backends.values() if s.health),
            "backend_stats": self._stats.copy()
        }















