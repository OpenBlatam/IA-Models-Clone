"""
Load Balancer

Utilities for load balancing and request distribution.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Base load balancer."""
    
    def __init__(self, servers: List[str]):
        """
        Initialize load balancer.
        
        Args:
            servers: List of server addresses
        """
        self.servers = servers
        self.connections: Dict[str, int] = defaultdict(int)
        self.health_status: Dict[str, bool] = {server: True for server in servers}
    
    def select_server(self) -> Optional[str]:
        """
        Select server for request.
        
        Returns:
            Server address or None
        """
        raise NotImplementedError
    
    def mark_healthy(self, server: str) -> None:
        """Mark server as healthy."""
        self.health_status[server] = True
    
    def mark_unhealthy(self, server: str) -> None:
        """Mark server as unhealthy."""
        self.health_status[server] = False
    
    def get_healthy_servers(self) -> List[str]:
        """Get list of healthy servers."""
        return [s for s in self.servers if self.health_status.get(s, True)]
    
    def increment_connections(self, server: str) -> None:
        """Increment connection count."""
        self.connections[server] += 1
    
    def decrement_connections(self, server: str) -> None:
        """Decrement connection count."""
        self.connections[server] = max(0, self.connections[server] - 1)


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""
    
    def __init__(self, servers: List[str]):
        """Initialize round-robin balancer."""
        super().__init__(servers)
        self.current_index = 0
    
    def select_server(self) -> Optional[str]:
        """Select server using round-robin."""
        healthy = self.get_healthy_servers()
        
        if not healthy:
            return None
        
        server = healthy[self.current_index % len(healthy)]
        self.current_index += 1
        
        return server


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancer."""
    
    def select_server(self) -> Optional[str]:
        """Select server with least connections."""
        healthy = self.get_healthy_servers()
        
        if not healthy:
            return None
        
        # Find server with minimum connections
        min_connections = min(self.connections[s] for s in healthy)
        candidates = [s for s in healthy if self.connections[s] == min_connections]
        
        # If tie, use round-robin
        return random.choice(candidates) if candidates else None


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round-robin load balancer."""
    
    def __init__(
        self,
        servers: List[str],
        weights: Optional[Dict[str, int]] = None
    ):
        """
        Initialize weighted round-robin balancer.
        
        Args:
            servers: List of servers
            weights: Server weights (default: equal weights)
        """
        super().__init__(servers)
        self.weights = weights or {server: 1 for server in servers}
        self.current_weights: Dict[str, int] = {server: 0 for server in servers}
    
    def select_server(self) -> Optional[str]:
        """Select server using weighted round-robin."""
        healthy = self.get_healthy_servers()
        
        if not healthy:
            return None
        
        # Add weight to current weights
        for server in healthy:
            self.current_weights[server] += self.weights.get(server, 1)
        
        # Select server with highest current weight
        selected = max(healthy, key=lambda s: self.current_weights[s])
        
        # Subtract total weight from selected server
        total_weight = sum(self.weights.get(s, 1) for s in healthy)
        self.current_weights[selected] -= total_weight
        
        return selected



