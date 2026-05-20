"""
API Gateway Patterns
===================

Advanced API gateway patterns for routing, load balancing, and request handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Routing strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"

@dataclass
class Route:
    """Route definition."""
    path: str
    target: str
    method: str = "GET"
    middleware: List[Callable] = None
    timeout: float = 30.0
    retries: int = 3
    circuit_breaker: bool = True

class APIGateway:
    """Advanced API gateway."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self.middleware: List[Callable] = []
        self.routing_strategy = RoutingStrategy.ROUND_ROBIN
        self.connection_counts: Dict[str, int] = {}
    
    def register_route(
        self,
        path: str,
        target: str,
        method: str = "GET",
        middleware: Optional[List[Callable]] = None,
        timeout: float = 30.0,
        retries: int = 3,
        circuit_breaker: bool = True
    ) -> Route:
        """Register a route."""
        route = Route(
            path=path,
            target=target,
            method=method,
            middleware=middleware or [],
            timeout=timeout,
            retries=retries,
            circuit_breaker=circuit_breaker
        )
        
        route_key = f"{method}:{path}"
        self.routes[route_key] = route
        logger.info(f"Route registered: {method} {path} -> {target}")
        
        return route
    
    def add_middleware(self, middleware: Callable):
        """Add global middleware."""
        self.middleware.append(middleware)
        logger.info(f"Middleware added: {middleware.__name__}")
    
    def find_route(self, path: str, method: str = "GET") -> Optional[Route]:
        """Find route for path and method."""
        route_key = f"{method}:{path}"
        
        # Exact match
        if route_key in self.routes:
            return self.routes[route_key]
        
        # Pattern matching (simple implementation)
        for key, route in self.routes.items():
            route_method, route_path = key.split(":", 1)
            
            if route_method == method:
                # Simple wildcard matching
                if self._match_path(route_path, path):
                    return route
        
        return None
    
    def _match_path(self, pattern: str, path: str) -> bool:
        """Simple path pattern matching."""
        # Convert pattern to regex-like matching
        if pattern == path:
            return True
        
        # Simple wildcard: * matches anything
        if "*" in pattern:
            import re
            regex_pattern = pattern.replace("*", ".*")
            return bool(re.match(regex_pattern, path))
        
        return False
    
    def select_target(
        self,
        targets: List[str],
        client_ip: Optional[str] = None
    ) -> Optional[str]:
        """Select target using routing strategy."""
        if not targets:
            return None
        
        if len(targets) == 1:
            return targets[0]
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round robin (would need state for real round robin)
            return targets[0]
        
        elif self.routing_strategy == RoutingStrategy.RANDOM:
            import random
            return random.choice(targets)
        
        elif self.routing_strategy == RoutingStrategy.LEAST_CONNECTIONS:
            # Select target with least connections
            min_connections = min(
                self.connection_counts.get(t, 0) for t in targets
            )
            for target in targets:
                if self.connection_counts.get(target, 0) == min_connections:
                    return target
            return targets[0]
        
        elif self.routing_strategy == RoutingStrategy.IP_HASH:
            # Hash client IP to select target
            if client_ip:
                hash_value = hash(client_ip) % len(targets)
                return targets[hash_value]
            return targets[0]
        
        else:
            return targets[0]
    
    def increment_connections(self, target: str):
        """Increment connection count for target."""
        self.connection_counts[target] = self.connection_counts.get(target, 0) + 1
    
    def decrement_connections(self, target: str):
        """Decrement connection count for target."""
        if target in self.connection_counts:
            self.connection_counts[target] = max(0, self.connection_counts[target] - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API gateway statistics."""
        return {
            "total_routes": len(self.routes),
            "total_middleware": len(self.middleware),
            "routing_strategy": self.routing_strategy.value,
            "connection_counts": self.connection_counts.copy(),
            "routes": {
                key: {
                    "path": route.path,
                    "target": route.target,
                    "method": route.method,
                    "timeout": route.timeout,
                    "retries": route.retries
                }
                for key, route in self.routes.items()
            }
        }

# Global instance
api_gateway = APIGateway()

















