"""
Traffic Manager
===============

Advanced traffic management and routing.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrafficPolicy(Enum):
    """Traffic management policies."""
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    WEIGHTED = "weighted"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"


@dataclass
class Route:
    """Traffic route definition."""
    path: str
    backend: str
    weight: int = 100
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class TrafficManager:
    """Advanced traffic manager."""
    
    def __init__(self):
        self._routes: Dict[str, List[Route]] = {}
        self._policies: Dict[str, TrafficPolicy] = {}
        self._traffic_stats: Dict[str, Dict[str, Any]] = {}
    
    def add_route(
        self,
        path: str,
        backend: str,
        weight: int = 100,
        conditions: Optional[Dict[str, Any]] = None
    ):
        """Add traffic route."""
        if path not in self._routes:
            self._routes[path] = []
        
        route = Route(
            path=path,
            backend=backend,
            weight=weight,
            conditions=conditions or {}
        )
        
        self._routes[path].append(route)
        logger.info(f"Added route: {path} -> {backend} (weight: {weight})")
    
    def set_policy(self, path: str, policy: TrafficPolicy):
        """Set traffic policy for path."""
        self._policies[path] = policy
        logger.info(f"Set policy {policy.value} for {path}")
    
    def route_request(
        self,
        path: str,
        client_ip: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Route request to backend."""
        if path not in self._routes:
            return None
        
        routes = self._routes[path]
        policy = self._policies.get(path, TrafficPolicy.WEIGHTED)
        
        # Apply policy
        if policy == TrafficPolicy.WEIGHTED:
            return self._weighted_routing(routes)
        elif policy == TrafficPolicy.CANARY:
            return self._canary_routing(routes, client_ip)
        elif policy == TrafficPolicy.BLUE_GREEN:
            return self._blue_green_routing(routes)
        elif policy == TrafficPolicy.GEOGRAPHIC:
            return self._geographic_routing(routes, client_ip, headers)
        else:
            return routes[0].backend if routes else None
    
    def _weighted_routing(self, routes: List[Route]) -> str:
        """Weighted routing."""
        import random
        total_weight = sum(r.weight for r in routes)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for route in routes:
            cumulative += route.weight
            if r <= cumulative:
                return route.backend
        
        return routes[0].backend
    
    def _canary_routing(self, routes: List[Route], client_ip: Optional[str]) -> str:
        """Canary routing."""
        # Simple canary: use IP hash
        if client_ip:
            hash_value = hash(client_ip)
            canary_percentage = 10  # 10% to canary
            if hash_value % 100 < canary_percentage:
                # Route to canary (last route)
                return routes[-1].backend if len(routes) > 1 else routes[0].backend
        
        # Route to stable (first route)
        return routes[0].backend
    
    def _blue_green_routing(self, routes: List[Route]) -> str:
        """Blue-green routing."""
        # Route to green (last route) or blue (first route)
        # In production, check deployment status
        return routes[-1].backend if len(routes) > 1 else routes[0].backend
    
    def _geographic_routing(
        self,
        routes: List[Route],
        client_ip: Optional[str],
        headers: Optional[Dict[str, str]]
    ) -> str:
        """Geographic routing."""
        # In production, use GeoIP to determine location
        # For now, use default route
        return routes[0].backend
    
    def record_traffic(self, path: str, backend: str, success: bool = True):
        """Record traffic statistics."""
        key = f"{path}:{backend}"
        if key not in self._traffic_stats:
            self._traffic_stats[key] = {
                "requests": 0,
                "successes": 0,
                "failures": 0
            }
        
        self._traffic_stats[key]["requests"] += 1
        if success:
            self._traffic_stats[key]["successes"] += 1
        else:
            self._traffic_stats[key]["failures"] += 1
    
    def get_traffic_stats(self) -> Dict[str, Any]:
        """Get traffic statistics."""
        return {
            "routes": len(self._routes),
            "total_paths": len(self._routes),
            "traffic_stats": self._traffic_stats.copy()
        }















