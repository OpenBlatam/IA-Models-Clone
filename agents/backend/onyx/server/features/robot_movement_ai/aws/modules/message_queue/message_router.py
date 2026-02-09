"""
Message Router
==============

Message routing and distribution.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies."""
    DIRECT = "direct"
    TOPIC = "topic"
    FANOUT = "fanout"
    ROUTING_KEY = "routing_key"


@dataclass
class Route:
    """Message route definition."""
    pattern: str
    target_queue: str
    strategy: RoutingStrategy
    handler: Optional[Callable] = None


class MessageRouter:
    """Message router."""
    
    def __init__(self):
        self._routes: List[Route] = []
        self._routing_table: Dict[str, List[str]] = {}  # pattern -> queues
    
    def add_route(
        self,
        pattern: str,
        target_queue: str,
        strategy: RoutingStrategy = RoutingStrategy.DIRECT,
        handler: Optional[Callable] = None
    ):
        """Add routing rule."""
        route = Route(
            pattern=pattern,
            target_queue=target_queue,
            strategy=strategy,
            handler=handler
        )
        
        self._routes.append(route)
        
        if pattern not in self._routing_table:
            self._routing_table[pattern] = []
        self._routing_table[pattern].append(target_queue)
        
        logger.info(f"Added route: {pattern} -> {target_queue}")
    
    def route_message(self, message: Any, routing_key: Optional[str] = None) -> List[str]:
        """Route message to target queues."""
        target_queues = []
        
        for route in self._routes:
            if self._matches_pattern(route.pattern, routing_key or ""):
                target_queues.append(route.target_queue)
                
                # Apply handler if exists
                if route.handler:
                    try:
                        route.handler(message)
                    except Exception as e:
                        logger.error(f"Route handler failed: {e}")
        
        return target_queues
    
    def _matches_pattern(self, pattern: str, routing_key: str) -> bool:
        """Check if routing key matches pattern."""
        if pattern == "*":
            return True
        
        if pattern == routing_key:
            return True
        
        # Simple wildcard matching
        if "*" in pattern:
            pattern_parts = pattern.split("*")
            if len(pattern_parts) == 2:
                return routing_key.startswith(pattern_parts[0]) and routing_key.endswith(pattern_parts[1])
        
        return False
    
    def get_routes(self) -> List[Route]:
        """Get all routes."""
        return self._routes.copy()
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_routes": len(self._routes),
            "routing_table": self._routing_table.copy()
        }















