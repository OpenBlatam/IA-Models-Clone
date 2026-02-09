"""
Router and Dispatcher Utilities for Piel Mejorador AI SAM3
=========================================================

Unified router and dispatcher pattern utilities.
"""

import asyncio
import logging
from typing import TypeVar, Callable, Any, Dict, Optional, List, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class Route:
    """Route definition."""
    pattern: str
    handler: Callable[[Any], Any]
    name: Optional[str] = None
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class Router:
    """Router for dispatching to handlers based on patterns."""
    
    def __init__(self, name: str = "router"):
        """
        Initialize router.
        
        Args:
            name: Router name
        """
        self.name = name
        self._routes: List[Route] = []
        self._lock = asyncio.Lock()
    
    def register(
        self,
        pattern: str,
        handler: Callable[[Any], Any],
        name: Optional[str] = None,
        priority: int = 0,
        enabled: bool = True,
        **metadata
    ) -> Route:
        """
        Register route.
        
        Args:
            pattern: Route pattern
            handler: Handler function
            name: Optional route name
            priority: Route priority (higher = checked first)
            enabled: Whether route is enabled
            **metadata: Additional metadata
            
        Returns:
            Route object
        """
        route = Route(
            pattern=pattern,
            handler=handler,
            name=name,
            priority=priority,
            enabled=enabled,
            metadata=metadata
        )
        
        self._routes.append(route)
        # Sort by priority (higher first)
        self._routes.sort(key=lambda r: r.priority, reverse=True)
        
        logger.debug(f"Registered route {pattern} in {self.name}")
        return route
    
    def unregister(self, pattern: str) -> bool:
        """
        Unregister route.
        
        Args:
            pattern: Route pattern
            
        Returns:
            True if unregistered
        """
        original_size = len(self._routes)
        self._routes = [r for r in self._routes if r.pattern != pattern]
        removed = len(self._routes) < original_size
        
        if removed:
            logger.debug(f"Unregistered route {pattern} from {self.name}")
        
        return removed
    
    def _match_pattern(self, pattern: str, value: str) -> bool:
        """
        Match pattern against value.
        
        Args:
            pattern: Pattern to match
            value: Value to match against
            
        Returns:
            True if matches
        """
        # Simple exact match
        if pattern == value:
            return True
        
        # Wildcard match
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return value.startswith(prefix)
        
        if pattern.startswith("*"):
            suffix = pattern[1:]
            return value.endswith(suffix)
        
        # Contains match
        if "*" in pattern:
            parts = pattern.split("*")
            if len(parts) == 2:
                return value.startswith(parts[0]) and value.endswith(parts[1])
        
        return False
    
    async def route(
        self,
        value: str,
        context: Any = None
    ) -> Optional[R]:
        """
        Route value to handler.
        
        Args:
            value: Value to route
            context: Optional context
            
        Returns:
            Handler result or None
        """
        async with self._lock:
            for route in self._routes:
                if not route.enabled:
                    continue
                
                if self._match_pattern(route.pattern, value):
                    logger.debug(f"Routing {value} to {route.pattern}")
                    try:
                        if asyncio.iscoroutinefunction(route.handler):
                            return await route.handler(value, context)
                        else:
                            return route.handler(value, context)
                    except Exception as e:
                        logger.error(f"Error in route handler {route.pattern}: {e}")
                        raise
        
        logger.warning(f"No route found for {value}")
        return None
    
    def list_routes(self) -> List[Route]:
        """
        List all routes.
        
        Returns:
            List of routes
        """
        return self._routes.copy()
    
    def get_route(self, pattern: str) -> Optional[Route]:
        """
        Get route by pattern.
        
        Args:
            pattern: Route pattern
            
        Returns:
            Route or None
        """
        for route in self._routes:
            if route.pattern == pattern:
                return route
        return None
    
    def enable_route(self, pattern: str):
        """Enable route."""
        route = self.get_route(pattern)
        if route:
            route.enabled = True
    
    def disable_route(self, pattern: str):
        """Disable route."""
        route = self.get_route(pattern)
        if route:
            route.enabled = False


class Dispatcher:
    """Dispatcher for routing messages to handlers."""
    
    def __init__(self, name: str = "dispatcher"):
        """
        Initialize dispatcher.
        
        Args:
            name: Dispatcher name
        """
        self.name = name
        self._handlers: Dict[str, List[Callable[[Any], Any]]] = {}
        self._lock = asyncio.Lock()
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Any], Any],
        priority: int = 0
    ):
        """
        Subscribe handler to event type.
        
        Args:
            event_type: Event type
            handler: Handler function
            priority: Handler priority
        """
        async with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            # Store with priority
            self._handlers[event_type].append((priority, handler))
            # Sort by priority (higher first)
            self._handlers[event_type].sort(key=lambda x: x[0], reverse=True)
            
            logger.debug(f"Subscribed handler to {event_type} in {self.name}")
    
    async def dispatch(
        self,
        event_type: str,
        payload: Any
    ) -> List[Any]:
        """
        Dispatch event to handlers.
        
        Args:
            event_type: Event type
            payload: Event payload
            
        Returns:
            List of handler results
        """
        async with self._lock:
            handlers = self._handlers.get(event_type, [])
        
        results = []
        for priority, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(payload)
                else:
                    result = handler(payload)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in dispatcher handler for {event_type}: {e}")
                raise
        
        return results
    
    def unsubscribe(self, event_type: str, handler: Callable[[Any], Any]):
        """
        Unsubscribe handler.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        async with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type] = [
                    (p, h) for p, h in self._handlers[event_type]
                    if h != handler
                ]


class RouterUtils:
    """Unified router utilities."""
    
    @staticmethod
    def create_router(name: str = "router") -> Router:
        """
        Create router.
        
        Args:
            name: Router name
            
        Returns:
            Router
        """
        return Router(name)
    
    @staticmethod
    def create_dispatcher(name: str = "dispatcher") -> Dispatcher:
        """
        Create dispatcher.
        
        Args:
            name: Dispatcher name
            
        Returns:
            Dispatcher
        """
        return Dispatcher(name)


# Convenience functions
def create_router(name: str = "router") -> Router:
    """Create router."""
    return RouterUtils.create_router(name)


def create_dispatcher(name: str = "dispatcher") -> Dispatcher:
    """Create dispatcher."""
    return RouterUtils.create_dispatcher(name)




