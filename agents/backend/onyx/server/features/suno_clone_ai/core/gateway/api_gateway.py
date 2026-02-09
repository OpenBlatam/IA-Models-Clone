"""
API Gateway

Utilities for API Gateway functionality.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class Route:
    """API route definition."""
    
    def __init__(
        self,
        path: str,
        handler: Callable,
        methods: List[str] = None,
        middleware: List[Callable] = None
    ):
        """
        Initialize route.
        
        Args:
            path: Route path
            handler: Handler function
            methods: HTTP methods
            middleware: Route middleware
        """
        self.path = path
        self.handler = handler
        self.methods = methods or ["GET", "POST"]
        self.middleware = middleware or []


class APIGateway:
    """API Gateway for request routing."""
    
    def __init__(self):
        """Initialize API gateway."""
        self.routes: Dict[str, List[Route]] = defaultdict(list)
        self.global_middleware: List[Callable] = []
    
    def add_route(
        self,
        path: str,
        handler: Callable,
        methods: List[str] = None,
        middleware: List[Callable] = None
    ) -> None:
        """
        Add route to gateway.
        
        Args:
            path: Route path
            handler: Handler function
            methods: HTTP methods
            middleware: Route middleware
        """
        route = Route(path, handler, methods, middleware)
        
        for method in route.methods:
            self.routes[method].append(route)
        
        logger.info(f"Added route: {method} {path}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add global middleware.
        
        Args:
            middleware: Middleware function
        """
        self.global_middleware.append(middleware)
    
    def route(
        self,
        method: str,
        path: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route request to handler.
        
        Args:
            method: HTTP method
            path: Request path
            request: Request dictionary
            
        Returns:
            Response dictionary
        """
        # Find matching route
        route = self._find_route(method, path)
        
        if not route:
            return {
                'status_code': 404,
                'error': 'Route not found'
            }
        
        # Apply global middleware
        for middleware in self.global_middleware:
            request = middleware(request)
        
        # Apply route middleware
        for middleware in route.middleware:
            request = middleware(request)
        
        # Call handler
        try:
            response = route.handler(request)
            return response
        except Exception as e:
            logger.error(f"Route handler error: {e}")
            return {
                'status_code': 500,
                'error': str(e)
            }
    
    def _find_route(
        self,
        method: str,
        path: str
    ) -> Optional[Route]:
        """
        Find matching route.
        
        Args:
            method: HTTP method
            path: Request path
            
        Returns:
            Route or None
        """
        for route in self.routes.get(method, []):
            if self._match_path(route.path, path):
                return route
        
        return None
    
    def _match_path(self, route_path: str, request_path: str) -> bool:
        """
        Match route path with request path.
        
        Args:
            route_path: Route path pattern
            request_path: Request path
            
        Returns:
            True if matches
        """
        # Simple exact match (can be extended for patterns)
        return route_path == request_path


def create_gateway() -> APIGateway:
    """Create API gateway."""
    return APIGateway()


def route_request(
    gateway: APIGateway,
    method: str,
    path: str,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Route request via gateway."""
    return gateway.route(method, path, request)


def add_route(
    gateway: APIGateway,
    path: str,
    handler: Callable,
    **kwargs
) -> None:
    """Add route to gateway."""
    gateway.add_route(path, handler, **kwargs)



