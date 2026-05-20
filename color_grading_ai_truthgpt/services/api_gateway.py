"""
API Gateway for Color Grading AI
=================================

Unified API gateway for routing, authentication, and request management.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class RouteMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class Route:
    """API route definition."""
    path: str
    method: RouteMethod
    handler: Callable
    middleware: List[Callable] = field(default_factory=list)
    auth_required: bool = False
    rate_limit: Optional[int] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Request context."""
    path: str
    method: RouteMethod
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Any = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """API response."""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class APIGateway:
    """
    API Gateway.
    
    Features:
    - Route management
    - Request routing
    - Middleware support
    - Authentication
    - Rate limiting
    - Request/response transformation
    - Error handling
    """
    
    def __init__(self):
        """Initialize API gateway."""
        self._routes: List[Route] = []
        self._middleware: List[Callable] = []
        self._auth_handler: Optional[Callable] = None
        self._rate_limiter: Optional[Any] = None
        self._lock = asyncio.Lock()
    
    def register_route(
        self,
        path: str,
        method: RouteMethod,
        handler: Callable,
        middleware: Optional[List[Callable]] = None,
        auth_required: bool = False,
        rate_limit: Optional[int] = None,
        timeout: Optional[float] = None
    ):
        """
        Register API route.
        
        Args:
            path: Route path (supports patterns like /api/v1/{id})
            method: HTTP method
            handler: Handler function
            middleware: Optional middleware list
            auth_required: Whether authentication is required
            rate_limit: Optional rate limit
            timeout: Optional timeout
        """
        route = Route(
            path=path,
            method=method,
            handler=handler,
            middleware=middleware or [],
            auth_required=auth_required,
            rate_limit=rate_limit,
            timeout=timeout
        )
        
        self._routes.append(route)
        logger.info(f"Registered route: {method.value} {path}")
    
    def register_middleware(self, middleware: Callable):
        """
        Register global middleware.
        
        Args:
            middleware: Middleware function
        """
        self._middleware.append(middleware)
        logger.info("Registered global middleware")
    
    def set_auth_handler(self, handler: Callable):
        """
        Set authentication handler.
        
        Args:
            handler: Authentication handler function
        """
        self._auth_handler = handler
        logger.info("Set authentication handler")
    
    def set_rate_limiter(self, rate_limiter: Any):
        """
        Set rate limiter.
        
        Args:
            rate_limiter: Rate limiter instance
        """
        self._rate_limiter = rate_limiter
        logger.info("Set rate limiter")
    
    async def handle_request(
        self,
        path: str,
        method: RouteMethod,
        context: RequestContext
    ) -> Response:
        """
        Handle API request.
        
        Args:
            path: Request path
            method: HTTP method
            context: Request context
            
        Returns:
            API response
        """
        # Find matching route
        route = self._find_route(path, method)
        if not route:
            return Response(
                status_code=404,
                data={"error": "Route not found"}
            )
        
        try:
            # Apply global middleware
            for middleware in self._middleware:
                context = await self._apply_middleware(middleware, context)
            
            # Apply route middleware
            for middleware in route.middleware:
                context = await self._apply_middleware(middleware, context)
            
            # Check authentication
            if route.auth_required:
                if not self._auth_handler:
                    return Response(
                        status_code=401,
                        data={"error": "Authentication required but no handler configured"}
                    )
                
                auth_result = await self._auth_handler(context)
                if not auth_result:
                    return Response(
                        status_code=401,
                        data={"error": "Authentication failed"}
                    )
            
            # Check rate limiting
            if route.rate_limit and self._rate_limiter:
                if not await self._rate_limiter.allow(context.user_id or "anonymous"):
                    return Response(
                        status_code=429,
                        data={"error": "Rate limit exceeded"}
                    )
            
            # Execute handler
            if route.timeout:
                handler_result = await asyncio.wait_for(
                    self._execute_handler(route.handler, context),
                    timeout=route.timeout
                )
            else:
                handler_result = await self._execute_handler(route.handler, context)
            
            return Response(
                status_code=200,
                data=handler_result
            )
        
        except asyncio.TimeoutError:
            return Response(
                status_code=504,
                data={"error": "Request timeout"}
            )
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return Response(
                status_code=500,
                data={"error": str(e)}
            )
    
    def _find_route(self, path: str, method: RouteMethod) -> Optional[Route]:
        """Find matching route."""
        for route in self._routes:
            if route.method != method:
                continue
            
            # Simple pattern matching (supports {param})
            pattern = route.path.replace("{", "(?P<").replace("}", ">[^/]+)")
            match = re.match(f"^{pattern}$", path)
            if match:
                return route
        
        return None
    
    async def _apply_middleware(
        self,
        middleware: Callable,
        context: RequestContext
    ) -> RequestContext:
        """Apply middleware."""
        if asyncio.iscoroutinefunction(middleware):
            result = await middleware(context)
        else:
            result = middleware(context)
        
        if isinstance(result, RequestContext):
            return result
        return context
    
    async def _execute_handler(
        self,
        handler: Callable,
        context: RequestContext
    ) -> Any:
        """Execute route handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(context)
        else:
            return handler(context)
    
    def get_routes(self) -> List[Route]:
        """Get all registered routes."""
        return self._routes.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            "total_routes": len(self._routes),
            "middleware_count": len(self._middleware),
            "auth_enabled": self._auth_handler is not None,
            "rate_limiting_enabled": self._rate_limiter is not None,
        }


