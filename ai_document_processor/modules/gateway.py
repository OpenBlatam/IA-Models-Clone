"""
API Gateway - Ultra-Modular Gateway System
=========================================

Ultra-modular API gateway for routing, load balancing, and service orchestration.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import httpx
from urllib.parse import urlparse, urljoin

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RouteMethod(str, Enum):
    """HTTP method enumeration."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class MiddlewareType(str, Enum):
    """Middleware type enumeration."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMITING = "rate_limiting"
    LOGGING = "logging"
    CACHING = "caching"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    CUSTOM = "custom"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    id: str
    name: str
    url: str
    weight: int = 1
    health_check_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_connections: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    response_time_avg: float = 0.0
    request_count: int = 0


@dataclass
class Route:
    """API route configuration."""
    id: str
    path: str
    methods: List[RouteMethod]
    service_endpoints: List[ServiceEndpoint]
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    timeout: int = 30
    retry_attempts: int = 3
    middleware: List[str] = field(default_factory=list)
    authentication_required: bool = False
    rate_limit: Optional[int] = None
    rate_limit_window: int = 60
    cache_ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Middleware(ABC):
    """Base middleware interface."""
    
    def __init__(self, name: str, middleware_type: MiddlewareType):
        self.name = name
        self.middleware_type = middleware_type
        self.enabled = True
        self.priority = 0
        self.error_count = 0
        self.last_error = None
        self.processed_count = 0
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request."""
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        return {
            'name': self.name,
            'type': self.middleware_type.value,
            'enabled': self.enabled,
            'priority': self.priority,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'processed_count': self.processed_count
        }


class AuthenticationMiddleware(Middleware):
    """Authentication middleware."""
    
    def __init__(self, name: str = "authentication"):
        super().__init__(name, MiddlewareType.AUTHENTICATION)
        self.required_headers = ["Authorization"]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process authentication."""
        try:
            headers = request.get('headers', {})
            
            # Check for required headers
            for header in self.required_headers:
                if header not in headers:
                    return {
                        'status': 'error',
                        'error': 'authentication_required',
                        'message': f'Missing required header: {header}'
                    }
            
            # Validate token (simplified)
            auth_header = headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return {
                    'status': 'error',
                    'error': 'invalid_token',
                    'message': 'Invalid authentication token format'
                }
            
            # Add user info to request
            request['user'] = {
                'id': 'user123',  # In production, decode from token
                'roles': ['user']
            }
            
            self.processed_count += 1
            return request
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return {
                'status': 'error',
                'error': 'authentication_failed',
                'message': str(e)
            }
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process response (no changes needed)."""
        return response


class RateLimitingMiddleware(Middleware):
    """Rate limiting middleware."""
    
    def __init__(self, name: str = "rate_limiting", requests_per_minute: int = 100):
        super().__init__(name, MiddlewareType.RATE_LIMITING)
        self.requests_per_minute = requests_per_minute
        self._request_counts: Dict[str, List[datetime]] = {}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process rate limiting."""
        try:
            # Get client IP
            client_ip = request.get('client_ip', 'unknown')
            now = datetime.utcnow()
            
            # Clean old requests
            if client_ip in self._request_counts:
                self._request_counts[client_ip] = [
                    req_time for req_time in self._request_counts[client_ip]
                    if (now - req_time).total_seconds() < 60
                ]
            else:
                self._request_counts[client_ip] = []
            
            # Check rate limit
            if len(self._request_counts[client_ip]) >= self.requests_per_minute:
                return {
                    'status': 'error',
                    'error': 'rate_limit_exceeded',
                    'message': f'Rate limit exceeded: {self.requests_per_minute} requests per minute'
                }
            
            # Record request
            self._request_counts[client_ip].append(now)
            self.processed_count += 1
            
            return request
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return request  # Allow request on error
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process response (no changes needed)."""
        return response


class LoggingMiddleware(Middleware):
    """Logging middleware."""
    
    def __init__(self, name: str = "logging"):
        super().__init__(name, MiddlewareType.LOGGING)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming request."""
        try:
            logger.info(f"Request: {request.get('method', 'UNKNOWN')} {request.get('path', '/')}")
            self.processed_count += 1
            return request
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return request
    
    async def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Log outgoing response."""
        try:
            logger.info(f"Response: {response.get('status_code', 200)}")
            return response
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return response


class APIGateway:
    """Ultra-modular API gateway."""
    
    def __init__(self, name: str = "api_gateway"):
        self.name = name
        self._routes: Dict[str, Route] = {}
        self._middleware: Dict[str, Middleware] = {}
        self._service_endpoints: Dict[str, ServiceEndpoint] = {}
        self._load_balancers: Dict[str, int] = {}  # Round-robin counters
        self._stats = {
            'requests_processed': 0,
            'requests_failed': 0,
            'routes_configured': 0,
            'middleware_registered': 0,
            'start_time': datetime.utcnow()
        }
        self._lock = asyncio.Lock()
        self._http_client = httpx.AsyncClient(timeout=30.0)
    
    async def register_route(self, route: Route) -> bool:
        """Register a route."""
        async with self._lock:
            try:
                # Validate route
                if not route.path or not route.methods or not route.service_endpoints:
                    return False
                
                # Store route
                self._routes[route.id] = route
                
                # Store service endpoints
                for endpoint in route.service_endpoints:
                    self._service_endpoints[endpoint.id] = endpoint
                
                # Initialize load balancer
                self._load_balancers[route.id] = 0
                
                self._stats['routes_configured'] = len(self._routes)
                
                logger.info(f"Registered route: {route.path} ({route.id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register route: {e}")
                return False
    
    async def unregister_route(self, route_id: str) -> bool:
        """Unregister a route."""
        async with self._lock:
            if route_id not in self._routes:
                return False
            
            route = self._routes[route_id]
            
            # Remove service endpoints
            for endpoint in route.service_endpoints:
                if endpoint.id in self._service_endpoints:
                    del self._service_endpoints[endpoint.id]
            
            # Remove route
            del self._routes[route_id]
            del self._load_balancers[route_id]
            
            self._stats['routes_configured'] = len(self._routes)
            
            logger.info(f"Unregistered route: {route_id}")
            return True
    
    async def register_middleware(self, middleware: Middleware) -> bool:
        """Register middleware."""
        async with self._lock:
            try:
                self._middleware[middleware.name] = middleware
                self._stats['middleware_registered'] = len(self._middleware)
                
                logger.info(f"Registered middleware: {middleware.name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register middleware: {e}")
                return False
    
    async def unregister_middleware(self, middleware_name: str) -> bool:
        """Unregister middleware."""
        async with self._lock:
            if middleware_name not in self._middleware:
                return False
            
            del self._middleware[middleware_name]
            self._stats['middleware_registered'] = len(self._middleware)
            
            logger.info(f"Unregistered middleware: {middleware_name}")
            return True
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming request."""
        try:
            # Find matching route
            route = await self._find_route(request)
            if not route:
                return {
                    'status_code': 404,
                    'body': {'error': 'route_not_found', 'message': 'No route found for request'}
                }
            
            # Apply middleware
            processed_request = await self._apply_middleware(route, request, 'request')
            if processed_request.get('status') == 'error':
                return {
                    'status_code': 400,
                    'body': processed_request
                }
            
            # Select service endpoint
            endpoint = await self._select_endpoint(route)
            if not endpoint:
                return {
                    'status_code': 503,
                    'body': {'error': 'service_unavailable', 'message': 'No healthy endpoints available'}
                }
            
            # Forward request
            response = await self._forward_request(endpoint, processed_request)
            
            # Apply response middleware
            processed_response = await self._apply_middleware(route, response, 'response')
            
            # Update stats
            self._stats['requests_processed'] += 1
            
            return processed_response
            
        except Exception as e:
            self._stats['requests_failed'] += 1
            logger.error(f"Request processing failed: {e}")
            return {
                'status_code': 500,
                'body': {'error': 'internal_error', 'message': str(e)}
            }
    
    async def _find_route(self, request: Dict[str, Any]) -> Optional[Route]:
        """Find matching route for request."""
        path = request.get('path', '/')
        method = request.get('method', 'GET')
        
        for route in self._routes.values():
            if self._path_matches(route.path, path) and RouteMethod(method) in route.methods:
                return route
        
        return None
    
    def _path_matches(self, route_path: str, request_path: str) -> bool:
        """Check if request path matches route path."""
        # Simple path matching (in production, use proper routing library)
        if route_path == request_path:
            return True
        
        # Handle wildcards
        if route_path.endswith('*'):
            prefix = route_path[:-1]
            return request_path.startswith(prefix)
        
        return False
    
    async def _apply_middleware(self, route: Route, data: Dict[str, Any], direction: str) -> Dict[str, Any]:
        """Apply middleware to request or response."""
        try:
            # Get middleware in priority order
            middleware_names = sorted(route.middleware, key=lambda name: self._middleware.get(name, {}).get('priority', 0))
            
            for middleware_name in middleware_names:
                middleware = self._middleware.get(middleware_name)
                if not middleware or not middleware.enabled:
                    continue
                
                if direction == 'request':
                    data = await middleware.process_request(data)
                else:
                    data = await middleware.process_response(data)
                
                # Check for errors
                if data.get('status') == 'error':
                    break
            
            return data
            
        except Exception as e:
            logger.error(f"Middleware processing failed: {e}")
            return data
    
    async def _select_endpoint(self, route: Route) -> Optional[ServiceEndpoint]:
        """Select service endpoint using load balancing strategy."""
        healthy_endpoints = [ep for ep in route.service_endpoints if ep.is_healthy]
        
        if not healthy_endpoints:
            return None
        
        if route.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round-robin selection
            counter = self._load_balancers.get(route.id, 0)
            selected_endpoint = healthy_endpoints[counter % len(healthy_endpoints)]
            self._load_balancers[route.id] = counter + 1
            return selected_endpoint
        
        elif route.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select endpoint with least connections (simplified)
            return min(healthy_endpoints, key=lambda ep: ep.request_count)
        
        elif route.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            # Random selection
            import random
            return random.choice(healthy_endpoints)
        
        else:
            # Default to first healthy endpoint
            return healthy_endpoints[0]
    
    async def _forward_request(self, endpoint: ServiceEndpoint, request: Dict[str, Any]) -> Dict[str, Any]:
        """Forward request to service endpoint."""
        try:
            start_time = datetime.utcnow()
            
            # Prepare request
            url = urljoin(endpoint.url, request.get('path', '/'))
            method = request.get('method', 'GET')
            headers = request.get('headers', {})
            body = request.get('body')
            
            # Make request
            response = await self._http_client.request(
                method=method,
                url=url,
                headers=headers,
                content=body
            )
            
            # Update endpoint stats
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            endpoint.request_count += 1
            endpoint.response_time_avg = (
                (endpoint.response_time_avg * (endpoint.request_count - 1) + response_time) / 
                endpoint.request_count
            )
            
            return {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'body': response.text
            }
            
        except Exception as e:
            endpoint.error_count += 1
            endpoint.is_healthy = False
            logger.error(f"Request forwarding failed: {e}")
            raise
    
    async def health_check_endpoint(self, endpoint_id: str) -> bool:
        """Perform health check on endpoint."""
        if endpoint_id not in self._service_endpoints:
            return False
        
        endpoint = self._service_endpoints[endpoint_id]
        
        try:
            health_url = endpoint.health_check_url or urljoin(endpoint.url, '/health')
            
            response = await self._http_client.get(health_url, timeout=5.0)
            
            is_healthy = response.status_code == 200
            endpoint.is_healthy = is_healthy
            endpoint.last_health_check = datetime.utcnow()
            
            if not is_healthy:
                endpoint.error_count += 1
            
            return is_healthy
            
        except Exception as e:
            endpoint.is_healthy = False
            endpoint.error_count += 1
            logger.error(f"Health check failed for endpoint {endpoint_id}: {e}")
            return False
    
    async def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            **self._stats,
            'routes_configured': len(self._routes),
            'middleware_registered': len(self._middleware),
            'service_endpoints': len(self._service_endpoints),
            'healthy_endpoints': len([ep for ep in self._service_endpoints.values() if ep.is_healthy]),
            'uptime_seconds': (datetime.utcnow() - self._stats['start_time']).total_seconds()
        }
    
    async def get_route_stats(self) -> Dict[str, Any]:
        """Get route statistics."""
        route_stats = {}
        for route_id, route in self._routes.items():
            route_stats[route_id] = {
                'path': route.path,
                'methods': [m.value for m in route.methods],
                'endpoints': len(route.service_endpoints),
                'healthy_endpoints': len([ep for ep in route.service_endpoints if ep.is_healthy]),
                'load_balancing_strategy': route.load_balancing_strategy.value,
                'middleware': route.middleware
            }
        
        return route_stats
    
    async def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        middleware_stats = {}
        for name, middleware in self._middleware.items():
            middleware_stats[name] = middleware.get_stats()
        
        return middleware_stats
    
    async def close(self):
        """Close the gateway and cleanup resources."""
        await self._http_client.aclose()
        logger.info("API Gateway closed")


# Global API gateway instance
_api_gateway: Optional[APIGateway] = None


def get_api_gateway() -> APIGateway:
    """Get global API gateway instance."""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = APIGateway()
    return _api_gateway

















