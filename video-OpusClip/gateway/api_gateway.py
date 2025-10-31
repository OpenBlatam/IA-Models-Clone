#!/usr/bin/env python3
"""
API Gateway System

Advanced API gateway with:
- Request routing and load balancing
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- Circuit breaking and fault tolerance
- Monitoring and observability
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
from collections import defaultdict, deque
import httpx
from urllib.parse import urlparse, urljoin

logger = structlog.get_logger("api_gateway")

# =============================================================================
# API GATEWAY MODELS
# =============================================================================

class RouteMethod(Enum):
    """HTTP methods for routing."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    ALL = "ALL"

class RouteStatus(Enum):
    """Route status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AuthenticationType(Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"

@dataclass
class Route:
    """API route definition."""
    route_id: str
    path: str
    methods: List[RouteMethod]
    target_service: str
    target_path: str
    status: RouteStatus
    authentication: AuthenticationType
    rate_limit: Optional[int] = None
    timeout: int = 30
    retry_count: int = 3
    circuit_breaker: bool = True
    transform_request: bool = False
    transform_response: bool = False
    headers: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "route_id": self.route_id,
            "path": self.path,
            "methods": [method.value for method in self.methods],
            "target_service": self.target_service,
            "target_path": self.target_path,
            "status": self.status.value,
            "authentication": self.authentication.value,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "circuit_breaker": self.circuit_breaker,
            "transform_request": self.transform_request,
            "transform_response": self.transform_response,
            "headers": self.headers,
            "metadata": self.metadata
        }

@dataclass
class RouteMatch:
    """Route match result."""
    route: Route
    path_params: Dict[str, str]
    query_params: Dict[str, str]
    match_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "route": self.route.to_dict(),
            "path_params": self.path_params,
            "query_params": self.query_params,
            "match_score": self.match_score
        }

@dataclass
class GatewayRequest:
    """Gateway request data."""
    request_id: str
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[bytes]
    client_ip: str
    user_agent: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "body": self.body.decode('utf-8') if self.body else None,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class GatewayResponse:
    """Gateway response data."""
    status_code: int
    headers: Dict[str, str]
    body: Optional[bytes]
    response_time: float
    route_used: Optional[Route]
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status_code": self.status_code,
            "headers": self.headers,
            "body": self.body.decode('utf-8') if self.body else None,
            "response_time": self.response_time,
            "route_used": self.route_used.to_dict() if self.route_used else None,
            "error_message": self.error_message
        }

# =============================================================================
# API GATEWAY
# =============================================================================

class APIGateway:
    """Advanced API gateway implementation."""
    
    def __init__(self, gateway_name: str = "video-opusclip-gateway"):
        self.gateway_name = gateway_name
        self.routes: Dict[str, Route] = {}
        self.route_index: Dict[str, List[Route]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'failure_count': 0,
            'last_failure': None,
            'state': 'closed',
            'failure_threshold': 5,
            'recovery_timeout': 60
        })
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'circuit_breaker_requests': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0
        }
        
        # Request/response transformers
        self.request_transformers: Dict[str, Callable] = {}
        self.response_transformers: Dict[str, Callable] = {}
        
        # Authentication handlers
        self.auth_handlers: Dict[AuthenticationType, Callable] = {}
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def start(self) -> None:
        """Start the API gateway."""
        logger.info("API Gateway started", gateway_name=self.gateway_name)
    
    async def stop(self) -> None:
        """Stop the API gateway."""
        await self.http_client.aclose()
        logger.info("API Gateway stopped", gateway_name=self.gateway_name)
    
    def add_route(self, route: Route) -> None:
        """Add a route to the gateway."""
        self.routes[route.route_id] = route
        
        # Index by path for faster lookup
        self.route_index[route.path].append(route)
        
        logger.info(
            "Route added",
            route_id=route.route_id,
            path=route.path,
            target_service=route.target_service
        )
    
    def remove_route(self, route_id: str) -> bool:
        """Remove a route from the gateway."""
        if route_id not in self.routes:
            return False
        
        route = self.routes[route_id]
        
        # Remove from index
        if route.path in self.route_index:
            self.route_index[route.path].remove(route)
            if not self.route_index[route.path]:
                del self.route_index[route.path]
        
        # Remove route
        del self.routes[route_id]
        
        logger.info("Route removed", route_id=route_id)
        return True
    
    async def handle_request(self, request: GatewayRequest) -> GatewayResponse:
        """Handle incoming request through the gateway."""
        start_time = time.time()
        
        try:
            # Update statistics
            self.stats['total_requests'] += 1
            
            # Find matching route
            route_match = await self._find_route(request)
            if not route_match:
                return GatewayResponse(
                    status_code=404,
                    headers={'Content-Type': 'application/json'},
                    body=json.dumps({'error': 'Route not found'}).encode(),
                    response_time=time.time() - start_time,
                    route_used=None,
                    error_message="Route not found"
                )
            
            route = route_match.route
            
            # Check route status
            if route.status != RouteStatus.ACTIVE:
                return GatewayResponse(
                    status_code=503,
                    headers={'Content-Type': 'application/json'},
                    body=json.dumps({'error': 'Service unavailable'}).encode(),
                    response_time=time.time() - start_time,
                    route_used=route,
                    error_message="Service unavailable"
                )
            
            # Check rate limiting
            if not await self._check_rate_limit(request, route):
                self.stats['rate_limited_requests'] += 1
                return GatewayResponse(
                    status_code=429,
                    headers={'Content-Type': 'application/json'},
                    body=json.dumps({'error': 'Rate limit exceeded'}).encode(),
                    response_time=time.time() - start_time,
                    route_used=route,
                    error_message="Rate limit exceeded"
                )
            
            # Check circuit breaker
            if not await self._check_circuit_breaker(route):
                self.stats['circuit_breaker_requests'] += 1
                return GatewayResponse(
                    status_code=503,
                    headers={'Content-Type': 'application/json'},
                    body=json.dumps({'error': 'Service temporarily unavailable'}).encode(),
                    response_time=time.time() - start_time,
                    route_used=route,
                    error_message="Circuit breaker open"
                )
            
            # Authenticate request
            auth_result = await self._authenticate_request(request, route)
            if not auth_result['success']:
                return GatewayResponse(
                    status_code=401,
                    headers={'Content-Type': 'application/json'},
                    body=json.dumps({'error': auth_result['message']}).encode(),
                    response_time=time.time() - start_time,
                    route_used=route,
                    error_message=auth_result['message']
                )
            
            # Transform request if needed
            if route.transform_request:
                request = await self._transform_request(request, route)
            
            # Forward request to target service
            response = await self._forward_request(request, route, route_match)
            
            # Transform response if needed
            if route.transform_response:
                response = await self._transform_response(response, route)
            
            # Update statistics
            response_time = time.time() - start_time
            response.response_time = response_time
            self._update_stats(response_time, response.status_code < 400)
            
            return response
            
        except Exception as e:
            logger.error("Gateway request handling error", error=str(e))
            self.stats['failed_requests'] += 1
            
            return GatewayResponse(
                status_code=500,
                headers={'Content-Type': 'application/json'},
                body=json.dumps({'error': 'Internal server error'}).encode(),
                response_time=time.time() - start_time,
                route_used=None,
                error_message=str(e)
            )
    
    async def _find_route(self, request: GatewayRequest) -> Optional[RouteMatch]:
        """Find matching route for request."""
        # Direct path match
        if request.path in self.route_index:
            routes = self.route_index[request.path]
            
            # Find route with matching method
            for route in routes:
                if RouteMethod.ALL in route.methods or RouteMethod(request.method) in route.methods:
                    return RouteMatch(
                        route=route,
                        path_params={},
                        query_params=request.query_params,
                        match_score=1.0
                    )
        
        # Pattern matching (simplified)
        for route in self.routes.values():
            if self._match_path_pattern(request.path, route.path):
                if RouteMethod.ALL in route.methods or RouteMethod(request.method) in route.methods:
                    path_params = self._extract_path_params(request.path, route.path)
                    return RouteMatch(
                        route=route,
                        path_params=path_params,
                        query_params=request.query_params,
                        match_score=0.8
                    )
        
        return None
    
    def _match_path_pattern(self, request_path: str, route_path: str) -> bool:
        """Match request path against route pattern."""
        # Simple pattern matching (can be enhanced with regex)
        if '{' in route_path and '}' in route_path:
            # Convert route pattern to regex
            import re
            pattern = route_path.replace('{', '(?P<').replace('}', '>[^/]+)')
            return bool(re.match(f'^{pattern}$', request_path))
        
        return request_path == route_path
    
    def _extract_path_params(self, request_path: str, route_path: str) -> Dict[str, str]:
        """Extract path parameters from request."""
        # Simple parameter extraction (can be enhanced)
        params = {}
        
        if '{' in route_path and '}' in route_path:
            import re
            pattern = route_path.replace('{', '(?P<').replace('}', '>[^/]+)')
            match = re.match(f'^{pattern}$', request_path)
            if match:
                params = match.groupdict()
        
        return params
    
    async def _check_rate_limit(self, request: GatewayRequest, route: Route) -> bool:
        """Check rate limiting for request."""
        if not route.rate_limit:
            return True
        
        # Simple rate limiting (can be enhanced with Redis)
        key = f"{request.client_ip}:{route.route_id}"
        current_time = time.time()
        
        # Clean old entries
        rate_queue = self.rate_limiter[key]
        while rate_queue and current_time - rate_queue[0] > 60:  # 1 minute window
            rate_queue.popleft()
        
        # Check if limit exceeded
        if len(rate_queue) >= route.rate_limit:
            return False
        
        # Add current request
        rate_queue.append(current_time)
        return True
    
    async def _check_circuit_breaker(self, route: Route) -> bool:
        """Check circuit breaker for route."""
        if not route.circuit_breaker:
            return True
        
        cb_key = f"{route.target_service}:{route.target_path}"
        cb_state = self.circuit_breakers[cb_key]
        
        if cb_state['state'] == 'open':
            # Check if recovery timeout has passed
            if (time.time() - cb_state['last_failure']) > cb_state['recovery_timeout']:
                cb_state['state'] = 'half-open'
                cb_state['failure_count'] = 0
            else:
                return False
        
        return True
    
    async def _authenticate_request(self, request: GatewayRequest, route: Route) -> Dict[str, Any]:
        """Authenticate request based on route configuration."""
        if route.authentication == AuthenticationType.NONE:
            return {'success': True, 'message': 'No authentication required'}
        
        # Get authentication handler
        auth_handler = self.auth_handlers.get(route.authentication)
        if not auth_handler:
            return {'success': False, 'message': 'Authentication handler not configured'}
        
        try:
            result = await auth_handler(request, route)
            return result
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            return {'success': False, 'message': 'Authentication failed'}
    
    async def _transform_request(self, request: GatewayRequest, route: Route) -> GatewayRequest:
        """Transform request if transformer is configured."""
        transformer = self.request_transformers.get(route.route_id)
        if transformer:
            return await transformer(request, route)
        return request
    
    async def _transform_response(self, response: GatewayResponse, route: Route) -> GatewayResponse:
        """Transform response if transformer is configured."""
        transformer = self.response_transformers.get(route.route_id)
        if transformer:
            return await transformer(response, route)
        return response
    
    async def _forward_request(self, request: GatewayRequest, route: Route, 
                              route_match: RouteMatch) -> GatewayResponse:
        """Forward request to target service."""
        # Build target URL
        target_url = f"http://{route.target_service}{route.target_path}"
        
        # Replace path parameters
        for param, value in route_match.path_params.items():
            target_url = target_url.replace(f"{{{param}}}", value)
        
        # Add query parameters
        if route_match.query_params:
            query_string = "&".join([f"{k}={v}" for k, v in route_match.query_params.items()])
            target_url += f"?{query_string}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.update(route.headers)
        
        # Forward request
        try:
            async with self.http_client.stream(
                method=request.method,
                url=target_url,
                headers=headers,
                content=request.body,
                timeout=route.timeout
            ) as response:
                
                # Read response
                response_body = await response.aread()
                
                return GatewayResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response_body,
                    response_time=0.0,  # Will be set by caller
                    route_used=route,
                    error_message=None
                )
        
        except httpx.TimeoutException:
            await self._record_circuit_breaker_failure(route)
            raise Exception("Request timeout")
        
        except Exception as e:
            await self._record_circuit_breaker_failure(route)
            raise e
    
    async def _record_circuit_breaker_failure(self, route: Route) -> None:
        """Record circuit breaker failure."""
        if not route.circuit_breaker:
            return
        
        cb_key = f"{route.target_service}:{route.target_path}"
        cb_state = self.circuit_breakers[cb_key]
        
        cb_state['failure_count'] += 1
        cb_state['last_failure'] = time.time()
        
        if cb_state['failure_count'] >= cb_state['failure_threshold']:
            cb_state['state'] = 'open'
            logger.warning(
                "Circuit breaker opened",
                service=route.target_service,
                path=route.target_path,
                failure_count=cb_state['failure_count']
            )
    
    def _update_stats(self, response_time: float, success: bool) -> None:
        """Update gateway statistics."""
        self.stats['total_response_time'] += response_time
        
        if success:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1
        
        # Update average response time
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        if total_requests > 0:
            self.stats['average_response_time'] = self.stats['total_response_time'] / total_requests
    
    def add_request_transformer(self, route_id: str, transformer: Callable) -> None:
        """Add request transformer for route."""
        self.request_transformers[route_id] = transformer
    
    def add_response_transformer(self, route_id: str, transformer: Callable) -> None:
        """Add response transformer for route."""
        self.response_transformers[route_id] = transformer
    
    def add_auth_handler(self, auth_type: AuthenticationType, handler: Callable) -> None:
        """Add authentication handler."""
        self.auth_handlers[auth_type] = handler
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            **self.stats,
            'total_routes': len(self.routes),
            'active_routes': len([r for r in self.routes.values() if r.status == RouteStatus.ACTIVE]),
            'circuit_breakers': {
                key: {
                    'state': state['state'],
                    'failure_count': state['failure_count'],
                    'last_failure': state['last_failure']
                }
                for key, state in self.circuit_breakers.items()
            }
        }
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """Get all routes."""
        return [route.to_dict() for route in self.routes.values()]

# =============================================================================
# GLOBAL API GATEWAY INSTANCE
# =============================================================================

# Global API gateway
api_gateway = APIGateway()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'RouteMethod',
    'RouteStatus',
    'AuthenticationType',
    'Route',
    'RouteMatch',
    'GatewayRequest',
    'GatewayResponse',
    'APIGateway',
    'api_gateway'
]





























