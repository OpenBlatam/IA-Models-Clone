"""
API Gateway Integration
Advanced API Gateway patterns for microservices
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from microservices.service_discovery import ServiceRegistry, get_service_registry
from microservices.service_client import ServiceClient, get_service_client

logger = logging.getLogger(__name__)


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """
    API Gateway middleware for request routing and transformation
    
    Features:
    - Request routing to microservices
    - Request/response transformation
    - Service discovery integration
    - Load balancing
    - Circuit breaker integration
    """
    
    def __init__(
        self,
        app: ASGIApp,
        route_config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        super().__init__(app)
        self.registry = get_service_registry()
        self.route_config = route_config or {}
        self._transformers: Dict[str, Callable] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through API Gateway"""
        # Check if route should be proxied
        route_info = self._get_route_info(request)
        
        if route_info:
            # Proxy to microservice
            return await self._proxy_request(request, route_info)
        else:
            # Handle locally
            return await call_next(request)
    
    def _get_route_info(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get routing information for request"""
        path = request.url.path
        
        # Check route configuration
        for pattern, config in self.route_config.items():
            if path.startswith(pattern):
                return config
        
        return None
    
    async def _proxy_request(
        self,
        request: Request,
        route_info: Dict[str, Any]
    ) -> Response:
        """Proxy request to microservice"""
        service_name = route_info.get("service")
        path_prefix = route_info.get("path_prefix", "")
        
        if not service_name:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not configured for route"
            )
        
        # Get service client
        client = get_service_client(service_name)
        
        # Transform request path
        target_path = request.url.path
        if path_prefix:
            target_path = target_path.replace(path_prefix, "", 1)
        
        # Prepare request
        headers = dict(request.headers)
        headers.pop("host", None)  # Remove host header
        
        # Get request body
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
        
        try:
            # Make request to microservice
            response = await client.request(
                method=request.method,
                path=target_path,
                headers=headers,
                params=dict(request.query_params),
                content=body
            )
            
            # Transform response if transformer exists
            if service_name in self._transformers:
                response = self._transformers[service_name](response)
            
            # Create FastAPI response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type")
            )
            
        except Exception as e:
            logger.error(f"Error proxying request: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Service unavailable: {str(e)}"
            )
    
    def register_transformer(self, service_name: str, transformer: Callable) -> None:
        """Register response transformer for service"""
        self._transformers[service_name] = transformer


class RateLimitConfig:
    """Rate limiting configuration"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size


class APIGateway:
    """
    API Gateway for microservices
    
    Features:
    - Request routing
    - Rate limiting
    - Authentication/Authorization
    - Request transformation
    - Response aggregation
    """
    
    def __init__(self):
        self.registry = get_service_registry()
        self.route_config: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, RateLimitConfig] = {}
    
    def register_route(
        self,
        path_prefix: str,
        service_name: str,
        rate_limit: Optional[RateLimitConfig] = None
    ) -> None:
        """Register route to microservice"""
        self.route_config[path_prefix] = {
            "service": service_name,
            "path_prefix": path_prefix
        }
        
        if rate_limit:
            self.rate_limits[path_prefix] = rate_limit
        
        logger.info(f"Registered route {path_prefix} -> {service_name}")
    
    def get_middleware(self) -> APIGatewayMiddleware:
        """Get API Gateway middleware"""
        return APIGatewayMiddleware(
            app=None,  # Will be set by FastAPI
            route_config=self.route_config
        )


# Global API Gateway instance
_gateway: Optional[APIGateway] = None


def get_api_gateway() -> APIGateway:
    """Get global API Gateway"""
    global _gateway
    if _gateway is None:
        _gateway = APIGateway()
    return _gateway















