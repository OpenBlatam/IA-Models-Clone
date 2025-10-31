"""
API Gateway for Export IA
========================

Advanced API Gateway that provides unified access to all Export IA
microservices with routing, authentication, rate limiting, and monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import time
import hashlib
import hmac
import base64
from pathlib import Path
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
import yaml

from ..microservices.service_orchestrator import ServiceOrchestrator, get_global_orchestrator

logger = logging.getLogger(__name__)

class AuthenticationMethod(Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    NONE = "none"

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RouteConfig:
    """Route configuration."""
    path: str
    methods: List[str]
    service: str
    endpoint: str
    authentication_required: bool = True
    rate_limit: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    cache_ttl: int = 0
    transform_request: Optional[Callable] = None
    transform_response: Optional[Callable] = None

@dataclass
class APIConfig:
    """API Gateway configuration."""
    title: str = "Export IA API Gateway"
    version: str = "1.0.0"
    description: str = "Unified API Gateway for Export IA Services"
    authentication_method: AuthenticationMethod = AuthenticationMethod.JWT
    jwt_secret: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    default_rate_limit: str = "100/minute"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_ttl: int = 300

# Prometheus metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_gateway_active_connections', 'Active connections')
ERROR_COUNT = Counter('api_gateway_errors_total', 'Total API errors', ['error_type'])

class AuthenticationManager:
    """Authentication and authorization manager."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self._load_api_keys()
    
    def _load_api_keys(self) -> None:
        """Load API keys from configuration."""
        # This would typically load from a database or secure storage
        self.api_keys = {
            "demo-key": {
                "user_id": "demo_user",
                "permissions": ["read", "write"],
                "rate_limit": "1000/hour",
                "expires_at": None
            }
        }
    
    async def authenticate_jwt(self, token: str) -> Dict[str, Any]:
        """Authenticate JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check expiration
            if payload.get('exp', 0) < time.time():
                raise HTTPException(status_code=401, detail="Token expired")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate API key."""
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        key_info = self.api_keys[api_key]
        
        # Check expiration
        if key_info.get('expires_at') and key_info['expires_at'] < datetime.now():
            raise HTTPException(status_code=401, detail="API key expired")
        
        return {
            "user_id": key_info["user_id"],
            "permissions": key_info["permissions"],
            "rate_limit": key_info.get("rate_limit")
        }
    
    async def authenticate_request(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Authenticate request based on configured method."""
        if self.config.authentication_method == AuthenticationMethod.JWT:
            return await self.authenticate_jwt(credentials.credentials)
        elif self.config.authentication_method == AuthenticationMethod.API_KEY:
            return await self.authenticate_api_key(credentials.credentials)
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
    
    def generate_jwt_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "iat": time.time(),
            "exp": time.time() + self.config.jwt_expiration
        }
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, redis_client: redis.Redis, strategy: RateLimitStrategy):
        self.redis = redis_client
        self.strategy = strategy
    
    async def is_allowed(self, key: str, limit: str) -> bool:
        """Check if request is allowed based on rate limit."""
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(key, limit)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(key, limit)
        else:
            return True  # Default to allowing
    
    async def _sliding_window_check(self, key: str, limit: str) -> bool:
        """Sliding window rate limiting."""
        try:
            count, window = limit.split('/')
            count = int(count)
            
            # Parse window (e.g., "minute", "hour", "day")
            window_seconds = self._parse_window(window)
            
            # Use Redis sorted set for sliding window
            now = time.time()
            pipeline = self.redis.pipeline()
            
            # Remove expired entries
            pipeline.zremrangebyscore(key, 0, now - window_seconds)
            
            # Count current entries
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiration
            pipeline.expire(key, window_seconds)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            return current_count < count
        
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow on error
    
    async def _fixed_window_check(self, key: str, limit: str) -> bool:
        """Fixed window rate limiting."""
        try:
            count, window = limit.split('/')
            count = int(count)
            
            window_seconds = self._parse_window(window)
            window_start = int(time.time() // window_seconds) * window_seconds
            window_key = f"{key}:{window_start}"
            
            current_count = await self.redis.incr(window_key)
            if current_count == 1:
                await self.redis.expire(window_key, window_seconds)
            
            return current_count <= count
        
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True
    
    def _parse_window(self, window: str) -> int:
        """Parse time window string to seconds."""
        window_map = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        return window_map.get(window, 60)

class CacheManager:
    """Response caching manager."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        try:
            cached = await self.redis.get(key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, data: Dict[str, Any], ttl: int = 300) -> None:
        """Cache response data."""
        try:
            await self.redis.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_data = {
            "path": request.url.path,
            "query": str(request.query_params),
            "method": request.method,
            "headers": dict(request.headers)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

class APIGateway:
    """Main API Gateway implementation."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.app = FastAPI(
            title=config.title,
            version=config.version,
            description=config.description
        )
        
        # Initialize components
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.auth_manager = AuthenticationManager(config)
        self.rate_limiter = RateLimiter(self.redis_client, config.rate_limit_strategy)
        self.cache_manager = CacheManager(self.redis_client)
        self.service_orchestrator = get_global_orchestrator()
        
        # Route configuration
        self.routes: Dict[str, RouteConfig] = {}
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_metrics()
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Log request
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(process_time)
            
            return response
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            if self.config.enable_metrics:
                return Response(
                    prometheus_client.generate_latest(),
                    media_type="text/plain"
                )
            else:
                raise HTTPException(status_code=404, detail="Metrics disabled")
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(credentials: dict):
            # This would validate user credentials
            user_id = credentials.get("user_id", "demo_user")
            token = self.auth_manager.generate_jwt_token(user_id)
            return {"access_token": token, "token_type": "bearer"}
        
        # Service discovery endpoints
        @self.app.get("/services")
        async def list_services():
            return {
                "services": list(self.service_orchestrator.service_definitions.keys()),
                "running": list(self.service_orchestrator.running_services.keys())
            }
        
        # Dynamic route handler
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(request: Request, path: str):
            return await self._handle_proxy_request(request, path)
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        if self.config.enable_metrics:
            # Start metrics server
            prometheus_client.start_http_server(9090)
    
    def add_route(self, route_config: RouteConfig) -> None:
        """Add a route configuration."""
        self.routes[route_config.path] = route_config
        logger.info(f"Added route: {route_config.path} -> {route_config.service}")
    
    def load_routes_from_config(self, config_file: str) -> None:
        """Load routes from configuration file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for route_data in config_data.get('routes', []):
                route_config = RouteConfig(**route_data)
                self.add_route(route_config)
            
            logger.info(f"Loaded {len(config_data.get('routes', []))} routes from {config_file}")
        
        except Exception as e:
            logger.error(f"Failed to load routes from {config_file}: {e}")
    
    async def _handle_proxy_request(self, request: Request, path: str) -> JSONResponse:
        """Handle proxy request to backend services."""
        
        # Find matching route
        route_config = self._find_matching_route(path, request.method)
        if not route_config:
            raise HTTPException(status_code=404, detail="Route not found")
        
        # Check authentication
        if route_config.authentication_required:
            auth_result = await self._authenticate_request(request)
        else:
            auth_result = None
        
        # Check rate limiting
        if route_config.rate_limit:
            rate_limit_key = self._get_rate_limit_key(request, auth_result)
            if not await self.rate_limiter.is_allowed(rate_limit_key, route_config.rate_limit):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Check cache
        if self.config.enable_caching and route_config.cache_ttl > 0:
            cache_key = self.cache_manager.generate_cache_key(request)
            cached_response = await self.cache_manager.get(cache_key)
            if cached_response:
                return JSONResponse(content=cached_response)
        
        # Transform request if needed
        request_data = await self._prepare_request_data(request)
        if route_config.transform_request:
            request_data = route_config.transform_request(request_data)
        
        # Make service call
        try:
            response_data = await self.service_orchestrator.call_service(
                service_name=route_config.service,
                endpoint=route_config.endpoint,
                method=request.method,
                data=request_data,
                headers=dict(request.headers)
            )
            
            # Transform response if needed
            if route_config.transform_response:
                response_data = route_config.transform_response(response_data)
            
            # Cache response
            if self.config.enable_caching and route_config.cache_ttl > 0:
                await self.cache_manager.set(cache_key, response_data, route_config.cache_ttl)
            
            return JSONResponse(content=response_data)
        
        except Exception as e:
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            logger.error(f"Service call failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _find_matching_route(self, path: str, method: str) -> Optional[RouteConfig]:
        """Find matching route configuration."""
        for route_path, route_config in self.routes.items():
            if self._path_matches(route_path, path) and method in route_config.methods:
                return route_config
        return None
    
    def _path_matches(self, route_path: str, request_path: str) -> bool:
        """Check if request path matches route pattern."""
        # Simple path matching - could be enhanced with regex or path parameters
        if route_path.endswith('*'):
            return request_path.startswith(route_path[:-1])
        else:
            return route_path == request_path
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate the request."""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        try:
            scheme, credentials = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                raise HTTPException(status_code=401, detail="Invalid authentication scheme")
            
            return await self.auth_manager.authenticate_request(
                HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
            )
        
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    def _get_rate_limit_key(self, request: Request, auth_result: Optional[Dict[str, Any]]) -> str:
        """Generate rate limit key for request."""
        if auth_result:
            return f"rate_limit:{auth_result.get('user_id', 'anonymous')}"
        else:
            # Use IP address for anonymous requests
            client_ip = request.client.host
            return f"rate_limit:ip:{client_ip}"
    
    async def _prepare_request_data(self, request: Request) -> Dict[str, Any]:
        """Prepare request data for service call."""
        data = {}
        
        # Add query parameters
        if request.query_params:
            data["query"] = dict(request.query_params)
        
        # Add request body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
                data["body"] = body
            except:
                data["body"] = {}
        
        return data
    
    async def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API Gateway."""
        logger.info(f"Starting API Gateway on {host}:{port}")
        
        # Initialize service orchestrator
        await self.service_orchestrator.initialize()
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self) -> None:
        """Shutdown the API Gateway."""
        await self.service_orchestrator.shutdown()
        logger.info("API Gateway shutdown complete")

# Global API Gateway instance
_global_gateway: Optional[APIGateway] = None

def get_global_gateway() -> APIGateway:
    """Get the global API Gateway instance."""
    global _global_gateway
    if _global_gateway is None:
        config = APIConfig()
        _global_gateway = APIGateway(config)
    return _global_gateway

# Example usage
if __name__ == "__main__":
    # Create API Gateway
    config = APIConfig(
        title="Export IA API Gateway",
        version="1.0.0",
        authentication_method=AuthenticationMethod.JWT,
        enable_metrics=True,
        enable_caching=True
    )
    
    gateway = APIGateway(config)
    
    # Add some example routes
    gateway.add_route(RouteConfig(
        path="/api/v1/documents",
        methods=["GET", "POST"],
        service="document-processor",
        endpoint="/documents",
        authentication_required=True,
        rate_limit="100/minute"
    ))
    
    gateway.add_route(RouteConfig(
        path="/api/v1/ai/analyze",
        methods=["POST"],
        service="ai-engine",
        endpoint="/analyze",
        authentication_required=True,
        rate_limit="50/minute"
    ))
    
    # Start the gateway
    asyncio.run(gateway.start())



























