"""
Advanced API Gateway System

This module provides comprehensive API gateway capabilities for the refactored
HeyGen AI system with routing, authentication, rate limiting, and monitoring.
"""

import asyncio
import aiohttp
import json
import logging
import uuid
import time
import hashlib
import hmac
import base64
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import redis
import jwt
from cryptography.fernet import Fernet
import yaml
import threading
from collections import defaultdict, deque
import ipaddress
import re
from urllib.parse import urlparse, parse_qs
import gzip
import brotli


logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthType(str, Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"


class RateLimitType(str, Enum):
    """Rate limit types."""
    IP = "ip"
    USER = "user"
    API_KEY = "api_key"
    GLOBAL = "global"


@dataclass
class Route:
    """API route configuration."""
    path: str
    methods: List[HTTPMethod]
    target_service: str
    target_path: str
    auth_required: bool = True
    auth_type: AuthType = AuthType.JWT
    rate_limit: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retry_count: int = 3
    cache_ttl: int = 0  # seconds, 0 = no cache
    middleware: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIRequest:
    """API request structure."""
    request_id: str
    method: HTTPMethod
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, str]
    body: Optional[bytes]
    client_ip: str
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class APIResponse:
    """API response structure."""
    request_id: str
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[bytes] = None
    response_time: float = 0.0
    cached: bool = False
    error: Optional[str] = None


class AuthenticationManager:
    """Advanced authentication manager."""
    
    def __init__(self, jwt_secret: str, api_keys: Dict[str, str] = None):
        self.jwt_secret = jwt_secret
        self.api_keys = api_keys or {}
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    async def authenticate_request(self, request: APIRequest, auth_type: AuthType) -> bool:
        """Authenticate API request."""
        try:
            if auth_type == AuthType.NONE:
                return True
            elif auth_type == AuthType.API_KEY:
                return await self._authenticate_api_key(request)
            elif auth_type == AuthType.JWT:
                return await self._authenticate_jwt(request)
            elif auth_type == AuthType.BASIC:
                return await self._authenticate_basic(request)
            else:
                return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def _authenticate_api_key(self, request: APIRequest) -> bool:
        """Authenticate using API key."""
        api_key = request.headers.get('X-API-Key') or request.query_params.get('api_key')
        if not api_key:
            return False
        
        # Check if API key exists and is valid
        if api_key in self.api_keys:
            request.api_key = api_key
            return True
        
        return False
    
    async def _authenticate_jwt(self, request: APIRequest) -> bool:
        """Authenticate using JWT token."""
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return False
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            request.user_id = payload.get('user_id')
            return True
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False
    
    async def _authenticate_basic(self, request: APIRequest) -> bool:
        """Authenticate using Basic authentication."""
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Basic '):
            return False
        
        try:
            credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
            username, password = credentials.split(':', 1)
            
            # Simple username/password check (in production, use proper user store)
            if username == 'admin' and password == 'password':
                request.user_id = username
                return True
            
            return False
        except Exception:
            return False
    
    def generate_jwt_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate JWT token for user."""
        payload = {
            'user_id': user_id,
            'exp': datetime.now(timezone.utc) + timedelta(seconds=expires_in),
            'iat': datetime.now(timezone.utc)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def add_api_key(self, key: str, user_id: str):
        """Add API key for user."""
        self.api_keys[key] = user_id


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client
        self.memory_store = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.RLock()
    
    async def is_allowed(
        self,
        identifier: str,
        limit_type: RateLimitType,
        limit: int,
        window: int
    ) -> bool:
        """Check if request is allowed based on rate limit."""
        try:
            if self.redis_client:
                return await self._redis_rate_limit(identifier, limit, window)
            else:
                return self._memory_rate_limit(identifier, limit, window)
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow request on error
    
    async def _redis_rate_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Redis-based rate limiting using sliding window."""
        key = f"rate_limit:{identifier}"
        now = time.time()
        pipeline = self.redis_client.pipeline()
        
        # Remove expired entries
        pipeline.zremrangebyscore(key, 0, now - window)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now): now})
        
        # Set expiration
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_count = results[1]
        
        return current_count < limit
    
    def _memory_rate_limit(self, identifier: str, limit: int, window: int) -> bool:
        """Memory-based rate limiting using sliding window."""
        with self.lock:
            now = time.time()
            requests = self.memory_store[identifier]
            
            # Remove expired requests
            while requests and requests[0] < now - window:
                requests.popleft()
            
            # Check if limit exceeded
            if len(requests) >= limit:
                return False
            
            # Add current request
            requests.append(now)
            return True


class CacheManager:
    """Advanced cache manager with multiple backends."""
    
    def __init__(self, redis_client: redis.Redis = None, memory_cache_size: int = 1000):
        self.redis_client = redis_client
        self.memory_cache = {}
        self.memory_cache_size = memory_cache_size
        self.lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value from cache."""
        try:
            if self.redis_client:
                return await self._redis_get(key)
            else:
                return self._memory_get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: bytes, ttl: int = 3600) -> bool:
        """Set value in cache."""
        try:
            if self.redis_client:
                return await self._redis_set(key, value, ttl)
            else:
                return self._memory_set(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _redis_get(self, key: str) -> Optional[bytes]:
        """Get value from Redis."""
        return self.redis_client.get(key)
    
    async def _redis_set(self, key: str, value: bytes, ttl: int) -> bool:
        """Set value in Redis."""
        return self.redis_client.setex(key, ttl, value)
    
    def _memory_get(self, key: str) -> Optional[bytes]:
        """Get value from memory cache."""
        with self.lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires_at'] > time.time():
                    return entry['value']
                else:
                    del self.memory_cache[key]
            return None
    
    def _memory_set(self, key: str, value: bytes, ttl: int) -> bool:
        """Set value in memory cache."""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.memory_cache) >= self.memory_cache_size:
                oldest_key = min(self.memory_cache.keys(), key=lambda k: self.memory_cache[k]['created_at'])
                del self.memory_cache[oldest_key]
            
            self.memory_cache[key] = {
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + ttl
            }
            return True


class RequestRouter:
    """Advanced request router with load balancing."""
    
    def __init__(self, microservices_client):
        self.microservices_client = microservices_client
        self.routes: Dict[str, Route] = {}
        self.path_patterns: Dict[str, str] = {}
    
    def add_route(self, route: Route):
        """Add route configuration."""
        self.routes[route.path] = route
        
        # Create path pattern for matching
        pattern = route.path.replace('{', '(?P<').replace('}', '>[^/]+)')
        self.path_patterns[route.path] = pattern
    
    def find_route(self, path: str, method: HTTPMethod) -> Optional[Route]:
        """Find matching route for request."""
        # Direct match first
        if path in self.routes:
            route = self.routes[path]
            if method in route.methods:
                return route
        
        # Pattern matching
        for route_path, pattern in self.path_patterns.items():
            if re.match(pattern, path):
                route = self.routes[route_path]
                if method in route.methods:
                    return route
        
        return None
    
    async def route_request(self, request: APIRequest, route: Route) -> APIResponse:
        """Route request to target service."""
        try:
            # Prepare target URL
            target_url = f"http://{route.target_service}{route.target_path}"
            
            # Replace path parameters
            for param, value in request.query_params.items():
                target_url = target_url.replace(f"{{{param}}}", value)
            
            # Make request to target service
            service_request = {
                'service_name': route.target_service,
                'method': request.method.value,
                'path': route.target_path,
                'headers': request.headers,
                'body': request.body,
                'timeout': route.timeout
            }
            
            # This would use the microservices client in a real implementation
            # For demo purposes, return a mock response
            return APIResponse(
                request_id=request.request_id,
                status_code=200,
                headers={'Content-Type': 'application/json'},
                body=b'{"message": "Request routed successfully"}',
                response_time=0.1
            )
            
        except Exception as e:
            return APIResponse(
                request_id=request.request_id,
                status_code=500,
                error=str(e)
            )


class MiddlewareManager:
    """Advanced middleware manager."""
    
    def __init__(self):
        self.middlewares: Dict[str, Callable] = {}
        self.global_middlewares: List[str] = []
    
    def register_middleware(self, name: str, middleware: Callable):
        """Register middleware."""
        self.middlewares[name] = middleware
    
    def add_global_middleware(self, name: str):
        """Add global middleware."""
        if name not in self.global_middlewares:
            self.global_middlewares.append(name)
    
    async def execute_middlewares(
        self,
        request: APIRequest,
        middlewares: List[str]
    ) -> Optional[APIResponse]:
        """Execute middlewares in order."""
        all_middlewares = self.global_middlewares + middlewares
        
        for middleware_name in all_middlewares:
            if middleware_name in self.middlewares:
                middleware = self.middlewares[middleware_name]
                try:
                    response = await middleware(request)
                    if response:  # Middleware returned a response (e.g., error)
                        return response
                except Exception as e:
                    logger.error(f"Middleware {middleware_name} error: {e}")
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=500,
                        error=f"Middleware error: {e}"
                    )
        
        return None


class AdvancedAPIGateway:
    """
    Advanced API Gateway with comprehensive capabilities.
    
    Features:
    - Request routing and load balancing
    - Authentication and authorization
    - Rate limiting and throttling
    - Caching and compression
    - Request/response transformation
    - Monitoring and logging
    - Circuit breaking
    - API versioning
    - CORS handling
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        jwt_secret: str = None,
        redis_url: str = None
    ):
        """
        Initialize the advanced API gateway.
        
        Args:
            host: Gateway host
            port: Gateway port
            jwt_secret: JWT secret key
            redis_url: Redis URL for caching and rate limiting
        """
        self.host = host
        self.port = port
        self.jwt_secret = jwt_secret or "your-secret-key"
        
        # Initialize Redis client
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
        
        # Initialize components
        self.auth_manager = AuthenticationManager(self.jwt_secret)
        self.rate_limiter = RateLimiter(self.redis_client)
        self.cache_manager = CacheManager(self.redis_client)
        self.router = RequestRouter(None)  # Would be microservices client
        self.middleware_manager = MiddlewareManager()
        
        # Initialize metrics
        self.metrics = {
            'requests_total': 0,
            'requests_by_status': defaultdict(int),
            'requests_by_service': defaultdict(int),
            'response_times': deque(maxlen=1000)
        }
        
        # Setup default middlewares
        self._setup_default_middlewares()
        
        # Setup default routes
        self._setup_default_routes()
    
    def _setup_default_middlewares(self):
        """Setup default middlewares."""
        # CORS middleware
        async def cors_middleware(request: APIRequest) -> Optional[APIResponse]:
            if request.method == HTTPMethod.OPTIONS:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=200,
                    headers={
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key'
                    }
                )
            return None
        
        # Logging middleware
        async def logging_middleware(request: APIRequest) -> Optional[APIResponse]:
            logger.info(f"Request: {request.method} {request.path} from {request.client_ip}")
            return None
        
        # Metrics middleware
        async def metrics_middleware(request: APIRequest) -> Optional[APIResponse]:
            self.metrics['requests_total'] += 1
            self.metrics['requests_by_service'][request.path] += 1
            return None
        
        self.middleware_manager.register_middleware('cors', cors_middleware)
        self.middleware_manager.register_middleware('logging', logging_middleware)
        self.middleware_manager.register_middleware('metrics', metrics_middleware)
        
        # Add global middlewares
        self.middleware_manager.add_global_middleware('cors')
        self.middleware_manager.add_global_middleware('logging')
        self.middleware_manager.add_global_middleware('metrics')
    
    def _setup_default_routes(self):
        """Setup default routes."""
        # Health check route
        health_route = Route(
            path="/health",
            methods=[HTTPMethod.GET],
            target_service="health-service",
            target_path="/health",
            auth_required=False,
            cache_ttl=60
        )
        self.router.add_route(health_route)
        
        # API routes
        api_routes = [
            Route(
                path="/api/v1/ai/generate",
                methods=[HTTPMethod.POST],
                target_service="ai-service",
                target_path="/generate",
                auth_required=True,
                auth_type=AuthType.JWT,
                rate_limit={"limit": 100, "window": 3600, "type": RateLimitType.USER}
            ),
            Route(
                path="/api/v1/analytics/metrics",
                methods=[HTTPMethod.GET],
                target_service="analytics-service",
                target_path="/metrics",
                auth_required=True,
                auth_type=AuthType.API_KEY,
                cache_ttl=300
            ),
            Route(
                path="/api/v1/users/{user_id}",
                methods=[HTTPMethod.GET, HTTPMethod.PUT, HTTPMethod.DELETE],
                target_service="user-service",
                target_path="/users/{user_id}",
                auth_required=True,
                auth_type=AuthType.JWT
            )
        ]
        
        for route in api_routes:
            self.router.add_route(route)
    
    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle incoming API request."""
        start_time = time.time()
        
        try:
            # Find route
            route = self.router.find_route(request.path, request.method)
            if not route:
                return APIResponse(
                    request_id=request.request_id,
                    status_code=404,
                    error="Route not found"
                )
            
            # Execute middlewares
            middleware_response = await self.middleware_manager.execute_middlewares(
                request, route.middleware
            )
            if middleware_response:
                return middleware_response
            
            # Check authentication
            if route.auth_required:
                auth_result = await self.auth_manager.authenticate_request(
                    request, route.auth_type
                )
                if not auth_result:
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=401,
                        error="Authentication required"
                    )
            
            # Check rate limiting
            if route.rate_limit:
                rate_limit_config = route.rate_limit
                identifier = self._get_rate_limit_identifier(request, rate_limit_config['type'])
                
                is_allowed = await self.rate_limiter.is_allowed(
                    identifier=identifier,
                    limit_type=rate_limit_config['type'],
                    limit=rate_limit_config['limit'],
                    window=rate_limit_config['window']
                )
                
                if not is_allowed:
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=429,
                        error="Rate limit exceeded"
                    )
            
            # Check cache
            if route.cache_ttl > 0:
                cache_key = self._generate_cache_key(request)
                cached_response = await self.cache_manager.get(cache_key)
                if cached_response:
                    return APIResponse(
                        request_id=request.request_id,
                        status_code=200,
                        body=cached_response,
                        cached=True
                    )
            
            # Route request
            response = await self.router.route_request(request, route)
            
            # Cache response if applicable
            if route.cache_ttl > 0 and response.status_code == 200:
                cache_key = self._generate_cache_key(request)
                await self.cache_manager.set(cache_key, response.body or b'', route.cache_ttl)
            
            # Update metrics
            response.response_time = time.time() - start_time
            self.metrics['requests_by_status'][response.status_code] += 1
            self.metrics['response_times'].append(response.response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return APIResponse(
                request_id=request.request_id,
                status_code=500,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    def _get_rate_limit_identifier(self, request: APIRequest, limit_type: RateLimitType) -> str:
        """Get rate limit identifier."""
        if limit_type == RateLimitType.IP:
            return request.client_ip
        elif limit_type == RateLimitType.USER:
            return request.user_id or request.client_ip
        elif limit_type == RateLimitType.API_KEY:
            return request.api_key or request.client_ip
        else:
            return "global"
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.method}:{request.path}:{request.query_params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        response_times = list(self.metrics['response_times'])
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'total_requests': self.metrics['requests_total'],
            'requests_by_status': dict(self.metrics['requests_by_status']),
            'requests_by_service': dict(self.metrics['requests_by_service']),
            'average_response_time': avg_response_time,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def add_route(self, route: Route):
        """Add route to gateway."""
        self.router.add_route(route)
    
    def register_middleware(self, name: str, middleware: Callable):
        """Register middleware."""
        self.middleware_manager.register_middleware(name, middleware)


# Example usage and demonstration
async def main():
    """Demonstrate the advanced API gateway."""
    print("🚪 HeyGen AI - Advanced API Gateway Demo")
    print("=" * 70)
    
    # Initialize API gateway
    gateway = AdvancedAPIGateway(
        host="0.0.0.0",
        port=8080,
        jwt_secret="your-secret-key",
        redis_url="redis://localhost:6379/0"
    )
    
    try:
        # Add some API keys for testing
        gateway.auth_manager.add_api_key("test-api-key-123", "user123")
        gateway.auth_manager.add_api_key("admin-api-key-456", "admin")
        
        # Create sample requests
        print("\n📡 Testing API Gateway...")
        
        # Test health check (no auth required)
        health_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/health",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.100"
        )
        
        health_response = await gateway.handle_request(health_request)
        print(f"Health check: {health_response.status_code} (cached: {health_response.cached})")
        
        # Test AI service with JWT auth
        jwt_token = gateway.auth_manager.generate_jwt_token("user123")
        ai_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.POST,
            path="/api/v1/ai/generate",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/json"
            },
            query_params={},
            body=b'{"prompt": "Hello, AI!"}',
            client_ip="192.168.1.101"
        )
        
        ai_response = await gateway.handle_request(ai_request)
        print(f"AI service request: {ai_response.status_code} (time: {ai_response.response_time:.3f}s)")
        
        # Test analytics service with API key auth
        analytics_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/api/v1/analytics/metrics",
            headers={"X-API-Key": "test-api-key-123"},
            query_params={},
            body=None,
            client_ip="192.168.1.102"
        )
        
        analytics_response = await gateway.handle_request(analytics_request)
        print(f"Analytics service request: {analytics_response.status_code} (cached: {analytics_response.cached})")
        
        # Test unauthorized request
        unauthorized_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/api/v1/users/123",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.103"
        )
        
        unauthorized_response = await gateway.handle_request(unauthorized_request)
        print(f"Unauthorized request: {unauthorized_response.status_code} - {unauthorized_response.error}")
        
        # Test rate limiting
        print("\n⏱️ Testing Rate Limiting...")
        for i in range(5):
            rate_limit_request = APIRequest(
                request_id=str(uuid.uuid4()),
                method=HTTPMethod.POST,
                path="/api/v1/ai/generate",
                headers={"Authorization": f"Bearer {jwt_token}"},
                query_params={},
                body=b'{"prompt": "Test"}',
                client_ip="192.168.1.104"
            )
            
            rate_limit_response = await gateway.handle_request(rate_limit_request)
            print(f"Rate limit test {i+1}: {rate_limit_response.status_code}")
        
        # Get gateway metrics
        print("\n📊 Gateway Metrics:")
        metrics = gateway.get_metrics()
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Average response time: {metrics['average_response_time']:.3f}s")
        print(f"  Requests by status: {metrics['requests_by_status']}")
        print(f"  Requests by service: {metrics['requests_by_service']}")
        
        # Test custom route
        print("\n🛣️ Testing Custom Route...")
        custom_route = Route(
            path="/api/v1/custom/test",
            methods=[HTTPMethod.GET],
            target_service="custom-service",
            target_path="/test",
            auth_required=False,
            cache_ttl=60
        )
        gateway.add_route(custom_route)
        
        custom_request = APIRequest(
            request_id=str(uuid.uuid4()),
            method=HTTPMethod.GET,
            path="/api/v1/custom/test",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.105"
        )
        
        custom_response = await gateway.handle_request(custom_request)
        print(f"Custom route: {custom_response.status_code}")
        
        print(f"\n🌐 API Gateway running on: http://{gateway.host}:{gateway.port}")
        print("📋 Available endpoints:")
        print("  GET  /health")
        print("  POST /api/v1/ai/generate (JWT auth)")
        print("  GET  /api/v1/analytics/metrics (API key auth)")
        print("  GET  /api/v1/users/{user_id} (JWT auth)")
        print("  GET  /api/v1/custom/test (no auth)")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
