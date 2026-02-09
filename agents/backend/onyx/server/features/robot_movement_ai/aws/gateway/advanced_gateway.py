"""
Advanced API Gateway
====================

Production-ready API Gateway with:
- Rate limiting
- Authentication/Authorization
- Request transformation
- Response caching
- Request/Response logging
- Circuit breakers
- Load balancing
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.base import BaseHTTPMiddleware
from aws.services.base_service import BaseMicroservice, ServiceConfig
from aws.services.service_client import ServiceClientFactory
from aws.services.service_registry import get_service_registry
from aws.core.config_manager import AppConfig
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis client."""
        if self.redis_client is None and self.redis_url:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self.redis_client
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int = 100,
        window: int = 60
    ):
        """
        Check rate limit.
        
        Returns:
            tuple: (allowed, remaining)
        """
        redis = await self._get_redis()
        if not redis:
            return True, limit  # No Redis, allow all
        
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, window)
        
        remaining = max(0, limit - current)
        allowed = current <= limit
        
        return allowed, remaining


class GatewayAuth:
    """Gateway authentication."""
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.jwt_secret = jwt_secret
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        if not self.jwt_secret:
            return None  # Auth disabled
        
        try:
            import jwt
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return None


class AdvancedAPIGateway(BaseMicroservice):
    """Advanced API Gateway with production features."""
    
    def __init__(self, config: Optional[ServiceConfig] = None, app_config: Optional[AppConfig] = None):
        if config is None:
            config = ServiceConfig(
                service_name="api-gateway",
                service_version="1.0.0",
                port=8000
            )
        super().__init__(config)
        self.app_config = app_config or AppConfig.from_env()
        self.registry = get_service_registry()
        self.rate_limiter = RateLimiter(
            redis_url=self.app_config.cache.redis_url
        )
        self.auth = GatewayAuth(
            jwt_secret=self.app_config.security.jwt_secret_key
        )
        self._routes: Dict[str, Dict[str, Any]] = {
            "/api/v1/move": {
                "service": "movement-service",
                "rate_limit": 100,
                "cache_ttl": 0,  # No cache for POST
                "require_auth": True,
            },
            "/api/v1/chat": {
                "service": "chat-service",
                "rate_limit": 200,
                "cache_ttl": 0,
                "require_auth": True,
            },
            "/api/v1/trajectory": {
                "service": "trajectory-service",
                "rate_limit": 50,
                "cache_ttl": 300,  # Cache for 5 minutes
                "require_auth": False,
            },
        }
        self._cache: Optional[aioredis.Redis] = None
    
    def create_app(self) -> FastAPI:
        """Create FastAPI app for API Gateway."""
        app = FastAPI(
            title="Robot Movement AI API Gateway",
            description="Advanced API Gateway for microservices",
            version=self.config.service_version
        )
        return app
    
    def get_dependencies(self) -> Dict[str, Any]:
        """Get service dependencies."""
        return {}
    
    def _setup_middleware(self):
        """Setup advanced middleware."""
        super()._setup_middleware()
        
        # Request ID middleware
        class RequestIDMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                request_id = request.headers.get("X-Request-ID") or f"req-{int(time.time() * 1000)}"
                request.state.request_id = request_id
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
        
        self.app.add_middleware(RequestIDMiddleware)
        
        # Rate limiting middleware
        gateway_instance = self
        
        class RateLimitMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Skip rate limiting for health checks
                if request.url.path in ["/health", "/metrics", "/gateway/services", "/gateway/health"]:
                    return await call_next(request)
                
                # Get client identifier
                client_ip = request.client.host if request.client else "unknown"
                rate_limit_key = f"rate_limit:{client_ip}:{request.url.path}"
                
                # Check rate limit
                route_config = gateway_instance._get_route_config(request.url.path)
                limit = route_config.get("rate_limit", 100) if route_config else 100
                
                allowed, remaining = await gateway_instance.rate_limiter.check_rate_limit(
                    rate_limit_key,
                    limit=limit,
                    window=60
                )
                
                if not allowed:
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={"error": "Rate limit exceeded"},
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": "60"
                        }
                    )
                
                response = await call_next(request)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                return response
        
        self.app.add_middleware(RateLimitMiddleware)
        
        # Logging middleware
        class LoggingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                start_time = time.time()
                request_id = getattr(request.state, "request_id", "unknown")
                
                logger.info(
                    "Gateway request",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "client_ip": request.client.host if request.client else None,
                    }
                )
                
                try:
                    response = await call_next(request)
                    process_time = time.time() - start_time
                    
                    logger.info(
                        "Gateway response",
                        extra={
                            "request_id": request_id,
                            "status_code": response.status_code,
                            "process_time": process_time,
                        }
                    )
                    
                    response.headers["X-Process-Time"] = str(process_time)
                    return response
                except Exception as e:
                    logger.error(
                        "Gateway error",
                        extra={
                            "request_id": request_id,
                            "error": str(e),
                        },
                        exc_info=True
                    )
                    raise
        
        self.app.add_middleware(LoggingMiddleware)
    
    def _get_route_config(self, path: str) -> Optional[Dict[str, Any]]:
        """Get route configuration."""
        for route_prefix, config in self._routes.items():
            if path.startswith(route_prefix.replace("/api/v1/", "")):
                return config
        return None
    
    async def _get_cache(self) -> Optional[aioredis.Redis]:
        """Get cache client."""
        if self._cache is None and self.app_config.cache.redis_url:
            self._cache = await aioredis.from_url(
                self.app_config.cache.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._cache
    
    def _generate_cache_key(self, method: str, path: str, params: Dict, body: Optional[bytes] = None) -> str:
        """Generate cache key."""
        key_parts = [method, path, json.dumps(params, sort_keys=True)]
        if body:
            key_parts.append(hashlib.md5(body).hexdigest())
        return f"gateway_cache:{hashlib.md5(':'.join(key_parts).encode()).hexdigest()}"
    
    def _setup_routes(self):
        """Setup API Gateway routes."""
        @self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
        async def gateway_route(request: Request, path: str):
            """Route requests to appropriate microservice."""
            # Get route configuration
            route_config = self._get_route_config(path)
            if not route_config:
                raise HTTPException(status_code=404, detail="Service not found")
            
            target_service = route_config["service"]
            require_auth = route_config.get("require_auth", False)
            cache_ttl = route_config.get("cache_ttl", 0)
            
            # Authentication
            if require_auth:
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                token = auth_header.replace("Bearer ", "")
                user = await self.auth.verify_token(token)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token"
                    )
                request.state.user = user
            
            # Check cache for GET requests
            if request.method == "GET" and cache_ttl > 0:
                cache = await self._get_cache()
                if cache:
                    cache_key = self._generate_cache_key(
                        request.method,
                        path,
                        dict(request.query_params)
                    )
                    cached_response = await cache.get(cache_key)
                    if cached_response:
                        return JSONResponse(
                            content=json.loads(cached_response),
                            headers={"X-Cache": "HIT"}
                        )
            
            # Get service client
            client = ServiceClientFactory.get_client(target_service)
            
            try:
                # Forward request
                method = request.method
                body = await request.body() if method in ["POST", "PUT", "PATCH"] else None
                params = dict(request.query_params)
                headers = dict(request.headers)
                
                # Remove gateway-specific headers
                headers.pop("host", None)
                headers.pop("content-length", None)
                headers.pop("x-forwarded-for", None)
                
                # Add request ID
                headers["X-Request-ID"] = getattr(request.state, "request_id", "")
                
                # Make request
                import json as json_lib
                request_kwargs = {
                    "params": params,
                    "headers": headers
                }
                if body:
                    try:
                        request_kwargs["json"] = json_lib.loads(body.decode())
                    except:
                        request_kwargs["content"] = body
                
                response = await client._make_request(
                    method,
                    f"/{path}",
                    **request_kwargs
                )
                
                response_data = response.json()
                status_code = response.status_code
                
                # Cache response for GET requests
                if request.method == "GET" and cache_ttl > 0 and status_code == 200:
                    cache = await self._get_cache()
                    if cache:
                        cache_key = self._generate_cache_key(
                            request.method,
                            path,
                            dict(request.query_params)
                        )
                        await cache.setex(
                            cache_key,
                            cache_ttl,
                            json.dumps(response_data)
                        )
                
                return JSONResponse(
                    content=response_data,
                    status_code=status_code,
                    headers={
                        **dict(response.headers),
                        "X-Cache": "MISS" if request.method == "GET" else "N/A"
                    }
                )
            except Exception as e:
                logger.error(f"Gateway routing failed: {e}", exc_info=True)
                raise HTTPException(status_code=502, detail=f"Service error: {str(e)}")
        
        @self.app.get("/gateway/services")
        async def list_services():
            """List all registered services."""
            services = []
            for service_name in self.registry.list_services():
                instances = self.registry.get_instances(service_name, healthy_only=True)
                services.append({
                    "name": service_name,
                    "instances": len(instances),
                    "healthy": len(instances) > 0
                })
            
            return {
                "services": services,
                "routes": {k: v["service"] for k, v in self._routes.items()}
            }
        
        @self.app.get("/gateway/health")
        async def gateway_health():
            """Gateway health check."""
            return {
                "status": "healthy",
                "service": "api-gateway",
                "services_registered": len(self.registry.list_services()),
                "cache_enabled": self.app_config.cache.redis_url is not None,
                "auth_enabled": self.app_config.security.jwt_secret_key != "change-me-in-production"
            }

