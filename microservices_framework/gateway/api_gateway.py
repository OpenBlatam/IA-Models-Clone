"""
Advanced API Gateway Implementation
Features: Rate limiting, request transformation, security filtering, load balancing
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import logging
from urllib.parse import urlparse, urljoin
import jwt
from datetime import datetime, timedelta

from ..shared.core.service_registry import ServiceRegistry, ServiceInstance
from ..shared.core.circuit_breaker import HTTPCircuitBreaker, CircuitBreakerConfig

logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Rate limit types"""
    IP = "ip"
    USER = "user"
    API_KEY = "api_key"
    GLOBAL = "global"

class SecurityLevel(Enum):
    """Security levels"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    window_size: int = 60  # seconds

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    api_key_header: str = "X-API-Key"
    rate_limit_by: RateLimitType = RateLimitType.IP
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class RouteConfig:
    """Route configuration"""
    path: str
    service_name: str
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    rate_limit: Optional[RateLimitConfig] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    request_transform: Optional[Dict[str, Any]] = None
    response_transform: Optional[Dict[str, Any]] = None

class RateLimiter:
    """
    Redis-based rate limiter with sliding window
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(
        self, 
        key: str, 
        config: RateLimitConfig,
        window_start: Optional[float] = None
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limit
        
        Args:
            key: Rate limit key (IP, user ID, etc.)
            config: Rate limit configuration
            window_start: Window start time (for testing)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        now = time.time()
        window_start = window_start or (now - config.window_size)
        
        # Create Redis keys
        window_key = f"rate_limit:{key}:{int(window_start)}"
        counter_key = f"rate_limit:{key}:counter"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Increment counter
        pipe.incr(counter_key)
        pipe.expire(counter_key, config.window_size)
        
        # Get current count
        pipe.get(counter_key)
        
        results = await pipe.execute()
        current_count = int(results[2] or 0)
        
        # Check limits
        is_allowed = current_count <= config.requests_per_minute
        
        # Calculate reset time
        reset_time = int(window_start + config.window_size)
        
        rate_limit_info = {
            "limit": config.requests_per_minute,
            "remaining": max(0, config.requests_per_minute - current_count),
            "reset": reset_time,
            "window_size": config.window_size
        }
        
        return is_allowed, rate_limit_info

class SecurityManager:
    """
    Security manager for authentication and authorization
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def generate_jwt_token(self, user_id: str, roles: List[str] = None) -> str:
        """Generate JWT token"""
        payload = {
            "user_id": user_id,
            "roles": roles or [],
            "exp": datetime.utcnow() + timedelta(seconds=self.config.jwt_expiration),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        return request.headers.get(self.config.api_key_header)
    
    def check_security_level(self, request: Request, required_level: SecurityLevel) -> bool:
        """Check if request meets security requirements"""
        if required_level == SecurityLevel.PUBLIC:
            return True
        
        # Check for API key
        api_key = self.extract_api_key(request)
        if api_key:
            # Validate API key (implement your logic)
            return self._validate_api_key(api_key)
        
        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = self.verify_jwt_token(token)
                return self._check_authorization(payload, required_level)
            except HTTPException:
                return False
        
        return False
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key (implement your logic)"""
        # This is a placeholder - implement your API key validation
        return len(api_key) > 10
    
    def _check_authorization(self, payload: Dict[str, Any], required_level: SecurityLevel) -> bool:
        """Check user authorization level"""
        user_roles = payload.get("roles", [])
        
        if required_level == SecurityLevel.AUTHENTICATED:
            return True
        elif required_level == SecurityLevel.AUTHORIZED:
            return "user" in user_roles or "admin" in user_roles
        elif required_level == SecurityLevel.ADMIN:
            return "admin" in user_roles
        
        return False

class RequestTransformer:
    """
    Request/Response transformer
    """
    
    @staticmethod
    async def transform_request(request: Request, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform incoming request"""
        # Extract request data
        body = await request.body()
        headers = dict(request.headers)
        query_params = dict(request.query_params)
        
        # Apply transformations
        if "add_headers" in config:
            for key, value in config["add_headers"].items():
                headers[key] = value
        
        if "remove_headers" in config:
            for header in config["remove_headers"]:
                headers.pop(header, None)
        
        if "modify_query" in config:
            for key, value in config["modify_query"].items():
                query_params[key] = value
        
        return {
            "body": body,
            "headers": headers,
            "query_params": query_params,
            "method": request.method,
            "url": str(request.url)
        }
    
    @staticmethod
    def transform_response(response_data: Any, config: Dict[str, Any]) -> Any:
        """Transform outgoing response"""
        if not config:
            return response_data
        
        # Apply response transformations
        if isinstance(response_data, dict):
            if "add_fields" in config:
                response_data.update(config["add_fields"])
            
            if "remove_fields" in config:
                for field in config["remove_fields"]:
                    response_data.pop(field, None)
        
        return response_data

class APIGateway:
    """
    Advanced API Gateway with comprehensive features
    """
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        redis_url: str = "redis://localhost:6379",
        security_config: Optional[SecurityConfig] = None
    ):
        self.service_registry = service_registry
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        self.security_config = security_config or SecurityConfig(
            jwt_secret="your-secret-key",
            jwt_algorithm="HS256"
        )
        
        self.rate_limiter: Optional[RateLimiter] = None
        self.security_manager = SecurityManager(self.security_config)
        self.request_transformer = RequestTransformer()
        
        self.routes: Dict[str, RouteConfig] = {}
        self.circuit_breakers: Dict[str, HTTPCircuitBreaker] = {}
        
        self.app = FastAPI(
            title="API Gateway",
            description="Advanced API Gateway with rate limiting, security, and load balancing",
            version="1.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    async def start(self):
        """Start the API Gateway"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(self.redis_client)
            
            # Start service registry
            await self.service_registry.start()
            
            logger.info("API Gateway started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start API Gateway: {e}")
            raise
    
    async def stop(self):
        """Stop the API Gateway"""
        if self.redis_client:
            await self.redis_client.close()
        
        await self.service_registry.stop()
        
        # Close circuit breakers
        for breaker in self.circuit_breakers.values():
            if breaker.session:
                await breaker.session.close()
        
        logger.info("API Gateway stopped")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.security_config.allowed_origins,
            allow_credentials=True,
            allow_methods=self.security_config.allowed_methods,
            allow_headers=self.security_config.allowed_headers,
        )
        
        # Gzip middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            logger.info(
                f"Request: {request.method} {request.url}",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            )
            
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                f"Response: {response.status_code} ({duration:.3f}s)",
                extra={
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            return response
    
    def _setup_routes(self):
        """Setup API Gateway routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "api-gateway"}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "circuit_breakers": {
                    name: breaker.get_metrics()
                    for name, breaker in self.circuit_breakers.items()
                },
                "service_registry": await self.service_registry.get_service_metrics()
            }
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(request: Request):
            """Login endpoint"""
            try:
                body = await request.json()
                user_id = body.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id required")
                
                token = self.security_manager.generate_jwt_token(user_id)
                return {"access_token": token, "token_type": "bearer"}
                
            except Exception as e:
                logger.error(f"Login failed: {e}")
                raise HTTPException(status_code=401, detail="Login failed")
    
    def add_route(self, config: RouteConfig):
        """Add a new route to the gateway"""
        self.routes[config.path] = config
        
        # Create circuit breaker for the service
        if config.circuit_breaker_config:
            self.circuit_breakers[config.service_name] = HTTPCircuitBreaker(
                config.service_name,
                config.circuit_breaker_config
            )
        
        # Register route with FastAPI
        self._register_route(config)
    
    def _register_route(self, config: RouteConfig):
        """Register route with FastAPI"""
        
        async def route_handler(request: Request):
            return await self._handle_request(request, config)
        
        # Register route for each HTTP method
        for method in config.methods:
            self.app.add_api_route(
                config.path,
                route_handler,
                methods=[method],
                name=f"{method}_{config.path.replace('/', '_')}"
            )
    
    async def _handle_request(self, request: Request, config: RouteConfig) -> Response:
        """Handle incoming request"""
        try:
            # Security check
            if not self.security_manager.check_security_level(request, config.security_level):
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Rate limiting
            if config.rate_limit:
                client_ip = request.client.host if request.client else "unknown"
                rate_limit_key = f"{config.path}:{client_ip}"
                
                is_allowed, rate_info = await self.rate_limiter.is_allowed(
                    rate_limit_key, config.rate_limit
                )
                
                if not is_allowed:
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={
                            "X-RateLimit-Limit": str(rate_info["limit"]),
                            "X-RateLimit-Remaining": str(rate_info["remaining"]),
                            "X-RateLimit-Reset": str(rate_info["reset"])
                        }
                    )
            
            # Get service instance
            service_instance = await self.service_registry.get_service_instance(config.service_name)
            if not service_instance:
                raise HTTPException(status_code=503, detail="Service unavailable")
            
            # Transform request
            transformed_request = await self.request_transformer.transform_request(
                request, config.request_transform or {}
            )
            
            # Make request to service
            response = await self._forward_request(
                request, service_instance, transformed_request, config
            )
            
            # Transform response
            if config.response_transform:
                response_data = await response.json()
                transformed_data = self.request_transformer.transform_response(
                    response_data, config.response_transform
                )
                return JSONResponse(content=transformed_data)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _forward_request(
        self,
        original_request: Request,
        service_instance: ServiceInstance,
        transformed_request: Dict[str, Any],
        config: RouteConfig
    ) -> Response:
        """Forward request to service instance"""
        
        # Build target URL
        target_url = f"http://{service_instance.host}:{service_instance.port}{original_request.url.path}"
        if original_request.query_params:
            target_url += f"?{original_request.query_params}"
        
        # Get circuit breaker
        breaker = self.circuit_breakers.get(config.service_name)
        
        if breaker:
            # Use circuit breaker
            async with breaker:
                response = await breaker.request(
                    original_request.method,
                    target_url,
                    headers=transformed_request["headers"],
                    data=transformed_request["body"]
                )
        else:
            # Direct request
            async with aiohttp.ClientSession() as session:
                response = await session.request(
                    original_request.method,
                    target_url,
                    headers=transformed_request["headers"],
                    data=transformed_request["body"],
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                )
        
        # Convert aiohttp response to FastAPI response
        response_body = await response.read()
        
        return Response(
            content=response_body,
            status_code=response.status,
            headers=dict(response.headers)
        )

# Example usage and configuration
def create_api_gateway() -> APIGateway:
    """Create and configure API Gateway"""
    
    # Service registry
    service_registry = ServiceRegistry()
    
    # Security configuration
    security_config = SecurityConfig(
        jwt_secret="your-super-secret-key-change-in-production",
        jwt_algorithm="HS256",
        jwt_expiration=3600,
        rate_limit_by=RateLimitType.IP
    )
    
    # Create API Gateway
    gateway = APIGateway(service_registry, security_config=security_config)
    
    # Add routes
    gateway.add_route(RouteConfig(
        path="/api/users",
        service_name="user-service",
        methods=["GET", "POST"],
        security_level=SecurityLevel.AUTHENTICATED,
        rate_limit=RateLimitConfig(requests_per_minute=100),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0
        )
    ))
    
    gateway.add_route(RouteConfig(
        path="/api/videos",
        service_name="video-service",
        methods=["GET", "POST", "PUT", "DELETE"],
        security_level=SecurityLevel.AUTHENTICATED,
        rate_limit=RateLimitConfig(requests_per_minute=50),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0
        )
    ))
    
    return gateway






























