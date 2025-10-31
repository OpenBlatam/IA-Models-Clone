"""
Gamma App - API Gateway Service
Advanced API gateway with routing, load balancing, and rate limiting
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from datetime import datetime, timedelta
import hashlib
import hmac
from urllib.parse import urlparse, parse_qs
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import redis
from prometheus_client import Counter, Histogram, Gauge
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class RouteMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    RANDOM = "random"

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    weight: int = 1
    health_check_path: str = "/health"
    timeout: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0

@dataclass
class Route:
    """API route configuration"""
    path: str
    methods: List[RouteMethod]
    service_name: str
    rate_limit: Optional[Dict[str, Any]] = None
    authentication_required: bool = False
    authorization_roles: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3
    cache_ttl: int = 0
    transform_request: Optional[Callable] = None
    transform_response: Optional[Callable] = None

@dataclass
class RateLimit:
    """Rate limit configuration"""
    strategy: RateLimitStrategy
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size: int = 60
    key_extractor: Optional[Callable] = None

@dataclass
class CircuitBreaker:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

class APIGatewayService:
    """Advanced API Gateway service"""
    
    def __init__(self):
        self.services = {}
        self.routes = {}
        self.rate_limits = {}
        self.circuit_breakers = {}
        self.redis_client = None
        self.metrics = self._initialize_metrics()
        self._initialize_redis()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        return {
            'requests_total': Counter('api_gateway_requests_total', 'Total API requests', ['service', 'method', 'status']),
            'request_duration': Histogram('api_gateway_request_duration_seconds', 'Request duration', ['service', 'method']),
            'active_connections': Gauge('api_gateway_active_connections', 'Active connections', ['service']),
            'rate_limit_hits': Counter('api_gateway_rate_limit_hits_total', 'Rate limit hits', ['service', 'client']),
            'circuit_breaker_state': Gauge('api_gateway_circuit_breaker_state', 'Circuit breaker state', ['service'])
        }
    
    def _initialize_redis(self):
        """Initialize Redis client"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis for API Gateway")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
    
    def register_service(self, service: ServiceEndpoint):
        """Register a service endpoint"""
        try:
            self.services[service.name] = service
            self.circuit_breakers[service.name] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': None,
                'half_open_calls': 0
            }
            logger.info(f"Registered service: {service.name}")
        except Exception as e:
            logger.error(f"Error registering service: {e}")
            raise
    
    def register_route(self, route: Route):
        """Register an API route"""
        try:
            route_key = f"{route.path}:{':'.join([m.value for m in route.methods])}"
            self.routes[route_key] = route
            
            # Initialize rate limiter if configured
            if route.rate_limit:
                self.rate_limits[route_key] = RateLimit(**route.rate_limit)
            
            logger.info(f"Registered route: {route.path}")
        except Exception as e:
            logger.error(f"Error registering route: {e}")
            raise
    
    async def handle_request(self, request: Request) -> Response:
        """Handle incoming API request"""
        try:
            start_time = time.time()
            
            # Extract route information
            route_key = self._get_route_key(request)
            if route_key not in self.routes:
                raise HTTPException(status_code=404, detail="Route not found")
            
            route = self.routes[route_key]
            
            # Check rate limiting
            if await self._check_rate_limit(route_key, request):
                self.metrics['rate_limit_hits'].labels(
                    service=route.service_name,
                    client=self._get_client_ip(request)
                ).inc()
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Check authentication
            if route.authentication_required:
                await self._authenticate_request(request, route)
            
            # Check authorization
            if route.authorization_roles:
                await self._authorize_request(request, route)
            
            # Check circuit breaker
            if not await self._check_circuit_breaker(route.service_name):
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            # Select service endpoint
            service_endpoint = await self._select_service_endpoint(route.service_name)
            if not service_endpoint:
                raise HTTPException(status_code=503, detail="No healthy service endpoints available")
            
            # Transform request if needed
            transformed_request = await self._transform_request(request, route)
            
            # Forward request to service
            response = await self._forward_request(service_endpoint, transformed_request, route)
            
            # Transform response if needed
            transformed_response = await self._transform_response(response, route)
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics['requests_total'].labels(
                service=route.service_name,
                method=request.method,
                status=response.status_code
            ).inc()
            self.metrics['request_duration'].labels(
                service=route.service_name,
                method=request.method
            ).observe(duration)
            
            # Update circuit breaker
            await self._update_circuit_breaker(route.service_name, response.status_code < 500)
            
            return transformed_response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    def _get_route_key(self, request: Request) -> str:
        """Get route key from request"""
        return f"{request.url.path}:{request.method}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _check_rate_limit(self, route_key: str, request: Request) -> bool:
        """Check if request exceeds rate limit"""
        try:
            if route_key not in self.rate_limits:
                return False
            
            rate_limit = self.rate_limits[route_key]
            client_ip = self._get_client_ip(request)
            
            if rate_limit.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window_rate_limit(route_key, client_ip, rate_limit)
            elif rate_limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window_rate_limit(route_key, client_ip, rate_limit)
            elif rate_limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket_rate_limit(route_key, client_ip, rate_limit)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
    
    async def _check_fixed_window_rate_limit(self, route_key: str, client_ip: str, rate_limit: RateLimit) -> bool:
        """Check fixed window rate limit"""
        try:
            if not self.redis_client:
                return False
            
            current_window = int(time.time() // rate_limit.window_size)
            key = f"rate_limit:{route_key}:{client_ip}:{current_window}"
            
            current_requests = await self.redis_client.get(key)
            if current_requests is None:
                await self.redis_client.setex(key, rate_limit.window_size, 1)
                return False
            
            if int(current_requests) >= rate_limit.requests_per_minute:
                return True
            
            await self.redis_client.incr(key)
            return False
            
        except Exception as e:
            logger.error(f"Error checking fixed window rate limit: {e}")
            return False
    
    async def _check_sliding_window_rate_limit(self, route_key: str, client_ip: str, rate_limit: RateLimit) -> bool:
        """Check sliding window rate limit"""
        try:
            if not self.redis_client:
                return False
            
            current_time = time.time()
            window_start = current_time - rate_limit.window_size
            
            key = f"rate_limit:{route_key}:{client_ip}"
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_requests = await self.redis_client.zcard(key)
            
            if current_requests >= rate_limit.requests_per_minute:
                return True
            
            # Add current request
            await self.redis_client.zadd(key, {str(current_time): current_time})
            await self.redis_client.expire(key, rate_limit.window_size)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking sliding window rate limit: {e}")
            return False
    
    async def _check_token_bucket_rate_limit(self, route_key: str, client_ip: str, rate_limit: RateLimit) -> bool:
        """Check token bucket rate limit"""
        try:
            if not self.redis_client:
                return False
            
            key = f"token_bucket:{route_key}:{client_ip}"
            
            # Get current bucket state
            bucket_data = await self.redis_client.hmget(key, "tokens", "last_refill")
            
            if bucket_data[0] is None:
                # Initialize bucket
                tokens = rate_limit.burst_size
                last_refill = time.time()
            else:
                tokens = float(bucket_data[0])
                last_refill = float(bucket_data[1])
            
            # Refill tokens
            current_time = time.time()
            time_passed = current_time - last_refill
            tokens_to_add = time_passed * (rate_limit.requests_per_minute / 60)
            tokens = min(rate_limit.burst_size, tokens + tokens_to_add)
            
            if tokens < 1:
                return True
            
            # Consume token
            tokens -= 1
            
            # Update bucket state
            await self.redis_client.hmset(key, {
                "tokens": tokens,
                "last_refill": current_time
            })
            await self.redis_client.expire(key, rate_limit.window_size)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking token bucket rate limit: {e}")
            return False
    
    async def _authenticate_request(self, request: Request, route: Route):
        """Authenticate request"""
        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            
            token = auth_header.split(" ")[1]
            
            # Verify JWT token
            try:
                payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
                request.state.user_id = payload.get("user_id")
                request.state.user_roles = payload.get("roles", [])
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail="Token expired")
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating request: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _authorize_request(self, request: Request, route: Route):
        """Authorize request"""
        try:
            user_roles = getattr(request.state, 'user_roles', [])
            
            if not any(role in user_roles for role in route.authorization_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authorizing request: {e}")
            raise HTTPException(status_code=403, detail="Authorization failed")
    
    async def _check_circuit_breaker(self, service_name: str) -> bool:
        """Check circuit breaker state"""
        try:
            if service_name not in self.circuit_breakers:
                return True
            
            circuit_breaker = self.circuit_breakers[service_name]
            
            if circuit_breaker['state'] == 'open':
                # Check if recovery timeout has passed
                if (circuit_breaker['last_failure_time'] and 
                    time.time() - circuit_breaker['last_failure_time'] > 60):
                    circuit_breaker['state'] = 'half_open'
                    circuit_breaker['half_open_calls'] = 0
                    return True
                return False
            
            elif circuit_breaker['state'] == 'half_open':
                if circuit_breaker['half_open_calls'] >= 3:
                    return False
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            return True
    
    async def _select_service_endpoint(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Select service endpoint using load balancing"""
        try:
            if service_name not in self.services:
                return None
            
            service = self.services[service_name]
            
            # Simple round-robin for now
            # In a real implementation, you'd implement different strategies
            if service.is_healthy:
                return service
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting service endpoint: {e}")
            return None
    
    async def _transform_request(self, request: Request, route: Route) -> Request:
        """Transform request if needed"""
        try:
            if route.transform_request:
                return await route.transform_request(request)
            return request
        except Exception as e:
            logger.error(f"Error transforming request: {e}")
            return request
    
    async def _forward_request(self, service_endpoint: ServiceEndpoint, request: Request, route: Route) -> Response:
        """Forward request to service endpoint"""
        try:
            # Prepare request data
            url = f"{service_endpoint.url}{request.url.path}"
            if request.url.query:
                url += f"?{request.url.query}"
            
            headers = dict(request.headers)
            headers.pop("host", None)  # Remove host header
            
            # Forward request
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    data=await request.body(),
                    timeout=aiohttp.ClientTimeout(total=route.timeout)
                ) as response:
                    response_body = await response.read()
                    
                    return Response(
                        content=response_body,
                        status_code=response.status,
                        headers=dict(response.headers)
                    )
                    
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Gateway timeout")
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            raise HTTPException(status_code=502, detail="Bad gateway")
    
    async def _transform_response(self, response: Response, route: Route) -> Response:
        """Transform response if needed"""
        try:
            if route.transform_response:
                return await route.transform_response(response)
            return response
        except Exception as e:
            logger.error(f"Error transforming response: {e}")
            return response
    
    async def _update_circuit_breaker(self, service_name: str, success: bool):
        """Update circuit breaker state"""
        try:
            if service_name not in self.circuit_breakers:
                return
            
            circuit_breaker = self.circuit_breakers[service_name]
            
            if success:
                if circuit_breaker['state'] == 'half_open':
                    circuit_breaker['half_open_calls'] += 1
                    if circuit_breaker['half_open_calls'] >= 3:
                        circuit_breaker['state'] = 'closed'
                        circuit_breaker['failure_count'] = 0
                else:
                    circuit_breaker['failure_count'] = 0
            else:
                circuit_breaker['failure_count'] += 1
                circuit_breaker['last_failure_time'] = time.time()
                
                if circuit_breaker['failure_count'] >= 5:
                    circuit_breaker['state'] = 'open'
            
            # Update metrics
            state_value = {'closed': 0, 'half_open': 1, 'open': 2}[circuit_breaker['state']]
            self.metrics['circuit_breaker_state'].labels(service=service_name).set(state_value)
            
        except Exception as e:
            logger.error(f"Error updating circuit breaker: {e}")
    
    async def health_check_service(self, service_name: str) -> bool:
        """Perform health check on service"""
        try:
            if service_name not in self.services:
                return False
            
            service = self.services[service_name]
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{service.url}{service.health_check_path}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    is_healthy = response.status == 200
                    service.is_healthy = is_healthy
                    service.last_health_check = datetime.now()
                    
                    return is_healthy
                    
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            if service_name in self.services:
                self.services[service_name].is_healthy = False
            return False
    
    async def health_check_all_services(self):
        """Perform health check on all services"""
        try:
            tasks = []
            for service_name in self.services:
                tasks.append(self.health_check_service(service_name))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error checking all services health: {e}")
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        try:
            stats = {
                'total_services': len(self.services),
                'healthy_services': len([s for s in self.services.values() if s.is_healthy]),
                'unhealthy_services': len([s for s in self.services.values() if not s.is_healthy]),
                'total_routes': len(self.routes),
                'total_rate_limits': len(self.rate_limits),
                'services': {}
            }
            
            for service_name, service in self.services.items():
                stats['services'][service_name] = {
                    'url': service.url,
                    'is_healthy': service.is_healthy,
                    'active_connections': service.active_connections,
                    'total_requests': service.total_requests,
                    'failed_requests': service.failed_requests,
                    'last_health_check': service.last_health_check.isoformat() if service.last_health_check else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting service statistics: {e}")
            return {}
    
    def export_configuration(self, output_path: str) -> bool:
        """Export API Gateway configuration"""
        try:
            config = {
                'services': [
                    {
                        'name': service.name,
                        'url': service.url,
                        'weight': service.weight,
                        'health_check_path': service.health_check_path,
                        'timeout': service.timeout,
                        'max_retries': service.max_retries,
                        'circuit_breaker_threshold': service.circuit_breaker_threshold,
                        'circuit_breaker_timeout': service.circuit_breaker_timeout
                    }
                    for service in self.services.values()
                ],
                'routes': [
                    {
                        'path': route.path,
                        'methods': [method.value for method in route.methods],
                        'service_name': route.service_name,
                        'rate_limit': route.rate_limit,
                        'authentication_required': route.authentication_required,
                        'authorization_roles': route.authorization_roles,
                        'timeout': route.timeout,
                        'retry_count': route.retry_count,
                        'cache_ttl': route.cache_ttl
                    }
                    for route in self.routes.values()
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"API Gateway configuration exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False

# Global API Gateway service instance
api_gateway_service = APIGatewayService()

def register_service_endpoint(service: ServiceEndpoint):
    """Register service endpoint using global service"""
    api_gateway_service.register_service(service)

def register_api_route(route: Route):
    """Register API route using global service"""
    api_gateway_service.register_route(route)

async def handle_gateway_request(request: Request) -> Response:
    """Handle gateway request using global service"""
    return await api_gateway_service.handle_request(request)

async def health_check_all_services():
    """Health check all services using global service"""
    await api_gateway_service.health_check_all_services()

def get_gateway_statistics() -> Dict[str, Any]:
    """Get gateway statistics using global service"""
    return api_gateway_service.get_service_statistics()

def export_gateway_configuration(output_path: str) -> bool:
    """Export gateway configuration using global service"""
    return api_gateway_service.export_configuration(output_path)
























