from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
import functools
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic, Awaitable, Type
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from pathlib import Path
import weakref
import contextlib
from urllib.parse import urljoin, urlparse
import structlog
from pydantic import BaseModel, Field
import numpy as np
import aiohttp
import httpx
import websockets
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
            import base64
from typing import Any, List, Dict, Optional
"""
🌐 Async API Client
==================

Comprehensive async API client module with:
- Dedicated async functions for all external API operations
- HTTP client management with connection pooling
- Authentication and authorization handling
- Rate limiting and circuit breaker patterns
- Request/response caching
- Error handling and retry logic
- Performance monitoring and metrics
- Multiple HTTP client backends
- Async request batching
- WebSocket support
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')
ResponseT = TypeVar('ResponseT')

class HTTPMethod(Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class ClientType(Enum):
    """HTTP client types"""
    AIOHTTP = "aiohttp"
    HTTPX = "httpx"
    WEBSOCKET = "websocket"

class AuthType(Enum):
    """Authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"

@dataclass
class APIConfig:
    """API configuration"""
    base_url: str
    client_type: ClientType = ClientType.AIOHTTP
    auth_type: AuthType = AuthType.NONE
    api_key: str = None
    bearer_token: str = None
    username: str = None
    password: str = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 100  # requests per minute
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_metrics: bool = True
    connection_pool_size: int = 20
    max_connections: int = 100
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

@dataclass
class APIRequest:
    """API request configuration"""
    method: HTTPMethod
    endpoint: str
    params: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = None
    cache_key: str = None
    retry_attempts: int = None

@dataclass
class APIResponse:
    """API response wrapper"""
    status_code: int
    data: Any
    headers: Dict[str, str]
    elapsed_time: float
    cache_hit: bool = False
    retry_count: int = 0
    error: str = None

@dataclass
class APIMetrics:
    """API performance metrics"""
    endpoint: str
    method: HTTPMethod
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    cache_hit_rate: float
    last_updated: datetime

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker logic"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, max_requests: int, time_window: float = 60.0):
        
    """__init__ function."""
self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire rate limit permit"""
        async with self.lock:
            now = time.time()
            
            # Remove expired requests
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_for_permit(self) -> Any:
        """Wait until a permit is available"""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class AsyncAPIClient:
    """Main async API client"""
    
    def __init__(self, config: APIConfig):
        
    """__init__ function."""
self.config = config
        self.session = None
        self.httpx_client = None
        self.redis_client = None
        
        # Circuit breaker
        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                config.circuit_breaker_threshold,
                config.circuit_breaker_timeout
            )
        else:
            self.circuit_breaker = None
        
        # Rate limiter
        self.rate_limiter = RateLimiter(config.rate_limit)
        
        # Performance tracking
        self.metrics: Dict[str, APIMetrics] = defaultdict(
            lambda: APIMetrics(
                endpoint="",
                method=HTTPMethod.GET,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0,
                min_response_time=float('inf'),
                max_response_time=0.0,
                p95_response_time=0.0,
                cache_hit_rate=0.0,
                last_updated=datetime.now()
            )
        )
        
        # Request history for metrics
        self.request_history: deque = deque(maxlen=10000)
        
        logger.info(f"Async API Client initialized for {config.base_url}")
    
    async def initialize(self) -> Any:
        """Initialize API client"""
        try:
            # Initialize HTTP client based on type
            if self.config.client_type == ClientType.AIOHTTP:
                await self._initialize_aiohttp()
            elif self.config.client_type == ClientType.HTTPX:
                await self._initialize_httpx()
            
            # Initialize caching
            if self.config.enable_caching:
                await self._initialize_caching()
            
            logger.info("API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise
    
    async async def _initialize_aiohttp(self) -> Any:
        """Initialize aiohttp client"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.connection_pool_size,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_default_headers()
        )
    
    async async def _initialize_httpx(self) -> Any:
        """Initialize httpx client"""
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.connection_pool_size
        )
        
        self.httpx_client = httpx.AsyncClient(
            limits=limits,
            timeout=self.config.timeout,
            headers=self._get_default_headers()
        )
    
    async def _initialize_caching(self) -> Any:
        """Initialize caching system"""
        self.redis_client = redis.from_url("redis://localhost:6379")
        await self.redis_client.ping()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            "User-Agent": "AsyncAPIClient/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Add authentication headers
        if self.config.auth_type == AuthType.API_KEY and self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        elif self.config.auth_type == AuthType.BEARER_TOKEN and self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self.config.auth_type == AuthType.BASIC_AUTH and self.config.username and self.config.password:
            credentials = base64.b64encode(
                f"{self.config.username}:{self.config.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    async async def request(self, api_request: APIRequest) -> APIResponse:
        """Make API request"""
        start_time = time.time()
        
        # Wait for rate limit permit
        await self.rate_limiter.wait_for_permit()
        
        # Check cache first
        if api_request.cache_key and self.config.enable_caching:
            cached_response = await self._get_from_cache(api_request.cache_key)
            if cached_response:
                return APIResponse(
                    status_code=200,
                    data=cached_response,
                    headers={},
                    elapsed_time=time.time() - start_time,
                    cache_hit=True
                )
        
        # Prepare request
        url = urljoin(self.config.base_url, api_request.endpoint)
        headers = {**self._get_default_headers(), **api_request.headers}
        timeout = api_request.timeout or self.config.timeout
        retry_attempts = api_request.retry_attempts or self.config.max_retries
        
        # Execute request with retries
        last_error = None
        for attempt in range(retry_attempts):
            try:
                if self.circuit_breaker:
                    response = await self.circuit_breaker.call(
                        self._make_request,
                        api_request.method,
                        url,
                        api_request.params,
                        api_request.data,
                        headers,
                        timeout
                    )
                else:
                    response = await self._make_request(
                        api_request.method,
                        url,
                        api_request.params,
                        api_request.data,
                        headers,
                        timeout
                    )
                
                # Record metrics
                self._record_metrics(api_request.endpoint, api_request.method, 
                                   time.time() - start_time, True)
                
                # Cache response if needed
                if api_request.cache_key and self.config.enable_caching:
                    await self._set_cache(api_request.cache_key, response.data)
                
                return APIResponse(
                    status_code=response.status_code,
                    data=response.data,
                    headers=dict(response.headers),
                    elapsed_time=time.time() - start_time,
                    retry_count=attempt
                )
                
            except Exception as e:
                last_error = e
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        # Record failed metrics
        self._record_metrics(api_request.endpoint, api_request.method, 
                           time.time() - start_time, False)
        
        return APIResponse(
            status_code=0,
            data=None,
            headers={},
            elapsed_time=time.time() - start_time,
            retry_count=retry_attempts,
            error=str(last_error)
        )
    
    async def _make_request(self, method: HTTPMethod, url: str, params: Dict[str, Any],
                           data: Any, headers: Dict[str, str], timeout: float):
        """Make HTTP request using configured client"""
        if self.config.client_type == ClientType.AIOHTTP:
            return await self._aiohttp_request(method, url, params, data, headers, timeout)
        elif self.config.client_type == ClientType.HTTPX:
            return await self._httpx_request(method, url, params, data, headers, timeout)
        else:
            raise ValueError(f"Unsupported client type: {self.config.client_type}")
    
    async def _aiohttp_request(self, method: HTTPMethod, url: str, params: Dict[str, Any],
                              data: Any, headers: Dict[str, str], timeout: float):
        """Make request using aiohttp"""
        if not self.session:
            raise RuntimeError("aiohttp session not initialized")
        
        # Prepare request data
        if data and isinstance(data, (dict, list)):
            json_data = data
            data = None
        else:
            json_data = None
        
        async with self.session.request(
            method.value,
            url,
            params=params,
            json=json_data,
            data=data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            response_data = await response.json() if response.content_type == "application/json" else await response.text()
            
            return type('Response', (), {
                'status_code': response.status,
                'data': response_data,
                'headers': response.headers
            })()
    
    async def _httpx_request(self, method: HTTPMethod, url: str, params: Dict[str, Any],
                            data: Any, headers: Dict[str, str], timeout: float):
        """Make request using httpx"""
        if not self.httpx_client:
            raise RuntimeError("httpx client not initialized")
        
        # Prepare request data
        if data and isinstance(data, (dict, list)):
            json_data = data
            data = None
        else:
            json_data = None
        
        response = await self.httpx_client.request(
            method.value,
            url,
            params=params,
            json=json_data,
            data=data,
            headers=headers,
            timeout=timeout
        )
        
        response_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
        
        return type('Response', (), {
            'status_code': response.status_code,
            'data': response_data,
            'headers': dict(response.headers)
        })()
    
    # Convenience methods for common HTTP methods
    
    async def get(self, endpoint: str, params: Dict[str, Any] = None, 
                  headers: Dict[str, str] = None, cache_key: str = None) -> APIResponse:
        """GET request"""
        request = APIRequest(
            method=HTTPMethod.GET,
            endpoint=endpoint,
            params=params or {},
            headers=headers or {},
            cache_key=cache_key
        )
        return await self.request(request)
    
    async def post(self, endpoint: str, data: Any = None, params: Dict[str, Any] = None,
                   headers: Dict[str, str] = None) -> APIResponse:
        """POST request"""
        request = APIRequest(
            method=HTTPMethod.POST,
            endpoint=endpoint,
            params=params or {},
            data=data,
            headers=headers or {}
        )
        return await self.request(request)
    
    async def put(self, endpoint: str, data: Any = None, params: Dict[str, Any] = None,
                  headers: Dict[str, str] = None) -> APIResponse:
        """PUT request"""
        request = APIRequest(
            method=HTTPMethod.PUT,
            endpoint=endpoint,
            params=params or {},
            data=data,
            headers=headers or {}
        )
        return await self.request(request)
    
    async def delete(self, endpoint: str, params: Dict[str, Any] = None,
                     headers: Dict[str, str] = None) -> APIResponse:
        """DELETE request"""
        request = APIRequest(
            method=HTTPMethod.DELETE,
            endpoint=endpoint,
            params=params or {},
            headers=headers or {}
        )
        return await self.request(request)
    
    async def patch(self, endpoint: str, data: Any = None, params: Dict[str, Any] = None,
                    headers: Dict[str, str] = None) -> APIResponse:
        """PATCH request"""
        request = APIRequest(
            method=HTTPMethod.PATCH,
            endpoint=endpoint,
            params=params or {},
            data=data,
            headers=headers or {}
        )
        return await self.request(request)
    
    # Batch operations
    
    async async def batch_request(self, requests: List[APIRequest], 
                           max_concurrent: int = 10) -> List[APIResponse]:
        """Execute multiple requests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async async def execute_request(request: APIRequest) -> APIResponse:
            async with semaphore:
                return await self.request(request)
        
        tasks = [execute_request(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # WebSocket support
    
    async def websocket_connect(self, endpoint: str, headers: Dict[str, str] = None) -> websockets.WebSocketServerProtocol:
        """Connect to WebSocket endpoint"""
        url = urljoin(self.config.base_url, endpoint)
        return await websockets.connect(url, extra_headers=headers or {})
    
    async def websocket_send(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Send message through WebSocket"""
        await websocket.send(message)
    
    async def websocket_receive(self, websocket: websockets.WebSocketServerProtocol) -> str:
        """Receive message from WebSocket"""
        return await websocket.recv()
    
    # Caching operations
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def _set_cache(self, key: str, value: Any, ttl: int = None):
        """Set value in cache"""
        if not self.redis_client:
            return
        
        try:
            ttl = ttl or self.config.cache_ttl
            serialized = json.dumps(value)
            await self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate_cache(self, pattern: str):
        """Invalidate cache by pattern"""
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    # Metrics and monitoring
    
    def _record_metrics(self, endpoint: str, method: HTTPMethod, 
                       response_time: float, success: bool):
        """Record API metrics"""
        key = f"{endpoint}_{method.value}"
        metrics = self.metrics[key]
        
        metrics.endpoint = endpoint
        metrics.method = method
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update response time statistics
        response_times = [m.response_time for m in self.request_history if m.endpoint == endpoint]
        response_times.append(response_time)
        
        metrics.avg_response_time = np.mean(response_times)
        metrics.min_response_time = min(metrics.min_response_time, response_time)
        metrics.max_response_time = max(metrics.max_response_time, response_time)
        metrics.p95_response_time = np.percentile(response_times, 95)
        
        # Update cache hit rate
        cache_hits = sum(1 for m in self.request_history if m.endpoint == endpoint and m.cache_hit)
        total_requests = sum(1 for m in self.request_history if m.endpoint == endpoint)
        metrics.cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0
        
        metrics.last_updated = datetime.now()
        
        # Store in history
        self.request_history.append(type('RequestMetric', (), {
            'endpoint': endpoint,
            'response_time': response_time,
            'cache_hit': False,
            'timestamp': datetime.now()
        })())
    
    def get_metrics(self, endpoint: str = None) -> Dict[str, Any]:
        """Get API metrics"""
        if endpoint:
            key = f"{endpoint}_GET"  # Default to GET method
            metrics = self.metrics.get(key)
            if metrics:
                return {
                    "endpoint": metrics.endpoint,
                    "method": metrics.method.value,
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "success_rate": metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0.0,
                    "avg_response_time": metrics.avg_response_time,
                    "p95_response_time": metrics.p95_response_time,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "last_updated": metrics.last_updated.isoformat()
                }
            return {"message": f"No metrics for endpoint: {endpoint}"}
        
        # Return all metrics
        return {
            endpoint: {
                "total_requests": metrics.total_requests,
                "success_rate": metrics.successful_requests / metrics.total_requests if metrics.total_requests > 0 else 0.0,
                "avg_response_time": metrics.avg_response_time,
                "cache_hit_rate": metrics.cache_hit_rate
            }
            for endpoint, metrics in self.metrics.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        # Calculate overall statistics
        all_response_times = []
        all_requests = 0
        all_successful = 0
        
        for metrics in self.metrics.values():
            all_requests += metrics.total_requests
            all_successful += metrics.successful_requests
            # Add response times from history
            response_times = [m.response_time for m in self.request_history 
                            if m.endpoint == metrics.endpoint]
            all_response_times.extend(response_times)
        
        # Find slowest endpoints
        slowest_endpoints = sorted(
            self.metrics.items(),
            key=lambda x: x[1].avg_response_time,
            reverse=True
        )[:5]
        
        # Find endpoints with most errors
        error_endpoints = sorted(
            self.metrics.items(),
            key=lambda x: x[1].failed_requests,
            reverse=True
        )[:5]
        
        return {
            "overall": {
                "total_requests": all_requests,
                "successful_requests": all_successful,
                "success_rate": all_successful / all_requests if all_requests > 0 else 0.0,
                "avg_response_time": np.mean(all_response_times) if all_response_times else 0.0,
                "p95_response_time": np.percentile(all_response_times, 95) if all_response_times else 0.0
            },
            "slowest_endpoints": [
                {
                    "endpoint": endpoint,
                    "avg_response_time": metrics.avg_response_time,
                    "total_requests": metrics.total_requests
                }
                for endpoint, metrics in slowest_endpoints
            ],
            "endpoints_with_most_errors": [
                {
                    "endpoint": endpoint,
                    "failed_requests": metrics.failed_requests,
                    "total_requests": metrics.total_requests
                }
                for endpoint, metrics in error_endpoints
            ],
            "circuit_breaker_status": {
                "state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "failure_count": self.circuit_breaker.failure_count if self.circuit_breaker else 0
            },
            "rate_limiter_status": {
                "current_requests": len(self.rate_limiter.requests),
                "max_requests": self.rate_limiter.max_requests
            }
        }
    
    async def cleanup(self) -> Any:
        """Cleanup API client resources"""
        try:
            if self.session:
                await self.session.close()
            
            if self.httpx_client:
                await self.httpx_client.aclose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("API client cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Decorators for API operations

def async_api_request(method: HTTPMethod, endpoint: str, cache_key: str = None, 
                     retry_attempts: int = 3, timeout: float = 30.0):
    """Decorator for async API requests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(api_client: AsyncAPIClient, *args, **kwargs):
            
    """wrapper function."""
# Extract parameters from function arguments
            params = {}
            data = None
            headers = {}
            
            # Map function arguments to request parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(api_client, *args, **kwargs)
            bound_args.apply_defaults()
            
            for param_name, param_value in bound_args.arguments.items():
                if param_name == 'api_client':
                    continue
                elif param_name in ['data', 'body', 'payload']:
                    data = param_value
                elif param_name == 'headers':
                    headers = param_value
                else:
                    params[param_name] = param_value
            
            # Create API request
            request = APIRequest(
                method=method,
                endpoint=endpoint,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout,
                cache_key=cache_key,
                retry_attempts=retry_attempts
            )
            
            # Execute request
            response = await api_client.request(request)
            
            # Handle response
            if response.error:
                raise HTTPException(
                    status_code=response.status_code or 500,
                    detail=f"API request failed: {response.error}"
                )
            
            return response.data
            
        return wrapper
    return decorator

# Example usage

async def example_api_operations():
    """Example usage of async API operations"""
    
    # Create API config
    config = APIConfig(
        base_url="https://api.example.com",
        client_type=ClientType.AIOHTTP,
        auth_type=AuthType.API_KEY,
        api_key="your-api-key",
        timeout=30.0,
        max_retries=3,
        rate_limit=100,
        enable_caching=True
    )
    
    # Create API client
    api_client = AsyncAPIClient(config)
    await api_client.initialize()
    
    try:
        # GET request
        response = await api_client.get("/users", {"page": 1}, cache_key="users_page_1")
        print(f"GET response: {response.data}")
        
        # POST request
        user_data = {"name": "John Doe", "email": "john@example.com"}
        response = await api_client.post("/users", user_data)
        print(f"POST response: {response.data}")
        
        # PUT request
        update_data = {"name": "John Smith"}
        response = await api_client.put("/users/1", update_data)
        print(f"PUT response: {response.data}")
        
        # DELETE request
        response = await api_client.delete("/users/1")
        print(f"DELETE response: {response.data}")
        
        # Batch requests
        requests = [
            APIRequest(HTTPMethod.GET, "/users/1", cache_key="user_1"),
            APIRequest(HTTPMethod.GET, "/users/2", cache_key="user_2"),
            APIRequest(HTTPMethod.GET, "/users/3", cache_key="user_3")
        ]
        responses = await api_client.batch_request(requests)
        print(f"Batch responses: {len(responses)}")
        
        # Get metrics
        metrics = api_client.get_metrics()
        print("API metrics:", metrics)
        
        # Get performance summary
        summary = api_client.get_performance_summary()
        print("Performance summary:", summary)
        
    finally:
        await api_client.cleanup()

match __name__:
    case "__main__":
    asyncio.run(example_api_operations()) 