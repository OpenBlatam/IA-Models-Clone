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
import aiohttp
import aiofiles
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, TypeVar, Generic
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import signal
import os
import hashlib
import pickle
import ssl
import certifi
from fastapi import Request, Response, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Async External API Operations for HeyGen AI FastAPI
Dedicated async functions for external API operations with connection pooling and optimization.
"""



logger = structlog.get_logger()

# =============================================================================
# External API Types
# =============================================================================

class APIMethod(Enum):
    """HTTP method enumeration."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

class APIType(Enum):
    """API type enumeration."""
    REST = "rest"
    GRAPHQL = "graphql"
    SOAP = "soap"
    GRPC = "grpc"
    WEBSOCKET = "websocket"

class ConnectionState(Enum):
    """Connection state enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"

@dataclass
class APIConfig:
    """External API configuration."""
    base_url: str
    api_type: APIType = APIType.REST
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_connections: int = 100
    max_connections_per_host: int = 10
    connection_timeout: float = 10.0
    enable_ssl_verification: bool = True
    enable_compression: bool = True
    enable_keepalive: bool = True
    keepalive_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_rate_limiting: bool = True
    rate_limit: int = 100  # requests per second
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    default_headers: Optional[Dict[str, str]] = None
    auth_token: Optional[str] = None
    api_key: Optional[str] = None

@dataclass
class APIMetrics:
    """API performance metrics."""
    api_id: str
    method: APIMethod
    endpoint: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = False
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    cache_hit: bool = False
    response_size_bytes: int = 0
    request_size_bytes: int = 0

# =============================================================================
# Async External API Manager
# =============================================================================

class AsyncExternalAPIManager:
    """Main async external API manager with connection pooling and optimization."""
    
    def __init__(self, config: APIConfig):
        
    """__init__ function."""
self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_pool = None
        self.api_metrics: Dict[str, APIMetrics] = {}
        self.rate_limit_semaphore = asyncio.Semaphore(config.rate_limit)
        self.circuit_breaker_state = "closed"  # closed, open, half-open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self._lock = asyncio.Lock()
        self._is_initialized = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> Any:
        """Initialize the external API manager."""
        if self._is_initialized:
            return
        
        try:
            # Create SSL context
            ssl_context = None
            if self.config.enable_ssl_verification:
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            else:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Create connector
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=ssl_context,
                keepalive_timeout=self.config.keepalive_timeout,
                enable_cleanup_closed=True
            )
            
            # Create timeout
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=self.config.connection_timeout
            )
            
            # Create session
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.config.default_headers or {},
                trust_env=True
            )
            
            # Test connection
            await self._test_connection()
            
            self._is_initialized = True
            
            # Start monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"External API manager initialized for {self.config.base_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize external API manager: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup the external API manager."""
        if not self._is_initialized:
            return
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close session
        if self.session:
            await self.session.close()
        
        self._is_initialized = False
        logger.info("External API manager cleaned up")
    
    async def _test_connection(self) -> Any:
        """Test API connection."""
        try:
            async with self.session.get(f"{self.config.base_url}/health", timeout=5.0) as response:
                if response.status >= 400:
                    raise Exception(f"API health check failed: {response.status}")
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
    
    async def _monitoring_loop(self) -> Any:
        """API monitoring loop."""
        while self._is_initialized:
            try:
                # Monitor circuit breaker state
                if self.circuit_breaker_state == "open":
                    if (self.circuit_breaker_last_failure and 
                        time.time() - self.circuit_breaker_last_failure > self.config.circuit_breaker_timeout):
                        self.circuit_breaker_state = "half-open"
                        logger.info("Circuit breaker moved to half-open state")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"API monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def get_api_metrics(self) -> Dict[str, APIMetrics]:
        """Get API performance metrics."""
        return self.api_metrics.copy()
    
    def get_circuit_breaker_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.circuit_breaker_state,
            "failures": self.circuit_breaker_failures,
            "last_failure": self.circuit_breaker_last_failure,
            "threshold": self.config.circuit_breaker_threshold,
            "timeout": self.config.circuit_breaker_timeout
        }

# =============================================================================
# Async External API Operations
# =============================================================================

class AsyncExternalAPIOperations:
    """Dedicated async functions for external API operations."""
    
    def __init__(self, api_manager: AsyncExternalAPIManager):
        
    """__init__ function."""
self.api_manager = api_manager
        self.response_cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async async def make_request(
        self,
        method: APIMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        retry_on_failure: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request to external API."""
        api_id = self._generate_api_id(method, endpoint, data, params)
        
        # Check cache for GET requests
        if method == APIMethod.GET and cache_key and self.api_manager.config.enable_caching:
            cached_response = await self._get_cached_response(cache_key, cache_ttl)
            if cached_response is not None:
                await self._record_api_metrics(api_id, method, endpoint, cache_hit=True)
                return cached_response
        
        # Check circuit breaker
        if not await self._check_circuit_breaker():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Apply rate limiting
        if self.api_manager.config.enable_rate_limiting:
            await self.api_manager.rate_limit_semaphore.acquire()
            try:
                await asyncio.sleep(1.0 / self.api_manager.config.rate_limit)
            finally:
                self.api_manager.rate_limit_semaphore.release()
        
        # Prepare request
        url = f"{self.api_manager.config.base_url}{endpoint}"
        request_headers = self._prepare_headers(headers)
        request_data = self._prepare_data(data, method)
        
        # Execute request with retry logic
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.api_manager.config.max_retries + 1):
            try:
                response = await self._execute_request(
                    method, url, request_data, params, request_headers, timeout
                )
                
                # Record success
                duration_ms = (time.time() - start_time) * 1000
                await self._record_api_metrics(
                    api_id, method, endpoint,
                    duration_ms=duration_ms, success=True,
                    status_code=response.get("status_code"),
                    response_size_bytes=len(json.dumps(response.get("data", "")))
                )
                
                # Reset circuit breaker on success
                if self.api_manager.circuit_breaker_state == "half-open":
                    self.api_manager.circuit_breaker_state = "closed"
                    self.api_manager.circuit_breaker_failures = 0
                
                # Cache response for GET requests
                if method == APIMethod.GET and cache_key and self.api_manager.config.enable_caching:
                    await self._cache_response(cache_key, response, cache_ttl)
                
                return response
                
            except Exception as e:
                last_exception = e
                
                # Record failure
                duration_ms = (time.time() - start_time) * 1000
                await self._record_api_metrics(
                    api_id, method, endpoint,
                    duration_ms=duration_ms, success=False,
                    error_message=str(e), retry_count=attempt
                )
                
                # Update circuit breaker
                await self._update_circuit_breaker()
                
                # Retry logic
                if attempt < self.api_manager.config.max_retries and retry_on_failure:
                    delay = self.api_manager.config.retry_delay * (self.api_manager.config.retry_backoff ** attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
        
        # All retries failed
        logger.error(f"API request failed after {self.api_manager.config.max_retries + 1} attempts: {last_exception}")
        raise HTTPException(status_code=500, detail=f"External API error: {last_exception}")
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.make_request(
            APIMethod.GET, endpoint, params=params, headers=headers,
            timeout=timeout, cache_key=cache_key, cache_ttl=cache_ttl
        )
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.make_request(
            APIMethod.POST, endpoint, data=data, params=params,
            headers=headers, timeout=timeout
        )
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.make_request(
            APIMethod.PUT, endpoint, data=data, params=params,
            headers=headers, timeout=timeout
        )
    
    async def patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make PATCH request."""
        return await self.make_request(
            APIMethod.PATCH, endpoint, data=data, params=params,
            headers=headers, timeout=timeout
        )
    
    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.make_request(
            APIMethod.DELETE, endpoint, params=params,
            headers=headers, timeout=timeout
        )
    
    async async def make_batch_requests(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Make multiple API requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async async def make_single_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.make_request(**request_data)
        
        return await asyncio.gather(
            *[make_single_request(req) for req in requests],
            return_exceptions=True
        )
    
    async def stream_request(
        self,
        method: APIMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ):
        """Stream API response."""
        # Check circuit breaker
        if not await self._check_circuit_breaker():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Prepare request
        url = f"{self.api_manager.config.base_url}{endpoint}"
        request_headers = self._prepare_headers(headers)
        request_data = self._prepare_data(data, method)
        
        try:
            if method == APIMethod.GET:
                async with self.api_manager.session.get(
                    url, params=params, headers=request_headers, timeout=timeout
                ) as response:
                    async for chunk in response.content.iter_chunked(8192):
                        yield chunk
            else:
                async with self.api_manager.session.post(
                    url, json=request_data, params=params, headers=request_headers, timeout=timeout
                ) as response:
                    async for chunk in response.content.iter_chunked(8192):
                        yield chunk
                        
        except Exception as e:
            logger.error(f"API stream error: {e}")
            raise HTTPException(status_code=500, detail=f"External API stream error: {e}")
    
    async async def upload_file(
        self,
        endpoint: str,
        file_path: str,
        file_field: str = "file",
        additional_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Upload file to external API."""
        # Check circuit breaker
        if not await self._check_circuit_breaker():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Prepare request
        url = f"{self.api_manager.config.base_url}{endpoint}"
        request_headers = self._prepare_headers(headers)
        
        try:
            # Read file
            async with aiofiles.open(file_path, 'rb') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                file_data = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field(file_field, file_data, filename=os.path.basename(file_path))
            
            if additional_data:
                for key, value in additional_data.items():
                    data.add_field(key, str(value))
            
            # Make request
            async with self.api_manager.session.post(
                url, data=data, headers=request_headers, timeout=timeout
            ) as response:
                response_data = await response.json() if response.content_type == "application/json" else await response.text()
                
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "data": response_data
                }
                
        except Exception as e:
            logger.error(f"API file upload error: {e}")
            raise HTTPException(status_code=500, detail=f"External API upload error: {e}")
    
    async async def download_file(
        self,
        endpoint: str,
        save_path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Download file from external API."""
        # Check circuit breaker
        if not await self._check_circuit_breaker():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        
        # Prepare request
        url = f"{self.api_manager.config.base_url}{endpoint}"
        request_headers = self._prepare_headers(headers)
        
        try:
            async with self.api_manager.session.get(
                url, params=params, headers=request_headers, timeout=timeout
            ) as response:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save file
                async with aiofiles.open(save_path, 'wb') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    async for chunk in response.content.iter_chunked(8192):
                        await file.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "file_path": save_path,
                    "file_size": os.path.getsize(save_path)
                }
                
        except Exception as e:
            logger.error(f"API file download error: {e}")
            raise HTTPException(status_code=500, detail=f"External API download error: {e}")
    
    async async def _execute_request(
        self,
        method: APIMethod,
        url: str,
        data: Optional[Any],
        params: Optional[Dict[str, Any]],
        headers: Dict[str, str],
        timeout: Optional[float]
    ) -> Dict[str, Any]:
        """Execute HTTP request."""
        if method == APIMethod.GET:
            async with self.api_manager.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                return await self._process_response(response)
        elif method == APIMethod.POST:
            async with self.api_manager.session.post(url, json=data, params=params, headers=headers, timeout=timeout) as response:
                return await self._process_response(response)
        elif method == APIMethod.PUT:
            async with self.api_manager.session.put(url, json=data, params=params, headers=headers, timeout=timeout) as response:
                return await self._process_response(response)
        elif method == APIMethod.PATCH:
            async with self.api_manager.session.patch(url, json=data, params=params, headers=headers, timeout=timeout) as response:
                return await self._process_response(response)
        elif method == APIMethod.DELETE:
            async with self.api_manager.session.delete(url, params=params, headers=headers, timeout=timeout) as response:
                return await self._process_response(response)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Process API response."""
        try:
            if response.content_type == "application/json":
                data = await response.json()
            else:
                data = await response.text()
            
            return {
                "status_code": response.status,
                "headers": dict(response.headers),
                "data": data
            }
        except Exception as e:
            logger.error(f"Response processing error: {e}")
            raise
    
    def _prepare_headers(self, headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Prepare request headers."""
        request_headers = {}
        
        # Add default headers
        if self.api_manager.config.default_headers:
            request_headers.update(self.api_manager.config.default_headers)
        
        # Add authentication
        if self.api_manager.config.auth_token:
            request_headers["Authorization"] = f"Bearer {self.api_manager.config.auth_token}"
        
        if self.api_manager.config.api_key:
            request_headers["X-API-Key"] = self.api_manager.config.api_key
        
        # Add custom headers
        if headers:
            request_headers.update(headers)
        
        return request_headers
    
    def _prepare_data(self, data: Optional[Dict[str, Any]], method: APIMethod) -> Optional[Any]:
        """Prepare request data."""
        if data is None:
            return None
        
        # For non-GET requests, return JSON data
        if method != APIMethod.GET:
            return data
        
        return None
    
    async def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state."""
        if self.api_manager.circuit_breaker_state == "open":
            return False
        return True
    
    async def _update_circuit_breaker(self) -> Any:
        """Update circuit breaker state."""
        async with self.api_manager._lock:
            self.api_manager.circuit_breaker_failures += 1
            self.api_manager.circuit_breaker_last_failure = time.time()
            
            if (self.api_manager.circuit_breaker_failures >= 
                self.api_manager.config.circuit_breaker_threshold):
                self.api_manager.circuit_breaker_state = "open"
                logger.warning("Circuit breaker opened")
    
    async def _generate_api_id(self, method: APIMethod, endpoint: str, data: Optional[Dict[str, Any]], params: Optional[Dict[str, Any]]) -> str:
        """Generate unique API request ID."""
        request_str = f"{method.value}:{endpoint}:{json.dumps(data or {}, sort_keys=True)}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(request_str.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str, ttl: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached API response."""
        async with self._lock:
            if cache_key in self.response_cache:
                cached_data = self.response_cache[cache_key]
                cache_ttl = ttl or self.api_manager.config.cache_ttl
                
                if time.time() - cached_data["timestamp"] < cache_ttl:
                    return cached_data["data"]
                else:
                    # Remove expired cache entry
                    del self.response_cache[cache_key]
        
        return None
    
    async def _cache_response(self, cache_key: str, response: Dict[str, Any], ttl: Optional[int] = None):
        """Cache API response."""
        async with self._lock:
            self.response_cache[cache_key] = {
                "data": response,
                "timestamp": time.time()
            }
    
    async def _record_api_metrics(
        self,
        api_id: str,
        method: APIMethod,
        endpoint: str,
        duration_ms: Optional[float] = None,
        success: bool = False,
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        cache_hit: bool = False,
        response_size_bytes: int = 0,
        request_size_bytes: int = 0
    ):
        """Record API performance metrics."""
        metrics = APIMetrics(
            api_id=api_id,
            method=method,
            endpoint=endpoint,
            end_time=datetime.now(timezone.utc) if duration_ms else None,
            duration_ms=duration_ms,
            success=success,
            status_code=status_code,
            error_message=error_message,
            retry_count=retry_count,
            cache_hit=cache_hit,
            response_size_bytes=response_size_bytes,
            request_size_bytes=request_size_bytes
        )
        
        self.api_manager.api_metrics[api_id] = metrics
    
    async def get_api_metrics(self) -> Dict[str, APIMetrics]:
        """Get API performance metrics."""
        return self.api_manager.api_metrics.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get API cache statistics."""
        async with self._lock:
            return {
                "cache_size": len(self.response_cache),
                "cache_ttl": self.api_manager.config.cache_ttl
            }

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "APIMethod",
    "APIType",
    "ConnectionState",
    "APIConfig",
    "APIMetrics",
    "AsyncExternalAPIManager",
    "AsyncExternalAPIOperations",
] 