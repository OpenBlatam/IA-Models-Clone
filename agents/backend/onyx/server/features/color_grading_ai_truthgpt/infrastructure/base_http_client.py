"""
Base HTTP Client for Color Grading AI
======================================

Unified base HTTP client with connection pooling, retry logic, and error handling.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class HTTPClientConfig:
    """HTTP client configuration."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 120.0
    connect_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    default_headers: Dict[str, str] = field(default_factory=dict)
    enable_http2: bool = True


class BaseHTTPClient:
    """
    Base HTTP client with unified functionality.
    
    Features:
    - Connection pooling
    - Retry logic
    - Error handling
    - Request/response logging
    - Timeout management
    - HTTP/2 support
    """
    
    def __init__(self, config: HTTPClientConfig):
        """
        Initialize base HTTP client.
        
        Args:
            config: HTTP client configuration
        """
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._request_count = 0
        self._error_count = 0
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                        keepalive_expiry=self.config.keepalive_expiry
                    )
                    timeout = httpx.Timeout(
                        self.config.timeout,
                        connect=self.config.connect_timeout
                    )
                    
                    headers = self.config.default_headers.copy()
                    if self.config.api_key:
                        headers.setdefault("Authorization", f"Bearer {self.config.api_key}")
                    
                    self._client = httpx.AsyncClient(
                        base_url=self.config.base_url,
                        limits=limits,
                        timeout=timeout,
                        http2=self.config.enable_http2,
                        headers=headers
                    )
                    logger.info(f"Initialized HTTP client for {self.config.base_url}")
        
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            try:
                await self._client.aclose()
                logger.info("HTTP client closed")
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
            finally:
                self._client = None
    
    async def request(
        self,
        method: HTTPMethod,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method
            path: Request path
            json: Optional JSON body
            params: Optional query parameters
            headers: Optional headers
            retry: Whether to retry on failure
            
        Returns:
            Response JSON
        """
        client = await self._get_client()
        
        request_headers = headers or {}
        
        async def _make_request():
            """Make the actual HTTP request."""
            response = await client.request(
                method.value,
                path,
                json=json,
                params=params,
                headers=request_headers
            )
            response.raise_for_status()
            return response.json()
        
        try:
            if retry:
                from .retry_helpers import retry_with_exponential_backoff
                data = await retry_with_exponential_backoff(
                    _make_request,
                    max_retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                    retryable_exceptions=(httpx.HTTPStatusError, httpx.TimeoutException),
                    operation_name=f"{method.value} {path}"
                )
            else:
                data = await _make_request()
            
            self._request_count += 1
            logger.debug(f"{method.value} {path} - Success")
            
            return data
        
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            self._error_count += 1
            logger.error(f"{method.value} {path} - Error: {e}")
            raise
    
    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request(HTTPMethod.GET, path, params=params, headers=headers, **kwargs)
    
    async def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request(HTTPMethod.POST, path, json=json, headers=headers, **kwargs)
    
    async def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request(HTTPMethod.PUT, path, json=json, headers=headers, **kwargs)
    
    async def delete(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request(HTTPMethod.DELETE, path, headers=headers, **kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "base_url": self.config.base_url,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._request_count - self._error_count) / self._request_count
                if self._request_count > 0 else 0.0
            ),
        }




