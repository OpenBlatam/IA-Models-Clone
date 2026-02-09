"""
Base Client for Piel Mejorador AI SAM3
======================================

Base class for HTTP clients with common functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import httpx
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseHTTPClient(ABC):
    """
    Base class for HTTP clients.
    
    Provides common functionality:
    - Connection pooling
    - Retry logic
    - Error handling
    - Resource management
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_connections: int = 100,
        max_keepalive: int = 20
    ):
        """
        Initialize base HTTP client.
        
        Args:
            base_url: Base API URL
            api_key: Optional API key
            timeout: Request timeout
            max_connections: Max connections in pool
            max_keepalive: Max keepalive connections
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive,
                        keepalive_expiry=30.0
                    )
                    timeout = httpx.Timeout(self.timeout, connect=10.0)
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get default headers.
        
        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        client = await self._get_client()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        headers = self._get_headers()
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            **kwargs
        )
        
        return response
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """GET request."""
        return await self._request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """POST request."""
        return await self._request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """PUT request."""
        return await self._request("PUT", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """DELETE request."""
        return await self._request("DELETE", endpoint, **kwargs)
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
            finally:
                self._client = None




