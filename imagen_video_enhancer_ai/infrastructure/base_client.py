"""
Base Client
===========

Base class for HTTP API clients with common functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import httpx
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseHTTPClient(ABC):
    """Base class for HTTP API clients."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        keepalive_expiry: float = 30.0
    ):
        """
        Initialize base HTTP client.
        
        Args:
            base_url: Base API URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
            connect_timeout: Connection timeout in seconds
            max_connections: Maximum connections in pool
            max_keepalive_connections: Maximum keepalive connections
            keepalive_expiry: Keepalive expiry in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        # Connection pool settings
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive_connections,
                        keepalive_expiry=self.keepalive_expiry
                    )
                    timeout = httpx.Timeout(
                        self.timeout,
                        connect=self.connect_timeout
                    )
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get default headers for requests.
        
        Returns:
            Headers dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"{self.__class__.__name__}/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            headers: Additional headers
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        client = await self._get_client()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)
        
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                **kwargs
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make GET request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response JSON
        """
        response = await self._request("GET", endpoint, **kwargs)
        return response.json()
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional request parameters
            
        Returns:
            Response JSON
        """
        response = await self._request("POST", endpoint, json=data, **kwargs)
        return response.json()
    
    async def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional request parameters
            
        Returns:
            Response JSON
        """
        response = await self._request("PUT", endpoint, json=data, **kwargs)
        return response.json()
    
    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make DELETE request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response JSON
        """
        response = await self._request("DELETE", endpoint, **kwargs)
        return response.json()
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing {self.__class__.__name__} client: {e}")
            finally:
                self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()




