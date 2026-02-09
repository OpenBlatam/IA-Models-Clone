"""
Client Base
===========

Base classes for API clients with common functionality.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Client configuration."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True


class BaseAPIClient(ABC):
    """Base class for API clients."""
    
    def __init__(self, config: ClientConfig):
        """
        Initialize API client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration": 0.0
        }
    
    async def initialize(self):
        """Initialize HTTP client."""
        if self.client is None:
            headers = self.config.headers.copy()
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers=headers,
                verify=self.config.verify_ssl
            )
            logger.info(f"Initialized {self.__class__.__name__}")
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info(f"Closed {self.__class__.__name__}")
    
    async def request(
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
            **kwargs: Request arguments
            
        Returns:
            HTTP response
        """
        if self.client is None:
            await self.initialize()
        
        import time
        start = time.time()
        
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            self.stats["total_requests"] += 1
            
            if response.is_success:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
            
            duration = time.time() - start
            self.stats["total_duration"] += duration
            
            return response
        except Exception as e:
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            logger.error(f"Request failed: {e}")
            raise
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self.request("PUT", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self.request("DELETE", endpoint, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_duration = (
            self.stats["total_duration"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0.0
        )
        
        return {
            **self.stats,
            "avg_duration": avg_duration,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            )
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class RetryableClient(BaseAPIClient):
    """Client with automatic retry logic."""
    
    async def request_with_retry(
        self,
        method: str,
        endpoint: str,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> httpx.Response:
        """
        Make request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            max_retries: Maximum retry attempts
            **kwargs: Request arguments
            
        Returns:
            HTTP response
        """
        max_retries = max_retries or self.config.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.request(method, endpoint, **kwargs)
                
                # Retry on server errors
                if response.status_code >= 500 and attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
        
        if last_exception:
            raise last_exception




