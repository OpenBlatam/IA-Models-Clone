"""
HTTP Client Manager
===================
Manages HTTP client lifecycle and connection pooling for ComfyUI
"""

import asyncio
import logging
import httpx
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HTTPClientConfig:
    """HTTP client configuration"""
    max_connections: int = 50
    max_keepalive: int = 10
    keepalive_expiry: float = 30.0
    timeout: float = 300.0
    connect_timeout: float = 10.0


class HTTPClientManager:
    """
    Manages HTTP client lifecycle with connection pooling.
    
    Uses double-checked locking pattern for thread-safe initialization.
    """
    
    def __init__(self, config: HTTPClientConfig):
        """
        Initialize HTTP client manager.
        
        Args:
            config: HTTP client configuration
        """
        self.config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def get_client(self) -> httpx.AsyncClient:
        """
        Get or create HTTP client with connection pooling.
        
        Returns:
            httpx.AsyncClient instance with connection pooling
        """
        if self._http_client is None:
            async with self._client_lock:
                if self._http_client is None:
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive,
                        keepalive_expiry=self.config.keepalive_expiry
                    )
                    timeout = httpx.Timeout(
                        self.config.timeout,
                        connect=self.config.connect_timeout
                    )
                    self._http_client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
                    logger.debug("HTTP client initialized")
        return self._http_client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
            finally:
                self._http_client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

