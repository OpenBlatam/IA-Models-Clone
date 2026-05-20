"""
HTTP Client Manager
===================

Manages HTTP client for ComfyUI API with connection pooling.
"""

import logging
import asyncio
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class HTTPClientManager:
    """Manages HTTP client with connection pooling."""
    
    def __init__(
        self,
        max_connections: int = 50,
        max_keepalive: int = 10,
        keepalive_expiry: float = 30.0,
        timeout: float = 300.0,
        connect_timeout: float = 10.0
    ):
        """
        Initialize HTTP Client Manager.
        
        Args:
            max_connections: Maximum number of connections
            max_keepalive: Maximum keepalive connections
            keepalive_expiry: Keepalive expiry time
            timeout: Request timeout
            connect_timeout: Connection timeout
        """
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.keepalive_expiry = keepalive_expiry
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def get_client(self) -> httpx.AsyncClient:
        """
        Get or create HTTP client with connection pooling.
        
        Uses double-checked locking pattern for thread-safe initialization.
        
        Returns:
            httpx.AsyncClient instance with connection pooling
        """
        if self._http_client is None:
            async with self._client_lock:
                if self._http_client is None:
                    limits = httpx.Limits(
                        max_connections=self.max_connections,
                        max_keepalive_connections=self.max_keepalive,
                        keepalive_expiry=self.keepalive_expiry
                    )
                    timeout = httpx.Timeout(
                        self.timeout,
                        connect=self.connect_timeout
                    )
                    self._http_client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
                    logger.debug("ComfyUI HTTP client initialized")
        return self._http_client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception as e:
                logger.warning(f"Error closing ComfyUI HTTP client: {e}")
            finally:
                self._http_client = None

