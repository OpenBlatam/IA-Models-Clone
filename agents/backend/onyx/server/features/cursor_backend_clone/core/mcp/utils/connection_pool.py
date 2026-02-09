"""
MCP Connection Pool - Pool de conexiones HTTP
==============================================

Pool de conexiones HTTP reutilizables para mejorar rendimiento.
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

try:
    import httpx
    _has_httpx = True
except ImportError:
    _has_httpx = False
    httpx = None


class HTTPConnectionPool:
    """Pool de conexiones HTTP reutilizables"""
    
    def __init__(
        self,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        timeout: float = 10.0
    ):
        if not _has_httpx:
            self._client = None
            logger.warning("httpx not available, connection pool disabled")
            return
        
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections
        )
        
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(timeout),
            http2=True
        )
        self._closed = False
    
    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs):
        """Hacer request usando el pool"""
        if not self._client or self._closed:
            raise RuntimeError("Connection pool is closed")
        
        try:
            response = await self._client.request(method, url, **kwargs)
            yield response
        finally:
            pass
    
    async def get(self, url: str, **kwargs):
        """GET request"""
        async with self.request("GET", url, **kwargs) as response:
            return response
    
    async def post(self, url: str, **kwargs):
        """POST request"""
        async with self.request("POST", url, **kwargs) as response:
            return response
    
    async def close(self):
        """Cerrar el pool"""
        if self._client and not self._closed:
            await self._client.aclose()
            self._closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

