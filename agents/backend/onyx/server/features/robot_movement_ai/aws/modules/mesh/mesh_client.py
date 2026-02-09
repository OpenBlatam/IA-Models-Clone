"""
Mesh Client
==========

Service mesh client for inter-service communication.
"""

import logging
import httpx
import asyncio
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MeshClient:
    """Service mesh client."""
    
    def __init__(
        self,
        service_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_enabled: bool = True
    ):
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._optimizer = AsyncOptimizer()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client
    
    async def call_service(
        self,
        service_url: str,
        method: str = "GET",
        path: str = "",
        **kwargs
    ) -> httpx.Response:
        """Call service through mesh."""
        client = await self._get_client()
        
        url = f"{service_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Apply retry logic
        max_retries = self.max_retries
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.error(f"Service call failed after {max_retries} attempts: {service_url} - {e}")
                raise
        
        raise last_exception
    
    async def discover_service(self, service_name: str) -> Optional[str]:
        """Discover service URL."""
        # In production, use service discovery (Consul, Eureka, etc.)
        # For now, return None
        return None
    
    async def close(self):
        """Close client."""
        if self._client:
            await self._client.aclose()
            self._client = None

