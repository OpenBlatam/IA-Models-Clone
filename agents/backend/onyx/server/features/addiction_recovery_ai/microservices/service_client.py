"""
Service Client
HTTP client for inter-service communication with circuit breaker and retry
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx
from aws.circuit_breaker import CircuitBreaker, CircuitBreakerError
from aws.retry_handler import retry_with_backoff
from microservices.service_discovery import ServiceRegistry, get_service_registry

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    HTTP client for inter-service communication
    
    Features:
    - Service discovery integration
    - Circuit breaker
    - Retry with exponential backoff
    - Load balancing
    - Timeout handling
    - Request/response logging
    """
    
    def __init__(
        self,
        service_name: str,
        registry: Optional[ServiceRegistry] = None,
        timeout: float = 5.0,
        max_retries: int = 3
    ):
        self.service_name = service_name
        self.registry = registry or get_service_registry()
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Circuit breaker per service
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60
        )
        
        # HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    async def request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make request to service
        
        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        # Get service instance
        instance = self.registry.get_instance(self.service_name)
        
        if not instance:
            raise ValueError(f"Service {self.service_name} not available")
        
        url = f"{instance.url}{path}"
        
        try:
            with self.circuit_breaker:
                response = await self.client.request(method, url, **kwargs)
                
                # Update heartbeat on successful request
                self.registry.heartbeat(self.service_name, instance.instance_id)
                
                # Raise for status
                response.raise_for_status()
                
                return response
                
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker open for {self.service_name}: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {self.service_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Request failed for {self.service_name}: {str(e)}")
            raise
    
    async def get(self, path: str, **kwargs) -> Dict[str, Any]:
        """GET request"""
        response = await self.request("GET", path, **kwargs)
        return response.json()
    
    async def post(self, path: str, **kwargs) -> Dict[str, Any]:
        """POST request"""
        response = await self.request("POST", path, **kwargs)
        return response.json()
    
    async def put(self, path: str, **kwargs) -> Dict[str, Any]:
        """PUT request"""
        response = await self.request("PUT", path, **kwargs)
        return response.json()
    
    async def delete(self, path: str, **kwargs) -> Dict[str, Any]:
        """DELETE request"""
        response = await self.request("DELETE", path, **kwargs)
        return response.json()
    
    async def patch(self, path: str, **kwargs) -> Dict[str, Any]:
        """PATCH request"""
        response = await self.request("PATCH", path, **kwargs)
        return response.json()


class ServiceClientPool:
    """Pool of service clients for multiple services"""
    
    def __init__(self):
        self._clients: Dict[str, ServiceClient] = {}
    
    def get_client(self, service_name: str) -> ServiceClient:
        """Get or create service client"""
        if service_name not in self._clients:
            self._clients[service_name] = ServiceClient(service_name)
        return self._clients[service_name]
    
    async def close_all(self) -> None:
        """Close all clients"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()


# Global client pool
_client_pool: Optional[ServiceClientPool] = None


def get_service_client(service_name: str) -> ServiceClient:
    """Get service client from pool"""
    global _client_pool
    if _client_pool is None:
        _client_pool = ServiceClientPool()
    return _client_pool.get_client(service_name)















