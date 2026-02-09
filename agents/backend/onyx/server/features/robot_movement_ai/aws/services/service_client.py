"""
Service Client
==============

HTTP client for inter-service communication with retries and circuit breakers.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
from aws.services.service_registry import get_service_registry, ServiceInstance

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_call(self) -> bool:
        """Check if call can be made."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if time_since_failure > self.recovery_timeout:
                    self.state = "half_open"
                    return True
            return False
        
        # half_open
        return True


class ServiceClient:
    """HTTP client for service-to-service communication."""
    
    def __init__(
        self,
        service_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.registry = get_service_registry()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_service_url(self, instance: Optional[ServiceInstance] = None) -> Optional[str]:
        """Get service URL."""
        if instance is None:
            instance = self.registry.get_instance(self.service_name)
        
        if instance is None:
            return None
        
        return f"http://{instance.host}:{instance.port}"
    
    async def _make_request(
        self,
        method: str,
        path: str,
        instance: Optional[ServiceInstance] = None,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retries."""
        if not self.circuit_breaker.can_call():
            raise Exception(f"Circuit breaker is open for {self.service_name}")
        
        base_url = self._get_service_url(instance)
        if not base_url:
            raise Exception(f"Service {self.service_name} not available")
        
        url = f"{base_url}{path}"
        client = await self._get_client()
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                
                if response.status_code < 500:
                    self.circuit_breaker.record_success()
                    return response
                else:
                    # Server error, retry
                    last_exception = Exception(f"Server error: {response.status_code}")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        self.circuit_breaker.record_failure()
        raise last_exception or Exception("Request failed after retries")
    
    async def get(self, path: str, **kwargs) -> Dict[str, Any]:
        """GET request."""
        response = await self._make_request("GET", path, **kwargs)
        return response.json()
    
    async def post(self, path: str, **kwargs) -> Dict[str, Any]:
        """POST request."""
        response = await self._make_request("POST", path, **kwargs)
        return response.json()
    
    async def put(self, path: str, **kwargs) -> Dict[str, Any]:
        """PUT request."""
        response = await self._make_request("PUT", path, **kwargs)
        return response.json()
    
    async def delete(self, path: str, **kwargs) -> Dict[str, Any]:
        """DELETE request."""
        response = await self._make_request("DELETE", path, **kwargs)
        return response.json()


class ServiceClientFactory:
    """Factory for creating service clients."""
    
    _clients: Dict[str, ServiceClient] = {}
    
    @classmethod
    def get_client(cls, service_name: str, **kwargs) -> ServiceClient:
        """Get or create service client."""
        if service_name not in cls._clients:
            cls._clients[service_name] = ServiceClient(service_name, **kwargs)
        return cls._clients[service_name]
    
    @classmethod
    async def close_all(cls):
        """Close all clients."""
        for client in cls._clients.values():
            await client.close()
        cls._clients.clear()















