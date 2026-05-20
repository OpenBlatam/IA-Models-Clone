"""
Service Mesh Patterns
Advanced service mesh patterns for microservices communication
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from microservices.service_discovery import ServiceRegistry, get_service_registry
from microservices.service_client import ServiceClient, get_service_client
from aws.circuit_breaker import CircuitBreaker
from aws.retry_handler import retry_with_backoff

logger = logging.getLogger(__name__)


class ServiceMesh:
    """
    Service mesh for microservices
    
    Features:
    - Service-to-service communication
    - Automatic retry and circuit breaking
    - Request tracing
    - Timeout management
    - Health-aware routing
    """
    
    def __init__(self):
        self.registry = get_service_registry()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._timeouts: Dict[str, float] = {}
        self._retry_configs: Dict[str, Dict[str, Any]] = {}
    
    def configure_service(
        self,
        service_name: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5
    ) -> None:
        """Configure service mesh settings for a service"""
        self._timeouts[service_name] = timeout
        self._retry_configs[service_name] = {
            "max_retries": max_retries,
            "backoff_factor": 2.0
        }
        
        self._circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout=60
        )
        
        logger.info(f"Configured service mesh for: {service_name}")
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for service"""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker()
        return self._circuit_breakers[service_name]
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call service through service mesh
        
        Args:
            service_name: Target service name
            method: HTTP method
            path: API path
            **kwargs: Request parameters
            
        Returns:
            Response data
        """
        circuit_breaker = self.get_circuit_breaker(service_name)
        timeout = self._timeouts.get(service_name, 5.0)
        retry_config = self._retry_configs.get(service_name, {"max_retries": 3})
        
        client = get_service_client(service_name)
        
        @retry_with_backoff(**retry_config)
        async def _call():
            with circuit_breaker:
                if method.upper() == "GET":
                    return await client.get(path, **kwargs)
                elif method.upper() == "POST":
                    return await client.post(path, **kwargs)
                elif method.upper() == "PUT":
                    return await client.put(path, **kwargs)
                elif method.upper() == "DELETE":
                    return await client.delete(path, **kwargs)
                else:
                    response = await client.request(method, path, **kwargs)
                    return response.json()
        
        try:
            return await _call()
        except Exception as e:
            logger.error(f"Service mesh call failed: {service_name} - {str(e)}")
            raise
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get service health through mesh"""
        info = self.registry.get_service_info(service_name)
        circuit_breaker = self.get_circuit_breaker(service_name)
        
        return {
            **info,
            "circuit_breaker_state": circuit_breaker.get_state().value,
            "timeout": self._timeouts.get(service_name, 5.0)
        }


class RequestQueue:
    """Request queue for throttling and rate limiting"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self._queues: Dict[str, list] = {}
    
    def enqueue(self, service_name: str, request: Dict[str, Any]) -> bool:
        """Enqueue request"""
        if service_name not in self._queues:
            self._queues[service_name] = []
        
        if len(self._queues[service_name]) >= self.max_queue_size:
            return False
        
        self._queues[service_name].append(request)
        return True
    
    def dequeue(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Dequeue request"""
        if service_name not in self._queues or not self._queues[service_name]:
            return None
        
        return self._queues[service_name].pop(0)
    
    def queue_size(self, service_name: str) -> int:
        """Get queue size"""
        return len(self._queues.get(service_name, []))


# Global service mesh
_mesh: Optional[ServiceMesh] = None


def get_service_mesh() -> ServiceMesh:
    """Get global service mesh"""
    global _mesh
    if _mesh is None:
        _mesh = ServiceMesh()
    return _mesh















