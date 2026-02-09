"""
Service Mesh

Utilities for service mesh networking.
"""

import logging
from typing import Dict, Any, Optional, Callable
from core.service_discovery import ServiceRegistry
from core.load_balancer import RoundRobinBalancer
from core.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class ServiceMesh:
    """Service mesh for inter-service communication."""
    
    def __init__(self):
        """Initialize service mesh."""
        self.registry = ServiceRegistry()
        self.balancers: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def register_service(
        self,
        service_name: str,
        address: str,
        port: int,
        **kwargs
    ) -> None:
        """
        Register service in mesh.
        
        Args:
            service_name: Service name
            address: Service address
            port: Service port
            **kwargs: Additional metadata
        """
        self.registry.register(service_name, address, port, kwargs)
        
        # Create load balancer for service
        if service_name not in self.balancers:
            services = self.registry.discover(service_name)
            servers = [f"{s['address']}:{s['port']}" for s in services]
            self.balancers[service_name] = RoundRobinBalancer(servers)
        
        # Create circuit breaker for service
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0
            )
        
        logger.info(f"Registered service in mesh: {service_name}")
    
    def call_service(
        self,
        service_name: str,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Call service through mesh.
        
        Args:
            service_name: Service name
            endpoint: Service endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            Service response
        """
        # Get service via load balancer
        balancer = self.balancers.get(service_name)
        if not balancer:
            services = self.registry.discover(service_name)
            if not services:
                raise ValueError(f"Service not found: {service_name}")
            
            servers = [f"{s['address']}:{s['port']}" for s in services]
            balancer = RoundRobinBalancer(servers)
            self.balancers[service_name] = balancer
        
        server = balancer.select_server()
        
        # Call with circuit breaker
        breaker = self.circuit_breakers.get(service_name)
        
        def make_request():
            # Make HTTP request to service
            import requests
            url = f"http://{server}/{endpoint}"
            response = requests.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        
        if breaker:
            return breaker.call(make_request)
        else:
            return make_request()
    
    def get_service_health(
        self,
        service_name: str
    ) -> Dict[str, Any]:
        """
        Get service health status.
        
        Args:
            service_name: Service name
            
        Returns:
            Health status
        """
        services = self.registry.discover(service_name)
        breaker = self.circuit_breakers.get(service_name)
        
        return {
            'service': service_name,
            'instances': len(services),
            'circuit_state': breaker.get_state().value if breaker else None,
            'services': services
        }


def create_mesh() -> ServiceMesh:
    """Create service mesh."""
    return ServiceMesh()


def register_service(
    mesh: ServiceMesh,
    service_name: str,
    address: str,
    port: int,
    **kwargs
) -> None:
    """Register service in mesh."""
    mesh.register_service(service_name, address, port, **kwargs)


def call_service(
    mesh: ServiceMesh,
    service_name: str,
    endpoint: str,
    **kwargs
) -> Any:
    """Call service through mesh."""
    return mesh.call_service(service_name, endpoint, **kwargs)



