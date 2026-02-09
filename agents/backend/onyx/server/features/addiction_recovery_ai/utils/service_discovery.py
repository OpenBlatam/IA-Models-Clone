"""
Service Discovery for Recovery AI
"""

from typing import Dict, List, Optional, Any
import logging
import time

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Service registry for discovery"""
    
    def __init__(self):
        """Initialize service registry"""
        self.services: Dict[str, Dict[str, Any]] = {}
        logger.info("ServiceRegistry initialized")
    
    def register_service(
        self,
        service_name: str,
        service_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register service
        
        Args:
            service_name: Service name
            service_url: Service URL
            metadata: Optional metadata
        """
        self.services[service_name] = {
            "url": service_url,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "healthy": True
        }
        logger.info(f"Service registered: {service_name} -> {service_url}")
    
    def unregister_service(self, service_name: str):
        """Unregister service"""
        if service_name in self.services:
            del self.services[service_name]
            logger.info(f"Service unregistered: {service_name}")
    
    def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service information"""
        return self.services.get(service_name)
    
    def list_services(self) -> List[str]:
        """List all services"""
        return list(self.services.keys())
    
    def mark_unhealthy(self, service_name: str):
        """Mark service as unhealthy"""
        if service_name in self.services:
            self.services[service_name]["healthy"] = False
            logger.warning(f"Service marked unhealthy: {service_name}")
    
    def mark_healthy(self, service_name: str):
        """Mark service as healthy"""
        if service_name in self.services:
            self.services[service_name]["healthy"] = True
            logger.info(f"Service marked healthy: {service_name}")


class ServiceDiscovery:
    """Service discovery client"""
    
    def __init__(self, registry: ServiceRegistry):
        """
        Initialize service discovery
        
        Args:
            registry: Service registry
        """
        self.registry = registry
        logger.info("ServiceDiscovery initialized")
    
    def discover_service(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Discover service
        
        Args:
            service_name: Service name
            healthy_only: Only return healthy services
        
        Returns:
            Service information or None
        """
        service = self.registry.get_service(service_name)
        
        if not service:
            return None
        
        if healthy_only and not service.get("healthy", False):
            return None
        
        return service
    
    def discover_all_services(
        self,
        healthy_only: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Discover all services
        
        Args:
            healthy_only: Only return healthy services
        
        Returns:
            Dictionary of services
        """
        services = {}
        
        for name, service in self.registry.services.items():
            if not healthy_only or service.get("healthy", False):
                services[name] = service
        
        return services

