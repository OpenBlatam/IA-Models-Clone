"""
Service Discovery

Utilities for service discovery and registration.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Registry for service discovery."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        service_name: str,
        address: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register service.
        
        Args:
            service_name: Service name
            address: Service address
            port: Service port
            metadata: Optional metadata
        """
        service_id = f"{service_name}_{address}_{port}"
        
        self.services[service_id] = {
            'name': service_name,
            'address': address,
            'port': port,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'last_heartbeat': datetime.now().isoformat()
        }
        
        logger.info(f"Registered service: {service_name} at {address}:{port}")
    
    def discover(
        self,
        service_name: str
    ) -> List[Dict[str, Any]]:
        """
        Discover services by name.
        
        Args:
            service_name: Service name
            
        Returns:
            List of service information
        """
        services = [
            service for service in self.services.values()
            if service['name'] == service_name
        ]
        
        return services
    
    def get_service(
        self,
        service_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get service by ID.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Service information or None
        """
        return self.services.get(service_id)
    
    def heartbeat(
        self,
        service_id: str
    ) -> None:
        """
        Update service heartbeat.
        
        Args:
            service_id: Service identifier
        """
        if service_id in self.services:
            self.services[service_id]['last_heartbeat'] = datetime.now().isoformat()
    
    def unregister(self, service_id: str) -> None:
        """
        Unregister service.
        
        Args:
            service_id: Service identifier
        """
        if service_id in self.services:
            del self.services[service_id]
            logger.info(f"Unregistered service: {service_id}")
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services."""
        return list(self.services.values())


# Global registry
_global_registry = ServiceRegistry()


def register_service(
    service_name: str,
    address: str,
    port: int,
    **kwargs
) -> None:
    """Register service in global registry."""
    _global_registry.register(service_name, address, port, **kwargs)


def discover_service(service_name: str) -> List[Dict[str, Any]]:
    """Discover services by name."""
    return _global_registry.discover(service_name)


def list_services() -> List[Dict[str, Any]]:
    """List all services."""
    return _global_registry.list_services()



