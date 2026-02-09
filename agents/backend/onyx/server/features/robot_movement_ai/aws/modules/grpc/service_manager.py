"""
gRPC Service Manager
====================

gRPC service management.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GRPCService:
    """gRPC service definition."""
    name: str
    methods: Dict[str, Callable] = None
    port: int = 50051
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = {}


class GRPCServiceManager:
    """gRPC service manager."""
    
    def __init__(self):
        self._services: Dict[str, GRPCService] = {}
        self._running_services: List[str] = []
    
    def register_service(
        self,
        name: str,
        methods: Dict[str, Callable],
        port: int = 50051,
        description: Optional[str] = None
    ) -> GRPCService:
        """Register gRPC service."""
        service = GRPCService(
            name=name,
            methods=methods,
            port=port,
            description=description
        )
        
        self._services[name] = service
        logger.info(f"Registered gRPC service: {name} on port {port}")
        return service
    
    def get_service(self, name: str) -> Optional[GRPCService]:
        """Get service by name."""
        return self._services.get(name)
    
    def register_method(self, service_name: str, method_name: str, handler: Callable):
        """Register method for service."""
        if service_name not in self._services:
            self._services[service_name] = GRPCService(name=service_name)
        
        self._services[service_name].methods[method_name] = handler
        logger.info(f"Registered method {method_name} for service {service_name}")
    
    async def call_method(self, service_name: str, method_name: str, *args, **kwargs) -> Any:
        """Call gRPC method."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service {service_name} not found")
        
        if method_name not in service.methods:
            raise ValueError(f"Method {method_name} not found in service {service_name}")
        
        handler = service.methods[method_name]
        
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        else:
            return handler(*args, **kwargs)
    
    def list_services(self) -> List[GRPCService]:
        """List all services."""
        return list(self._services.values())
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_services": len(self._services),
            "running_services": len(self._running_services),
            "services": {
                name: {
                    "methods": len(service.methods),
                    "port": service.port
                }
                for name, service in self._services.items()
            }
        }















