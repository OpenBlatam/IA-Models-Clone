"""
Service Registry
================

Registry for enhancement services.
"""

import logging
from typing import Dict, Any, Optional, List, Type, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EnhancementService(ABC):
    """Base class for enhancement services."""
    
    def __init__(self, name: str):
        """
        Initialize service.
        
        Args:
            name: Service name
        """
        self.name = name
    
    @abstractmethod
    async def process(
        self,
        file_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process enhancement request.
        
        Args:
            file_path: Path to file
            options: Enhancement options
            
        Returns:
            Enhancement result
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get service information.
        
        Returns:
            Service info dictionary
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }


class ServiceRegistry:
    """Registry for enhancement services."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, EnhancementService] = {}
        self.factories: Dict[str, Callable[[], EnhancementService]] = {}
    
    def register(
        self,
        name: str,
        service: Optional[EnhancementService] = None,
        factory: Optional[Callable[[], EnhancementService]] = None
    ):
        """
        Register a service.
        
        Args:
            name: Service name
            service: Service instance
            factory: Service factory function
        """
        if service:
            self.services[name] = service
        elif factory:
            self.factories[name] = factory
        else:
            raise ValueError("Either service or factory must be provided")
        
        logger.info(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[EnhancementService]:
        """
        Get service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None
        """
        if name in self.services:
            return self.services[name]
        
        if name in self.factories:
            service = self.factories[name]()
            self.services[name] = service
            return service
        
        return None
    
    def get_all(self) -> Dict[str, EnhancementService]:
        """Get all services."""
        # Ensure all factories are instantiated
        for name, factory in self.factories.items():
            if name not in self.services:
                self.services[name] = factory()
        
        return self.services.copy()
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        all_names = set(self.services.keys()) | set(self.factories.keys())
        return sorted(all_names)
    
    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get service information.
        
        Args:
            name: Service name
            
        Returns:
            Service info or None
        """
        service = self.get(name)
        if service:
            return service.get_info()
        return None
    
    def unregister(self, name: str):
        """
        Unregister a service.
        
        Args:
            name: Service name
        """
        self.services.pop(name, None)
        self.factories.pop(name, None)
        logger.info(f"Unregistered service: {name}")




