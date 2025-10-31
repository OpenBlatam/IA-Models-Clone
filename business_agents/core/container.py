"""
Service Container
=================

Dependency injection container for managing services and dependencies.
"""

from typing import Dict, Any, TypeVar, Type, Optional, Callable
import logging
from functools import lru_cache

from ..business_agents import BusinessAgentManager
from ..services import (
    HealthService, SystemInfoService, MetricsService,
    AgentService, WorkflowService, DocumentService
)

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ServiceContainer:
    """Dependency injection container for managing services."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._initialized = False
    
    def register_singleton(self, service_type: str, instance: Any) -> None:
        """Register a singleton service instance."""
        self._singletons[service_type] = instance
        logger.debug(f"Registered singleton: {service_type}")
    
    def register_factory(self, service_type: str, factory: Callable) -> None:
        """Register a service factory."""
        self._factories[service_type] = factory
        logger.debug(f"Registered factory: {service_type}")
    
    def get(self, service_type: str) -> Any:
        """Get a service instance."""
        # Check if already instantiated
        if service_type in self._services:
            return self._services[service_type]
        
        # Check if singleton exists
        if service_type in self._singletons:
            self._services[service_type] = self._singletons[service_type]
            return self._services[service_type]
        
        # Check if factory exists
        if service_type in self._factories:
            instance = self._factories[service_type]()
            self._services[service_type] = instance
            return instance
        
        raise ValueError(f"Service {service_type} not registered")
    
    def initialize(self) -> None:
        """Initialize the container with default services."""
        if self._initialized:
            return
        
        try:
            # Initialize core services
            agent_manager = BusinessAgentManager()
            self.register_singleton("agent_manager", agent_manager)
            
            # Initialize service layer
            health_service = HealthService(agent_manager)
            system_info_service = SystemInfoService(agent_manager)
            metrics_service = MetricsService(agent_manager)
            agent_service = AgentService(agent_manager)
            workflow_service = WorkflowService(agent_manager)
            document_service = DocumentService(agent_manager)
            
            self.register_singleton("health_service", health_service)
            self.register_singleton("system_info_service", system_info_service)
            self.register_singleton("metrics_service", metrics_service)
            self.register_singleton("agent_service", agent_service)
            self.register_singleton("workflow_service", workflow_service)
            self.register_singleton("document_service", document_service)
            
            self._initialized = True
            logger.info("Service container initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service container: {str(e)}")
            raise
    
    def shutdown(self) -> None:
        """Shutdown the container and cleanup resources."""
        try:
            # Cleanup services if needed
            for service_type, service in self._services.items():
                if hasattr(service, 'shutdown'):
                    service.shutdown()
            
            self._services.clear()
            self._initialized = False
            logger.info("Service container shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during service container shutdown: {str(e)}")

# Global container instance
_container: Optional[ServiceContainer] = None

@lru_cache()
def get_container() -> ServiceContainer:
    """Get the global service container instance."""
    global _container
    if _container is None:
        _container = ServiceContainer()
        _container.initialize()
    return _container

def reset_container() -> None:
    """Reset the global container (mainly for testing)."""
    global _container
    if _container:
        _container.shutdown()
    _container = None
    get_container.cache_clear()
