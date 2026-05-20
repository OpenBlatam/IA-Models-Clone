"""
Observability Module
Independent observability module for metrics and tracing
"""

from typing import List
from modules.base_module import BaseModule
from infrastructure.observability import ObservabilityServiceFactory
from core.service_container import get_container

logger = __import__("logging").getLogger(__name__)


class ObservabilityModule(BaseModule):
    """Observability feature module"""
    
    def __init__(self):
        super().__init__("observability", "1.0.0")
        self._factory = None
    
    def get_dependencies(self) -> List[str]:
        """Observability module has no dependencies"""
        return []
    
    def _on_initialize(self) -> None:
        """Initialize observability module"""
        self._factory = ObservabilityServiceFactory()
        
        # Register services in container
        container = get_container()
        container.register_service("metrics", self._factory.create_metrics_service())
        container.register_service("tracing", self._factory.create_tracing_service())
        
        logger.info("Observability module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown observability module"""
        logger.info("Observability module shut down")
    
    def get_metrics_service(self):
        """Get metrics service instance"""
        if not self._factory:
            raise RuntimeError("Observability module not initialized")
        return self._factory.create_metrics_service()
    
    def get_tracing_service(self):
        """Get tracing service instance"""
        if not self._factory:
            raise RuntimeError("Observability module not initialized")
        return self._factory.create_tracing_service()















