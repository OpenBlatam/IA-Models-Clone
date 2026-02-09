"""
Storage Module
Independent storage module with its own configuration and services
"""

from typing import List
from modules.base_module import BaseModule
from infrastructure.storage import StorageServiceFactory
from core.service_container import get_container

logger = __import__("logging").getLogger(__name__)


class StorageModule(BaseModule):
    """Storage feature module"""
    
    def __init__(self):
        super().__init__("storage", "1.0.0")
        self._factory = None
    
    def get_dependencies(self) -> List[str]:
        """Storage module has no dependencies"""
        return []
    
    def _on_initialize(self) -> None:
        """Initialize storage module"""
        self._factory = StorageServiceFactory()
        
        # Register storage service in container
        container = get_container()
        container.register_service("storage", self._factory.create_storage_service())
        
        logger.info("Storage module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown storage module"""
        # Cleanup connections if needed
        logger.info("Storage module shut down")
    
    def get_storage_service(self):
        """Get storage service instance"""
        if not self._factory:
            raise RuntimeError("Storage module not initialized")
        return self._factory.create_storage_service()















