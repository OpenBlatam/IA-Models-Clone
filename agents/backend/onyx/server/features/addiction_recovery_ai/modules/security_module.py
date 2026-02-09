"""
Security Module
Independent security module for authentication and authorization
"""

from typing import List
from modules.base_module import BaseModule
from infrastructure.security import SecurityServiceFactory
from core.service_container import get_container

logger = __import__("logging").getLogger(__name__)


class SecurityModule(BaseModule):
    """Security feature module"""
    
    def __init__(self):
        super().__init__("security", "1.0.0")
        self._factory = None
    
    def get_dependencies(self) -> List[str]:
        """Security module has no dependencies"""
        return []
    
    def _on_initialize(self) -> None:
        """Initialize security module"""
        self._factory = SecurityServiceFactory()
        
        # Register authentication service in container
        container = get_container()
        container.register_service(
            "authentication",
            self._factory.create_authentication_service()
        )
        
        logger.info("Security module initialized")
    
    def _on_shutdown(self) -> None:
        """Shutdown security module"""
        logger.info("Security module shut down")
    
    def get_authentication_service(self):
        """Get authentication service instance"""
        if not self._factory:
            raise RuntimeError("Security module not initialized")
        return self._factory.create_authentication_service()















