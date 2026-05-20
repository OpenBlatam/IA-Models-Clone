"""
Base Module Implementation
Base class for all feature modules
"""

import logging
from typing import List
from core.module_registry import IModule

logger = logging.getLogger(__name__)


class BaseModule(IModule):
    """Base implementation for modules"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Module name"""
        return self._name
    
    @property
    def version(self) -> str:
        """Module version"""
        return self._version
    
    def initialize(self) -> None:
        """Initialize module"""
        if self._initialized:
            logger.warning(f"Module {self.name} already initialized")
            return
        
        logger.info(f"Initializing module: {self.name} v{self.version}")
        self._on_initialize()
        self._initialized = True
        logger.info(f"Module {self.name} initialized")
    
    def shutdown(self) -> None:
        """Shutdown module"""
        if not self._initialized:
            return
        
        logger.info(f"Shutting down module: {self.name}")
        self._on_shutdown()
        self._initialized = False
        logger.info(f"Module {self.name} shut down")
    
    def get_dependencies(self) -> List[str]:
        """Get module dependencies (override in subclasses)"""
        return []
    
    def _on_initialize(self) -> None:
        """Override in subclasses for initialization logic"""
        pass
    
    def _on_shutdown(self) -> None:
        """Override in subclasses for shutdown logic"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if module is initialized"""
        return self._initialized















