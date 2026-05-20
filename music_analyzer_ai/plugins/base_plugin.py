"""
Base Plugin - Base implementation for plugins
"""

from typing import Dict, Any, Optional
import logging

from ..interfaces.plugin_interface import IPlugin

logger = logging.getLogger(__name__)


class BasePlugin(IPlugin):
    """
    Base implementation for plugins
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self._name = name
        self._version = version
        self._initialized = False
        self._config: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Plugin name"""
        return self._name
    
    @property
    def version(self) -> str:
        """Plugin version"""
        return self._version
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize plugin"""
        self._config = config or {}
        self._initialized = True
        logger.info(f"Plugin {self._name} initialized")
    
    def execute(self, data: Any) -> Any:
        """Execute plugin logic - override in subclasses"""
        if not self._initialized:
            raise RuntimeError(f"Plugin {self._name} not initialized")
        return data
    
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        self._initialized = False
        logger.info(f"Plugin {self._name} cleaned up")
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized"""
        return self._initialized








