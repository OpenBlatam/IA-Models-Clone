"""
Plugin Base

Base classes for plugins.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Interface for plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin.
        
        Args:
            config: Plugin configuration
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute plugin.
        
        Args:
            *args: Arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin."""
        pass


class BasePlugin(PluginInterface):
    """Base plugin implementation."""
    
    def __init__(self, name: str):
        """
        Initialize base plugin.
        
        Args:
            name: Plugin name
        """
        self.name = name
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin."""
        self.config = config
        self.initialized = True
        logger.info(f"Plugin initialized: {self.name}")
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin - to be implemented by subclasses."""
        if not self.initialized:
            raise RuntimeError(f"Plugin not initialized: {self.name}")
        
        raise NotImplementedError("Subclass must implement execute")
    
    def cleanup(self) -> None:
        """Cleanup plugin."""
        self.initialized = False
        logger.info(f"Plugin cleaned up: {self.name}")



