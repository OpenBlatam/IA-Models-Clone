"""
Resource manager for audio processing resources.
"""

from typing import Dict, Optional, Any, Callable
from contextlib import contextmanager

from ..logger import logger


class ResourceManager:
    """
    Manages shared resources across components.
    
    Provides:
    - Resource registration and retrieval
    - Automatic cleanup
    - Resource pooling
    """
    
    def __init__(self):
        """Initialize resource manager."""
        self._resources: Dict[str, Any] = {}
        self._cleanup_handlers: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        resource: Any,
        cleanup_handler: Optional[Callable] = None
    ):
        """
        Register a resource.
        
        Args:
            name: Resource name
            resource: Resource object
            cleanup_handler: Optional cleanup function
        """
        if name in self._resources:
            logger.warning(f"Resource '{name}' already registered, overwriting")
        
        self._resources[name] = resource
        if cleanup_handler:
            self._cleanup_handlers[name] = cleanup_handler
        
        logger.debug(f"Registered resource: {name}")
    
    def get(self, name: str) -> Optional[Any]:
        """
        Get a resource by name.
        
        Args:
            name: Resource name
            
        Returns:
            Resource or None if not found
        """
        return self._resources.get(name)
    
    def unregister(self, name: str):
        """
        Unregister a resource.
        
        Args:
            name: Resource name
        """
        if name in self._cleanup_handlers:
            try:
                self._cleanup_handlers[name](self._resources.get(name))
            except Exception as e:
                logger.warning(f"Error cleaning up resource '{name}': {str(e)}")
        
        if name in self._resources:
            del self._resources[name]
            logger.debug(f"Unregistered resource: {name}")
    
    def cleanup_all(self):
        """Clean up all registered resources."""
        for name in list(self._resources.keys()):
            self.unregister(name)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup_all()
        return False

