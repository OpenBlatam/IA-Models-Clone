"""
Base component with lifecycle management.

Refactored to:
- Extract error codes to constants
- Improve type hints and documentation
- Add state validation
- Enhance resource management
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator
from contextlib import contextmanager

from ..exceptions import AudioInitializationError
from ..logger import logger

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Error codes
ERROR_CODE_INIT_FAILED = "INIT_FAILED"
ERROR_CODE_NOT_INITIALIZED = "NOT_INITIALIZED"
ERROR_CODE_ALREADY_INITIALIZED = "ALREADY_INITIALIZED"
ERROR_CODE_RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"

# ════════════════════════════════════════════════════════════════════════════
# BASE COMPONENT
# ════════════════════════════════════════════════════════════════════════════

class BaseComponent(ABC):
    """
    Base class for all components with lifecycle management.
    
    Provides common functionality:
    - Initialization tracking and validation
    - Resource cleanup and management
    - State management (initialized, ready)
    - Error handling with specific error codes
    - Context manager support
    
    Usage:
        # Direct usage
        component = MyComponent()
        component.initialize()
        try:
            # use component
            pass
        finally:
            component.cleanup()
        
        # Context manager
        with MyComponent() as component:
            # use component
            pass
        
        # Managed context with initialization params
        with component.managed(param1=value1) as comp:
            # use component
            pass
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize base component.
        
        Args:
            name: Component name (defaults to class name)
        """
        self._name = name or self.__class__.__name__
        self._initialized = False
        self._ready = False
        self._resources: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Component name."""
        return self._name
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    @property
    def is_ready(self) -> bool:
        """Check if component is ready for use."""
        return self._ready
    
    def _ensure_initialized(self) -> None:
        """
        Ensure component is initialized.
        
        Raises:
            AudioInitializationError: If component is not initialized
        """
        if not self._initialized:
            raise AudioInitializationError(
                f"{self._name} is not initialized. Call initialize() first.",
                component=self._name,
                error_code=ERROR_CODE_NOT_INITIALIZED
            )
    
    def _ensure_ready(self) -> None:
        """
        Ensure component is ready for use.
        
        Raises:
            AudioInitializationError: If component is not ready
        """
        self._ensure_initialized()
        if not self._ready:
            raise AudioInitializationError(
                f"{self._name} is initialized but not ready for use.",
                component=self._name,
                error_code=ERROR_CODE_NOT_INITIALIZED
            )
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the component.
        
        Safe to call multiple times (idempotent).
        
        Args:
            **kwargs: Initialization parameters
            
        Returns:
            True if initialization successful
            
        Raises:
            AudioInitializationError: If initialization fails
        """
        if self._initialized:
            logger.debug(f"{self._name} already initialized")
            return True
        
        try:
            logger.debug(f"Initializing {self._name}")
            self._do_initialize(**kwargs)
            self._initialized = True
            self._ready = True
            logger.info(f"{self._name} initialized successfully")
            return True
        except AudioInitializationError:
            # Re-raise initialization errors as-is
            raise
        except Exception as e:
            self._ready = False
            raise AudioInitializationError(
                f"Failed to initialize {self._name}: {str(e)}",
                component=self._name,
                error_code=ERROR_CODE_INIT_FAILED
            ) from e
    
    @abstractmethod
    def _do_initialize(self, **kwargs) -> None:
        """
        Perform component-specific initialization.
        
        This method should be implemented by subclasses to perform
        their specific initialization logic.
        
        Args:
            **kwargs: Initialization parameters
            
        Raises:
            Exception: If initialization fails
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up component resources.
        
        Safe to call multiple times (idempotent).
        Automatically cleans up all registered resources.
        """
        if not self._initialized:
            return
        
        try:
            logger.debug(f"Cleaning up {self._name}")
            self._do_cleanup()
            self._cleanup_resources()
            self._ready = False
            logger.debug(f"{self._name} cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup of {self._name}: {str(e)}")
        finally:
            self._initialized = False
    
    def _do_cleanup(self) -> None:
        """
        Perform component-specific cleanup.
        
        Override in subclasses if additional cleanup is needed.
        This is called before resource cleanup.
        """
        pass
    
    def _cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        for name, resource in list(self._resources.items()):
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
            except Exception as e:
                logger.warning(
                    f"Error cleaning up resource '{name}' in {self._name}: {str(e)}"
                )
        
        self._resources.clear()
    
    def register_resource(self, name: str, resource: Any) -> None:
        """
        Register a resource for automatic cleanup.
        
        Resources with 'close()' or 'cleanup()' methods will be
        automatically cleaned up when component.cleanup() is called.
        
        Args:
            name: Resource name
            resource: Resource object
        """
        if name in self._resources:
            logger.warning(
                f"Resource '{name}' already registered in {self._name}, overwriting"
            )
        self._resources[name] = resource
        logger.debug(f"Registered resource '{name}' in {self._name}")
    
    def get_resource(self, name: str) -> Optional[Any]:
        """
        Get registered resource.
        
        Args:
            name: Resource name
            
        Returns:
            Resource or None if not found
        """
        return self._resources.get(name)
    
    def unregister_resource(self, name: str) -> bool:
        """
        Unregister a resource.
        
        Args:
            name: Resource name
            
        Returns:
            True if resource was found and unregistered, False otherwise
        """
        if name in self._resources:
            del self._resources[name]
            logger.debug(f"Unregistered resource '{name}' from {self._name}")
            return True
        return False
    
    def list_resources(self) -> Iterator[str]:
        """
        List all registered resource names.
        
        Returns:
            Iterator over resource names
        """
        return iter(self._resources.keys())
    
    @contextmanager
    def managed(self, **kwargs) -> Iterator['BaseComponent']:
        """
        Context manager for component lifecycle with initialization.
        
        Usage:
            with component.managed(param1=value1) as comp:
                # use component
                pass
        
        Args:
            **kwargs: Initialization parameters
            
        Yields:
            Self (the component instance)
        """
        self.initialize(**kwargs)
        try:
            yield self
        finally:
            self.cleanup()
    
    def __enter__(self) -> 'BaseComponent':
        """
        Context manager entry.
        
        Automatically initializes if not already initialized.
        
        Returns:
            Self (the component instance)
        """
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context manager exit.
        
        Automatically cleans up resources.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
            
        Returns:
            False to allow exceptions to propagate
        """
        self.cleanup()
        return False
    
    def __repr__(self) -> str:
        """String representation of the component."""
        status = "ready" if self._ready else ("initialized" if self._initialized else "uninitialized")
        return f"{self.__class__.__name__}(name='{self._name}', status='{status}')"
