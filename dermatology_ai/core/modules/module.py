"""
Module Definition
Base class for all modules with lifecycle management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModuleState(str, Enum):
    """Module lifecycle states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ModuleMetadata:
    """Module metadata"""
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)  # Module names
    optional_dependencies: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)  # Services provided
    requires: List[str] = field(default_factory=list)  # Services required
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


class Module(ABC):
    """
    Base class for all modules.
    Provides lifecycle management and dependency injection.
    """
    
    def __init__(self, metadata: ModuleMetadata):
        self.metadata = metadata
        self.state: ModuleState = ModuleState.UNLOADED
        self.config: Dict[str, Any] = {}
        self.dependencies: Dict[str, 'Module'] = {}
        self.provided_services: Dict[str, Any] = {}
        self.error: Optional[str] = None
    
    @abstractmethod
    async def load(self) -> bool:
        """
        Load module (import, validate)
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize module (setup resources)
        
        Args:
            config: Module configuration
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start module (begin operation)
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop module (cleanup)
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def unload(self) -> bool:
        """
        Unload module (remove from memory)
        
        Returns:
            True if successful
        """
        pass
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get service provided by this module
        
        Args:
            service_name: Name of service
            
        Returns:
            Service instance or None
        """
        return self.provided_services.get(service_name)
    
    def provide_service(self, service_name: str, service: Any):
        """Register service provided by this module"""
        self.provided_services[service_name] = service
        logger.debug(f"Module {self.metadata.name} provides service: {service_name}")
    
    def get_dependency(self, module_name: str) -> Optional['Module']:
        """Get dependency module"""
        return self.dependencies.get(module_name)
    
    def set_dependency(self, module_name: str, module: 'Module'):
        """Set dependency module"""
        self.dependencies[module_name] = module
        logger.debug(f"Module {self.metadata.name} depends on: {module_name}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get module state information"""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "state": self.state.value,
            "dependencies": list(self.dependencies.keys()),
            "provides": list(self.provided_services.keys()),
            "error": self.error
        }
    
    def is_healthy(self) -> bool:
        """Check if module is healthy"""
        return self.state in (ModuleState.RUNNING, ModuleState.INITIALIZED)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status
        """
        return {
            "healthy": self.is_healthy(),
            "state": self.state.value,
            "error": self.error
        }















