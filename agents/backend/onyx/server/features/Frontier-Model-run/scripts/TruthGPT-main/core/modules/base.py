"""
Base Module System
Foundation for highly modular architecture
"""

import abc
import logging
import time
from typing import Dict, Any, List, Optional, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ModuleState(Enum):
    """Module lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ModuleMetadata:
    """Metadata for a module"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

class BaseModule(abc.ABC):
    """Base class for all modules"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.state = ModuleState.UNINITIALIZED
        self.metadata = ModuleMetadata(name=name, version="1.0.0", description="")
        self.logger = logging.getLogger(f"module.{name}")
        self._lock = threading.Lock()
        self._dependencies = {}
        self._callbacks = {
            'on_initialize': [],
            'on_start': [],
            'on_stop': [],
            'on_error': []
        }
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the module"""
        pass
    
    @abc.abstractmethod
    def start(self) -> bool:
        """Start the module"""
        pass
    
    @abc.abstractmethod
    def stop(self) -> bool:
        """Stop the module"""
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> bool:
        """Cleanup module resources"""
        pass
    
    def get_state(self) -> ModuleState:
        """Get current module state"""
        with self._lock:
            return self.state
    
    def set_state(self, state: ModuleState) -> None:
        """Set module state"""
        with self._lock:
            old_state = self.state
            self.state = state
            self.logger.debug(f"State changed: {old_state.value} -> {state.value}")
    
    def add_dependency(self, name: str, module: 'BaseModule') -> None:
        """Add a dependency"""
        self._dependencies[name] = module
        self.metadata.dependencies.append(name)
    
    def get_dependency(self, name: str) -> Optional['BaseModule']:
        """Get a dependency"""
        return self._dependencies.get(name)
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """Add event callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger event callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback error for {event}: {e}")
    
    def is_healthy(self) -> bool:
        """Check if module is healthy"""
        return self.state in [ModuleState.INITIALIZED, ModuleState.RUNNING]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "dependencies": len(self._dependencies),
            "callbacks": sum(len(callbacks) for callbacks in self._callbacks.values())
        }

class ModuleRegistry:
    """Registry for managing modules"""
    
    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._lock = threading.Lock()
    
    def register(self, module: BaseModule) -> bool:
        """Register a module"""
        with self._lock:
            if module.name in self._modules:
                self.logger.warning(f"Module {module.name} already registered")
                return False
            
            self._modules[module.name] = module
            self.logger.info(f"Registered module: {module.name}")
            return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a module"""
        with self._lock:
            if name not in self._modules:
                return False
            
            module = self._modules.pop(name)
            module.cleanup()
            self.logger.info(f"Unregistered module: {name}")
            return True
    
    def get(self, name: str) -> Optional[BaseModule]:
        """Get a module by name"""
        with self._lock:
            return self._modules.get(name)
    
    def list_modules(self) -> List[str]:
        """List all registered modules"""
        with self._lock:
            return list(self._modules.keys())
    
    def get_healthy_modules(self) -> List[BaseModule]:
        """Get all healthy modules"""
        with self._lock:
            return [module for module in self._modules.values() if module.is_healthy()]
    
    def get_modules_by_state(self, state: ModuleState) -> List[BaseModule]:
        """Get modules by state"""
        with self._lock:
            return [module for module in self._modules.values() if module.get_state() == state]

class ModuleFactory(Generic[T]):
    """Factory for creating modules"""
    
    def __init__(self):
        self._creators: Dict[str, Type[T]] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, module_class: Type[T], config: Optional[Dict[str, Any]] = None) -> None:
        """Register a module class"""
        self._creators[name] = module_class
        if config:
            self._configs[name] = config
        logger.info(f"Registered module class: {name}")
    
    def create(self, name: str, **kwargs) -> Optional[T]:
        """Create a module instance"""
        if name not in self._creators:
            logger.error(f"Unknown module class: {name}")
            return None
        
        module_class = self._creators[name]
        config = self._configs.get(name, {})
        config.update(kwargs)
        
        try:
            instance = module_class(name, config)
            logger.info(f"Created module instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create module {name}: {e}")
            return None
    
    def list_available(self) -> List[str]:
        """List available module classes"""
        return list(self._creators.keys())
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a module class"""
        return self._configs.get(name, {})

class ModuleManager:
    """Manager for module lifecycle"""
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.factory = ModuleFactory()
        self._startup_order = []
        self._shutdown_order = []
    
    def register_module_class(self, name: str, module_class: Type[BaseModule], config: Optional[Dict[str, Any]] = None) -> None:
        """Register a module class"""
        self.factory.register(name, module_class, config)
    
    def create_module(self, name: str, module_class_name: str, **kwargs) -> Optional[BaseModule]:
        """Create and register a module"""
        module = self.factory.create(module_class_name, **kwargs)
        if module:
            self.registry.register(module)
            return module
        return None
    
    def start_module(self, name: str) -> bool:
        """Start a module"""
        module = self.registry.get(name)
        if not module:
            logger.error(f"Module not found: {name}")
            return False
        
        try:
            # Initialize if needed
            if module.get_state() == ModuleState.UNINITIALIZED:
                if not module.initialize():
                    logger.error(f"Failed to initialize module: {name}")
                    return False
                module.set_state(ModuleState.INITIALIZED)
            
            # Start the module
            if module.start():
                module.set_state(ModuleState.RUNNING)
                module._trigger_callbacks('on_start')
                logger.info(f"Started module: {name}")
                return True
            else:
                logger.error(f"Failed to start module: {name}")
                return False
                
        except Exception as e:
            module.set_state(ModuleState.ERROR)
            module._trigger_callbacks('on_error', e)
            logger.error(f"Error starting module {name}: {e}")
            return False
    
    def stop_module(self, name: str) -> bool:
        """Stop a module"""
        module = self.registry.get(name)
        if not module:
            logger.error(f"Module not found: {name}")
            return False
        
        try:
            module.set_state(ModuleState.STOPPING)
            if module.stop():
                module.set_state(ModuleState.STOPPED)
                module._trigger_callbacks('on_stop')
                logger.info(f"Stopped module: {name}")
                return True
            else:
                logger.error(f"Failed to stop module: {name}")
                return False
                
        except Exception as e:
            module.set_state(ModuleState.ERROR)
            module._trigger_callbacks('on_error', e)
            logger.error(f"Error stopping module {name}: {e}")
            return False
    
    def start_all(self) -> Dict[str, bool]:
        """Start all registered modules"""
        results = {}
        for name in self.registry.list_modules():
            results[name] = self.start_module(name)
        return results
    
    def stop_all(self) -> Dict[str, bool]:
        """Stop all registered modules"""
        results = {}
        for name in reversed(self.registry.list_modules()):
            results[name] = self.stop_module(name)
        return results
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        status = {}
        for name in self.registry.list_modules():
            module = self.registry.get(name)
            if module:
                status[name] = {
                    "state": module.get_state().value,
                    "healthy": module.is_healthy(),
                    "metrics": module.get_metrics()
                }
        return status

