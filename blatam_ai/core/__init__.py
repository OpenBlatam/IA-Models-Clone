"""
🏗️ BLATAM AI CORE ARCHITECTURE v6.0.0
======================================

Core architectural components for the Blatam AI system:
- 🎯 Clean interfaces and abstractions
- 🔧 Component lifecycle management
- ⚙️ Configuration management
- 📊 Performance monitoring
- 🚀 Async-first design
- 🧹 Resource management
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from pathlib import Path
import json
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 SYSTEM ENUMS AND CONSTANTS
# =============================================================================

class SystemMode(Enum):
    """System operation modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

class ComponentStatus(Enum):
    """Component lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"

class ComponentType(Enum):
    """Component types for categorization."""
    ENGINE = "engine"
    SERVICE = "service"
    UTILITY = "utility"
    INTERFACE = "interface"
    FACTORY = "factory"

# =============================================================================
# 🎯 BASE COMPONENT INTERFACES
# =============================================================================

T = TypeVar('T')

class BlatamComponent(ABC):
    """Base interface for all Blatam AI components."""
    
    def __init__(self, config: 'ComponentConfig'):
        self.config = config
        self.status = ComponentStatus.UNINITIALIZED
        self.component_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.last_activity = time.time()
        self.error_count = 0
        self.last_error = None
        self._shutdown_callbacks: List[Callable] = []
        
        # Performance tracking
        self._start_time = None
        self._operation_count = 0
        self._total_operation_time = 0.0
        
        logger.debug(f"🔧 Component '{self.__class__.__name__}' created with ID: {self.component_id}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the component gracefully."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check component health."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        return {
            'component_id': self.component_id,
            'status': self.status.value,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'error_count': self.error_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'operation_count': self._operation_count,
            'total_operation_time': self._total_operation_time,
            'avg_operation_time': (
                self._total_operation_time / self._operation_count 
                if self._operation_count > 0 else 0.0
            )
        }
    
    def add_shutdown_callback(self, callback: Callable) -> None:
        """Add a callback to be executed during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def _update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def _record_operation(self, operation_time: float) -> None:
        """Record operation timing for performance tracking."""
        self._operation_count += 1
        self._total_operation_time += operation_time
    
    async def _execute_with_timing(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with timing and error handling."""
        start_time = time.time()
        try:
            self._update_activity()
            result = await operation(*args, **kwargs)
            operation_time = time.time() - start_time
            self._record_operation(operation_time)
            return result
        except Exception as e:
            self.error_count += 1
            self.last_error = e
            self.status = ComponentStatus.ERROR
            logger.error(f"❌ Error in component '{self.__class__.__name__}': {e}")
            raise

class ComponentFactory(ABC, Generic[T]):
    """Factory interface for creating components."""
    
    @abstractmethod
    async def create_component(
        self, 
        config: 'ComponentConfig',
        **kwargs
    ) -> T:
        """Create a component instance."""
        pass
    
    @abstractmethod
    def get_component_type(self) -> ComponentType:
        """Get the type of component this factory creates."""
        pass
    
    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """Get required configuration fields."""
        pass

# =============================================================================
# 🎯 CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class ComponentConfig:
    """Base configuration for all components."""
    name: str
    component_type: ComponentType
    enabled: bool = True
    max_workers: int = 4
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_size: int = 1000
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'component_type': self.component_type.value,
            'enabled': self.enabled,
            'max_workers': self.max_workers,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'cache_size': self.cache_size,
            'optimization_level': self.optimization_level.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        """Create config from dictionary."""
        # Convert string enum values back to enum instances
        if 'component_type' in data and isinstance(data['component_type'], str):
            data['component_type'] = ComponentType(data['component_type'])
        if 'optimization_level' in data and isinstance(data['optimization_level'], str):
            data['optimization_level'] = OptimizationLevel(data['optimization_level'])
        
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration values."""
        if self.max_workers <= 0:
            logger.error("❌ max_workers must be positive")
            return False
        if self.timeout <= 0:
            logger.error("❌ timeout must be positive")
            return False
        if self.retry_attempts < 0:
            logger.error("❌ retry_attempts must be non-negative")
            return False
        if self.cache_size < 0:
            logger.error("❌ cache_size must be non-negative")
            return False
        return True

@dataclass
class SystemConfig:
    """System-wide configuration."""
    mode: SystemMode = SystemMode.DEVELOPMENT
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    max_components: int = 100
    enable_metrics: bool = True
    enable_profiling: bool = False
    log_level: str = "INFO"
    config_file: Optional[Path] = None
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.mode.value,
            'optimization_level': self.optimization_level.value,
            'max_components': self.max_components,
            'enable_metrics': self.enable_metrics,
            'enable_profiling': self.enable_profiling,
            'log_level': self.log_level,
            'config_file': str(self.config_file) if self.config_file else None,
            'environment': self.environment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create config from dictionary."""
        # Convert string enum values back to enum instances
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = SystemMode(data['mode'])
        if 'optimization_level' in data and isinstance(data['optimization_level'], str):
            data['optimization_level'] = OptimizationLevel(data['optimization_level'])
        
        # Convert config_file string back to Path
        if 'config_file' in data and data['config_file']:
            data['config_file'] = Path(data['config_file'])
        
        return cls(**data)
    
    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"💾 Configuration saved to: {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'SystemConfig':
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

# =============================================================================
# 🎯 SERVICE CONTAINER
# =============================================================================

class ServiceContainer:
    """Dependency injection container for services."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._service_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_service(self, name: str, service: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a service instance."""
        self._services[name] = service
        self._service_metadata[name] = metadata or {}
        logger.debug(f"🔧 Registered service: {name}")
    
    def register_factory(self, name: str, factory: Callable, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a service factory."""
        self._factories[name] = factory
        self._service_metadata[name] = metadata or {}
        logger.debug(f"🏭 Registered factory: {name}")
    
    def register_singleton(self, name: str, factory: Callable, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a singleton service factory."""
        self._factories[name] = factory
        self._service_metadata[name] = metadata or {}
        logger.debug(f"🔒 Registered singleton factory: {name}")
    
    def get_service(self, name: str) -> Any:
        """Get a service by name."""
        if name in self._services:
            return self._services[name]
        
        if name in self._factories:
            if name in self._singletons:
                return self._singletons[name]
            
            service = self._factories[name]()
            self._singletons[name] = service
            return service
        
        raise KeyError(f"Service '{name}' not found")
    
    def has_service(self, name: str) -> bool:
        """Check if a service exists."""
        return name in self._services or name in self._factories
    
    def get_all_services(self) -> List[str]:
        """Get all registered service names."""
        return list(set(self._services.keys()) | set(self._factories.keys()))
    
    def get_service_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a service."""
        return self._service_metadata.get(name, {})
    
    def unregister_service(self, name: str) -> None:
        """Unregister a service."""
        if name in self._services:
            del self._services[name]
        if name in self._factories:
            del self._factories[name]
        if name in self._singletons:
            del self._singletons[name]
        if name in self._service_metadata:
            del self._service_metadata[name]
        logger.debug(f"🗑️ Unregistered service: {name}")

# =============================================================================
# 🎯 PERFORMANCE MONITORING
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for components."""
    component_id: str
    operation_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    last_operation: Optional[float] = None
    
    def record_operation(self, operation_time: float) -> None:
        """Record operation timing."""
        self.operation_count += 1
        self.total_time += operation_time
        self.avg_time = self.total_time / self.operation_count
        self.min_time = min(self.min_time, operation_time)
        self.max_time = max(self.max_time, operation_time)
        self.last_operation = time.time()
    
    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'component_id': self.component_id,
            'operation_count': self.operation_count,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'error_count': self.error_count,
            'last_operation': self.last_operation
        }

class PerformanceMonitor:
    """System-wide performance monitoring."""
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._start_time = time.time()
        self._enabled = True
    
    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
        logger.info("📊 Performance monitoring enabled")
    
    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
        logger.info("📊 Performance monitoring disabled")
    
    def register_component(self, component_id: str) -> None:
        """Register a component for monitoring."""
        if self._enabled:
            self._metrics[component_id] = PerformanceMetrics(component_id)
            logger.debug(f"📊 Registered component for monitoring: {component_id}")
    
    def record_operation(self, component_id: str, operation_time: float) -> None:
        """Record operation timing for a component."""
        if self._enabled and component_id in self._metrics:
            self._metrics[component_id].record_operation(operation_time)
    
    def record_error(self, component_id: str) -> None:
        """Record an error for a component."""
        if self._enabled and component_id in self._metrics:
            self._metrics[component_id].record_error()
    
    def get_component_metrics(self, component_id: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific component."""
        return self._metrics.get(component_id)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all component metrics."""
        return self._metrics.copy()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        if not self._metrics:
            return {}
        
        total_operations = sum(m.operation_count for m in self._metrics.values())
        total_time = sum(m.total_time for m in self._metrics.values())
        total_errors = sum(m.error_count for m in self._metrics.values())
        
        return {
            'uptime': time.time() - self._start_time,
            'total_components': len(self._metrics),
            'total_operations': total_operations,
            'total_time': total_time,
            'avg_operation_time': total_time / total_operations if total_operations > 0 else 0.0,
            'total_errors': total_errors,
            'error_rate': total_errors / total_operations if total_operations > 0 else 0.0
        }

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_default_config() -> SystemConfig:
    """Create default system configuration."""
    return SystemConfig()

def create_development_config() -> SystemConfig:
    """Create development environment configuration."""
    return SystemConfig(
        mode=SystemMode.DEVELOPMENT,
        optimization_level=OptimizationLevel.MINIMAL,
        enable_metrics=True,
        enable_profiling=True,
        log_level="DEBUG"
    )

def create_production_config() -> SystemConfig:
    """Create production environment configuration."""
    return SystemConfig(
        mode=SystemMode.PRODUCTION,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_metrics=True,
        enable_profiling=False,
        log_level="WARNING"
    )

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SystemMode",
    "OptimizationLevel", 
    "ComponentStatus",
    "ComponentType",
    
    # Base classes
    "BlatamComponent",
    "ComponentFactory",
    
    # Configuration
    "ComponentConfig",
    "SystemConfig",
    
    # Services
    "ServiceContainer",
    
    # Performance
    "PerformanceMetrics",
    "PerformanceMonitor",
    
    # Factory functions
    "create_default_config",
    "create_development_config",
    "create_production_config"
] 