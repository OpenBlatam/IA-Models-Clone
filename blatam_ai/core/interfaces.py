"""
🎯 BLATAM AI CORE INTERFACES v6.0.0
====================================

Core interfaces and abstractions for the Blatam AI system:
- 🎯 Clean interface definitions
- 🔧 Protocol-based abstractions
- ⚡ Async-first design patterns
- 🏗️ Builder and factory patterns
- 📊 Observable patterns
- 🚀 Performance optimization interfaces
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Protocol, runtime_checkable
from typing_extensions import Self
import time
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 TYPE VARIABLES AND GENERICS
# =============================================================================

T = TypeVar('T')
TConfig = TypeVar('TConfig', bound='BaseConfig')
TResult = TypeVar('TResult')
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')

# =============================================================================
# 🎯 BASE INTERFACES
# =============================================================================

class Initializable(Protocol):
    """Protocol for components that can be initialized."""
    
    async def initialize(self) -> bool:
        """Initialize the component."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the component."""
        ...

class HealthCheckable(Protocol):
    """Protocol for components that support health checks."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Check component health."""
        ...

class Configurable(Protocol):
    """Protocol for components that can be configured."""
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update configuration."""
        ...

class Observable(Protocol):
    """Protocol for components that can be observed."""
    
    def add_observer(self, observer: 'Observer') -> None:
        """Add an observer."""
        ...
    
    def remove_observer(self, observer: 'Observer') -> None:
        """Remove an observer."""
        ...

class Observer(Protocol):
    """Protocol for observers."""
    
    async def on_event(self, event: 'Event') -> None:
        """Handle an event."""
        ...

# =============================================================================
# 🎯 EVENT SYSTEM
# =============================================================================

class EventType(Enum):
    """System event types."""
    COMPONENT_INITIALIZED = "component_initialized"
    COMPONENT_SHUTDOWN = "component_shutdown"
    PERFORMANCE_UPDATE = "performance_update"
    ERROR_OCCURRED = "error_occurred"
    CONFIGURATION_CHANGED = "configuration_changed"
    HEALTH_CHECK = "health_check"
    METRICS_UPDATE = "metrics_update"
    OPTIMIZATION_TRIGGERED = "optimization_triggered"
    LEARNING_COMPLETED = "learning_completed"
    SELF_HEALING_TRIGGERED = "self_healing_triggered"

@dataclass
class Event:
    """System event with metadata."""
    event_type: EventType
    source: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_type': self.event_type.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'data': self.data,
            'event_id': self.event_id
        }

class EventBus:
    """Centralized event bus for system-wide communication."""
    
    def __init__(self):
        self._observers: Dict[EventType, List[Observer]] = {}
        self._event_history: List[Event] = []
        self._max_history_size = 10000
        self._enabled = True
    
    def enable(self) -> None:
        """Enable event bus."""
        self._enabled = True
        logger.info("📡 Event bus enabled")
    
    def disable(self) -> None:
        """Disable event bus."""
        self._enabled = False
        logger.info("📡 Event bus disabled")
    
    def subscribe(self, event_type: EventType, observer: Observer) -> None:
        """Subscribe an observer to an event type."""
        if event_type not in self._observers:
            self._observers[event_type] = []
        self._observers[event_type].append(observer)
        logger.debug(f"📡 Observer subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, observer: Observer) -> None:
        """Unsubscribe an observer from an event type."""
        if event_type in self._observers:
            try:
                self._observers[event_type].remove(observer)
                logger.debug(f"📡 Observer unsubscribed from {event_type.value}")
            except ValueError:
                pass
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        if not self._enabled:
            return
        
        # Store event in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
        
        # Notify observers
        if event.event_type in self._observers:
            tasks = []
            for observer in self._observers[event.event_type]:
                task = asyncio.create_task(self._notify_observer(observer, event))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.debug(f"📡 Event published: {event.event_type.value} from {event.source}")
    
    async def _notify_observer(self, observer: Observer, event: Event) -> None:
        """Notify a single observer of an event."""
        try:
            await observer.on_event(event)
        except Exception as e:
            logger.error(f"❌ Error in event observer: {e}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type."""
        if event_type:
            filtered = [e for e in self._event_history if e.event_type == event_type]
            return filtered[-limit:]
        return self._event_history[-limit:]
    
    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get number of subscribers for an event type."""
        return len(self._observers.get(event_type, []))

# =============================================================================
# 🎯 CONFIGURATION INTERFACES
# =============================================================================

class BaseConfig(ABC):
    """Base configuration class with validation and serialization."""
    
    def validate(self) -> bool:
        """Validate configuration values."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        raise NotImplementedError
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create configuration from dictionary."""
        raise NotImplementedError
    
    def merge(self, other: 'BaseConfig') -> Self:
        """Merge with another configuration."""
        raise NotImplementedError

@dataclass
class CacheConfig(BaseConfig):
    """Cache configuration."""
    enabled: bool = True
    max_size: int = 10000
    ttl_seconds: int = 3600
    eviction_policy: str = "lru"
    compression_enabled: bool = False
    
    def validate(self) -> bool:
        """Validate cache configuration."""
        if self.max_size <= 0:
            logger.error("❌ Cache max_size must be positive")
            return False
        if self.ttl_seconds < 0:
            logger.error("❌ Cache TTL must be non-negative")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'eviction_policy': self.eviction_policy,
            'compression_enabled': self.compression_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(**data)
    
    def merge(self, other: 'CacheConfig') -> Self:
        """Merge with another cache config."""
        return CacheConfig(
            enabled=other.enabled if other.enabled is not None else self.enabled,
            max_size=other.max_size if other.max_size is not None else self.max_size,
            ttl_seconds=other.ttl_seconds if other.ttl_seconds is not None else self.ttl_seconds,
            eviction_policy=other.eviction_policy if other.eviction_policy else self.eviction_policy,
            compression_enabled=other.compression_enabled if other.compression_enabled is not None else self.compression_enabled
        )

@dataclass
class PerformanceConfig(BaseConfig):
    """Performance configuration."""
    enable_profiling: bool = False
    enable_metrics: bool = True
    sampling_rate: float = 0.1
    max_metrics_history: int = 10000
    performance_threshold: float = 1000.0  # milliseconds
    
    def validate(self) -> bool:
        """Validate performance configuration."""
        if not 0.0 <= self.sampling_rate <= 1.0:
            logger.error("❌ Sampling rate must be between 0.0 and 1.0")
            return False
        if self.max_metrics_history <= 0:
            logger.error("❌ Max metrics history must be positive")
            return False
        if self.performance_threshold <= 0:
            logger.error("❌ Performance threshold must be positive")
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enable_profiling': self.enable_profiling,
            'enable_metrics': self.enable_metrics,
            'sampling_rate': self.sampling_rate,
            'max_metrics_history': self.max_metrics_history,
            'performance_threshold': self.performance_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(**data)
    
    def merge(self, other: 'PerformanceConfig') -> Self:
        """Merge with another performance config."""
        return PerformanceConfig(
            enable_profiling=other.enable_profiling if other.enable_profiling is not None else self.enable_profiling,
            enable_metrics=other.enable_metrics if other.enable_metrics is not None else self.enable_metrics,
            sampling_rate=other.sampling_rate if other.sampling_rate is not None else self.sampling_rate,
            max_metrics_history=other.max_metrics_history if other.max_metrics_history is not None else self.max_metrics_history,
            performance_threshold=other.performance_threshold if other.performance_threshold is not None else self.performance_threshold
        )

# =============================================================================
# 🎯 BUILDER PATTERNS
# =============================================================================

class Builder(Generic[T], ABC):
    """Generic builder interface."""
    
    @abstractmethod
    def build(self) -> T:
        """Build and return the final object."""
        pass
    
    def reset(self) -> Self:
        """Reset the builder to initial state."""
        return self

class ComponentBuilder(Builder[T], ABC):
    """Builder for components."""
    
    def __init__(self):
        self._config: Optional[BaseConfig] = None
        self._dependencies: List[str] = []
        self._metadata: Dict[str, Any] = {}
    
    def with_config(self, config: BaseConfig) -> Self:
        """Set component configuration."""
        self._config = config
        return self
    
    def with_dependencies(self, dependencies: List[str]) -> Self:
        """Set component dependencies."""
        self._dependencies = dependencies
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> Self:
        """Set component metadata."""
        self._metadata = metadata
        return self
    
    def reset(self) -> Self:
        """Reset builder state."""
        self._config = None
        self._dependencies = []
        self._metadata = {}
        return self

# =============================================================================
# 🎯 FACTORY PATTERNS
# =============================================================================

class AbstractFactory(ABC, Generic[T]):
    """Abstract factory interface."""
    
    @abstractmethod
    def create(self, config: BaseConfig, **kwargs) -> T:
        """Create an instance."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get supported instance types."""
        pass

class FactoryRegistry:
    """Registry for factories."""
    
    def __init__(self):
        self._factories: Dict[str, AbstractFactory] = {}
        self._factory_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register_factory(self, name: str, factory: AbstractFactory, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a factory."""
        self._factories[name] = factory
        self._factory_metadata[name] = metadata or {}
        logger.info(f"🏭 Registered factory: {name}")
    
    def get_factory(self, name: str) -> AbstractFactory:
        """Get a factory by name."""
        if name not in self._factories:
            raise KeyError(f"Factory '{name}' not found")
        return self._factories[name]
    
    def get_factory_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a factory."""
        return self._factory_metadata.get(name, {})
    
    def list_factories(self) -> List[str]:
        """List all registered factories."""
        return list(self._factories.keys())
    
    def unregister_factory(self, name: str) -> None:
        """Unregister a factory."""
        if name in self._factories:
            del self._factories[name]
        if name in self._factory_metadata:
            del self._factory_metadata[name]
        logger.info(f"🗑️ Unregistered factory: {name}")

# =============================================================================
# 🎯 OBSERVABLE PATTERNS
# =============================================================================

class Subject:
    """Subject in observer pattern."""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._observer_lock = asyncio.Lock()
    
    async def add_observer(self, observer: Observer) -> None:
        """Add an observer."""
        async with self._observer_lock:
            if observer not in self._observers:
                self._observers.append(observer)
                logger.debug(f"👁️ Observer added to {self.__class__.__name__}")
    
    async def remove_observer(self, observer: Observer) -> None:
        """Remove an observer."""
        async with self._observer_lock:
            try:
                self._observers.remove(observer)
                logger.debug(f"👁️ Observer removed from {self.__class__.__name__}")
            except ValueError:
                pass
    
    async def notify_observers(self, event: Event) -> None:
        """Notify all observers of an event."""
        async with self._observer_lock:
            if not self._observers:
                return
            
            tasks = []
            for observer in self._observers:
                task = asyncio.create_task(observer.on_event(event))
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

# =============================================================================
# 🎯 PERFORMANCE INTERFACES
# =============================================================================

class Profilable(Protocol):
    """Protocol for components that support profiling."""
    
    def start_profiling(self) -> None:
        """Start profiling."""
        ...
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        ...
    
    def is_profiling(self) -> bool:
        """Check if profiling is active."""
        ...

class Optimizable(Protocol):
    """Protocol for components that can be optimized."""
    
    async def optimize(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize based on metrics."""
        ...
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions."""
        ...
    
    def can_optimize(self) -> bool:
        """Check if optimization is possible."""
        ...

class SelfHealing(Protocol):
    """Protocol for components that support self-healing."""
    
    async def detect_issues(self) -> List[Dict[str, Any]]:
        """Detect potential issues."""
        ...
    
    async def self_heal(self) -> bool:
        """Attempt self-healing."""
        ...
    
    def get_healing_strategies(self) -> List[str]:
        """Get available healing strategies."""
        ...

# =============================================================================
# 🎯 LEARNING INTERFACES
# =============================================================================

class Learnable(Protocol):
    """Protocol for components that can learn."""
    
    async def learn(self, data: Any) -> bool:
        """Learn from data."""
        ...
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress."""
        ...
    
    def can_learn(self) -> bool:
        """Check if learning is possible."""
        ...

class Adaptive(Protocol):
    """Protocol for components that can adapt."""
    
    async def adapt(self, feedback: Dict[str, Any]) -> bool:
        """Adapt based on feedback."""
        ...
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history."""
        ...
    
    def is_adaptive(self) -> bool:
        """Check if adaptation is enabled."""
        ...

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    "Initializable",
    "HealthCheckable", 
    "Configurable",
    "Observable",
    "Observer",
    "Profilable",
    "Optimizable",
    "SelfHealing",
    "Learnable",
    "Adaptive",
    
    # Event system
    "EventType",
    "Event",
    "EventBus",
    
    # Configuration
    "BaseConfig",
    "CacheConfig",
    "PerformanceConfig",
    
    # Builders
    "Builder",
    "ComponentBuilder",
    
    # Factories
    "AbstractFactory",
    "FactoryRegistry",
    
    # Observable patterns
    "Subject",
    
    # Type variables
    "T",
    "TConfig",
    "TResult",
    "TInput",
    "TOutput"
] 