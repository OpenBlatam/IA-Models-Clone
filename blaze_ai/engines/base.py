"""
Base Engine Infrastructure for Blaze AI System.

This module provides the foundational classes, protocols, and enums
for all engine implementations in the Blaze AI system.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Callable, Awaitable, Protocol

from ..core.interfaces import HealthStatus
from ..utils.logging import get_logger

# =============================================================================
# Core Protocols
# =============================================================================

class Executable(Protocol):
    """Protocol for executable operations."""
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any: ...

class HealthCheckable(Protocol):
    """Protocol for health-checkable components."""
    def get_health_status(self) -> HealthStatus: ...

class Configurable(Protocol):
    """Protocol for configurable components."""
    def get_config(self) -> Dict[str, Any]: ...
    def update_config(self, config: Dict[str, Any]) -> None: ...

class MetricsProvider(Protocol):
    """Protocol for components that provide metrics."""
    def get_metrics(self) -> Dict[str, Any]: ...
    def reset_metrics(self) -> None: ...

# =============================================================================
# Enums and Data Classes
# =============================================================================

class EngineStatus(Enum):
    """Engine status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"
    MAINTENANCE = "maintenance"
    SCALING = "scaling"

class EngineType(Enum):
    """Engine type enumeration."""
    LLM = "llm"
    DIFFUSION = "diffusion"
    ROUTER = "router"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class EnginePriority(Enum):
    """Engine priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class EngineMetadata:
    """Engine metadata information."""
    name: str
    type: EngineType
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: list[str] = None
    priority: EnginePriority = EnginePriority.NORMAL
    created_at: float = None
    updated_at: float = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = time.time()
        self.updated_at = time.time()

@dataclass
class EngineCapabilities:
    """Engine capabilities and limitations."""
    supported_operations: list[str] = None
    max_batch_size: int = 1
    max_concurrent_requests: int = 10
    supports_streaming: bool = False
    supports_async: bool = True
    memory_requirements: Dict[str, Any] = None
    hardware_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.supported_operations is None:
            self.supported_operations = []
        if self.memory_requirements is None:
            self.memory_requirements = {}
        if self.hardware_requirements is None:
            self.hardware_requirements = {}

# =============================================================================
# Abstract Base Engine Class
# =============================================================================

class Engine(ABC, Executable, HealthCheckable, Configurable, MetricsProvider):
    """Abstract base class for all engines in the Blaze AI system."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = get_logger(f"engine.{name}")
        self.metadata = self._create_metadata()
        self.capabilities = self._create_capabilities()
        self.status = EngineStatus.INITIALIZING
        self._initialization_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._metrics = {}
        self._last_operation_time = 0.0
        
        # Initialize engine-specific components
        self._initialize_components()
    
    def _create_metadata(self) -> EngineMetadata:
        """Create engine metadata - can be overridden by subclasses."""
        return EngineMetadata(
            name=self.name,
            type=self._get_engine_type(),
            description=self._get_description(),
            priority=self._get_priority()
        )
    
    def _create_capabilities(self) -> EngineCapabilities:
        """Create engine capabilities - can be overridden by subclasses."""
        return EngineCapabilities(
            supported_operations=self._get_supported_operations(),
            max_batch_size=self._get_max_batch_size(),
            max_concurrent_requests=self._get_max_concurrent_requests(),
            supports_streaming=self._supports_streaming(),
            supports_async=self._supports_async()
        )
    
    def _initialize_components(self):
        """Initialize engine-specific components - can be overridden by subclasses."""
        pass
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    async def _initialize_engine(self) -> None:
        """Initialize specific engine - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_operation(self, operation: str, params: Dict[str, Any]) -> Any:
        """Execute specific operation - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_engine_type(self) -> EngineType:
        """Get engine type - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_description(self) -> str:
        """Get engine description - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_priority(self) -> EnginePriority:
        """Get engine priority - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_supported_operations(self) -> list[str]:
        """Get supported operations - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_max_batch_size(self) -> int:
        """Get maximum batch size - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_max_concurrent_requests(self) -> int:
        """Get maximum concurrent requests - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _supports_streaming(self) -> bool:
        """Check if engine supports streaming - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _supports_async(self) -> bool:
        """Check if engine supports async operations - to be implemented by subclasses."""
        pass
    
    # Concrete methods with default implementations
    async def initialize(self) -> None:
        """Initialize the engine."""
        async with self._initialization_lock:
            if self.status == EngineStatus.INITIALIZING:
                try:
                    await self._initialize_engine()
                    self.status = EngineStatus.IDLE
                    self.logger.info(f"Engine {self.name} initialized successfully")
                except Exception as e:
                    self.status = EngineStatus.ERROR
                    self.logger.error(f"Failed to initialize engine {self.name}: {e}")
                    raise
    
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """Execute an operation with validation and metrics tracking."""
        if self.status == EngineStatus.INITIALIZING:
            await self.initialize()
        
        if self.status == EngineStatus.ERROR:
            raise Exception(f"Engine {self.name} is in ERROR state")
        
        if operation not in self.capabilities.supported_operations:
            raise ValueError(f"Operation '{operation}' not supported by engine {self.name}")
        
        self.status = EngineStatus.BUSY
        start_time = time.time()
        
        try:
            result = await self._execute_operation(operation, params)
            self._update_metrics("successful_requests", 1)
            return result
        except Exception as e:
            self._update_metrics("failed_requests", 1)
            self.status = EngineStatus.ERROR
            raise e
        finally:
            self._update_metrics("total_requests", 1)
            response_time = time.time() - start_time
            self._update_metrics("total_response_time", response_time)
            self._update_metrics("average_response_time", 
                               self._get_metrics("total_response_time") / self._get_metrics("total_requests"))
            self._last_operation_time = time.time()
            self.status = EngineStatus.IDLE
    
    def get_health_status(self) -> HealthStatus:
        """Get engine health status."""
        return HealthStatus(
            component=self.name,
            status=self.status.value,
            message=f"Engine {self.name} is {self.status.value}",
            timestamp=time.time(),
            details={
                "metadata": {
                    "type": self.metadata.type.value,
                    "version": self.metadata.version,
                    "priority": self.metadata.priority.value
                },
                "capabilities": {
                    "supported_operations": self.capabilities.supported_operations,
                    "max_batch_size": self.capabilities.max_batch_size,
                    "max_concurrent_requests": self.capabilities.max_concurrent_requests
                },
                "metrics": self.get_metrics()
            }
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get engine configuration."""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update engine configuration."""
        self.config.update(config)
        self.logger.info(f"Configuration updated for engine {self.name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset engine metrics."""
        self._metrics.clear()
        self.logger.info(f"Metrics reset for engine {self.name}")
    
    def _update_metrics(self, key: str, value: Any):
        """Update metrics with a key-value pair."""
        if key not in self._metrics:
            self._metrics[key] = 0
        
        if isinstance(value, (int, float)):
            if key.startswith("total_"):
                self._metrics[key] += value
            else:
                self._metrics[key] = value
        else:
            self._metrics[key] = value
    
    def _get_metrics(self, key: str, default: Any = 0) -> Any:
        """Get metric value by key."""
        return self._metrics.get(key, default)
    
    async def shutdown(self):
        """Shutdown the engine."""
        self._shutdown_event.set()
        self.status = EngineStatus.OFFLINE
        self.logger.info(f"Shutting down engine {self.name}")
    
    def is_healthy(self) -> bool:
        """Check if engine is healthy."""
        return self.status not in [EngineStatus.ERROR, EngineStatus.OFFLINE]
    
    def can_handle_operation(self, operation: str) -> bool:
        """Check if engine can handle a specific operation."""
        return operation in self.capabilities.supported_operations
    
    def get_utilization(self) -> float:
        """Get engine utilization percentage."""
        if self.capabilities.max_concurrent_requests == 0:
            return 0.0
        
        current_requests = self._get_metrics("current_requests", 0)
        return min(100.0, (current_requests / self.capabilities.max_concurrent_requests) * 100.0)


