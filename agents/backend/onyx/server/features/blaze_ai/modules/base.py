"""
Base Module System for Blaze AI

Provides the foundation for all modular components with common functionality
including lifecycle management, configuration, and status tracking.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Type, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ModuleStatus(Enum):
    """Module operational status."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    IDLE = auto()
    PAUSED = auto()
    ERROR = auto()
    SHUTDOWN = auto()

class ModuleType(Enum):
    """Module classification types."""
    CORE = auto()
    CACHE = auto()
    MONITORING = auto()
    OPTIMIZATION = auto()
    STORAGE = auto()
    EXECUTION = auto()
    UTILITY = auto()

class ModulePriority(Enum):
    """Module execution priority."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ModuleConfig:
    """Base configuration for all modules."""
    name: str
    module_type: ModuleType
    priority: ModulePriority = ModulePriority.NORMAL
    enabled: bool = True
    auto_start: bool = True
    max_workers: int = 4
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval: float = 60.0
    dependencies: Set[str] = field(default_factory=set)
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "module_type": self.module_type.name,
            "priority": self.priority.name,
            "enabled": self.enabled,
            "auto_start": self.auto_start,
            "max_workers": self.max_workers,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "enable_logging": self.enabled,
            "enable_metrics": self.enable_metrics,
            "enable_health_checks": self.enable_health_checks,
            "health_check_interval": self.health_check_interval,
            "dependencies": list(self.dependencies),
            "config_data": self.config_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModuleConfig':
        """Create config from dictionary."""
        return cls(
            name=data["name"],
            module_type=ModuleType[data["module_type"]],
            priority=ModulePriority[data.get("priority", "NORMAL")],
            enabled=data.get("enabled", True),
            auto_start=data.get("auto_start", True),
            max_workers=data.get("max_workers", 4),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            retry_attempts=data.get("retry_attempts", 3),
            enable_logging=data.get("enable_logging", True),
            enable_metrics=data.get("enable_metrics", True),
            enable_health_checks=data.get("enable_health_checks", True),
            health_check_interval=data.get("health_check_interval", 60.0),
            dependencies=set(data.get("dependencies", [])),
            config_data=data.get("config_data", {})
        )

@dataclass
class ModuleMetrics:
    """Module performance metrics."""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    operation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def __post_init__(self):
        if self.end_time is None:
            self.end_time = time.perf_counter()
        if self.duration is None:
            self.duration = self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        return self.success_count / max(total, 1)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.success_count + self.error_count
        return self.error_count / max(total, 1)

@dataclass
class HealthStatus:
    """Module health status."""
    status: ModuleStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

# ============================================================================
# BASE MODULE CLASS
# ============================================================================

class BaseModule(ABC):
    """Base abstract class for all Blaze AI modules."""
    
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.module_id = str(uuid.uuid4())
        self.status = ModuleStatus.UNINITIALIZED
        self.metrics = ModuleMetrics()
        self.health_status = HealthStatus(ModuleStatus.UNINITIALIZED, "Module not initialized")
        
        # Internal state
        self._lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._is_initialized = False
        self._is_shutting_down = False
        self._dependencies_ready = False
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Setup logging
        if self.config.enable_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}.{config.name}")
        else:
            self.logger = logging.getLogger(__name__)
        
        # Setup metrics collection
        if self.config.enable_metrics:
            self._setup_metrics_collection()
        
        # Setup health checks
        if self.config.enable_health_checks:
            self._setup_health_checks()
    
    # ============================================================================
    # ABSTRACT METHODS
    # ============================================================================
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Module-specific initialization."""
        pass
    
    @abstractmethod
    async def _shutdown_impl(self) -> bool:
        """Module-specific shutdown."""
        pass
    
    @abstractmethod
    async def _health_check_impl(self) -> HealthStatus:
        """Module-specific health check."""
        pass
    
    # ============================================================================
    # LIFECYCLE MANAGEMENT
    # ============================================================================
    
    async def initialize(self) -> bool:
        """Initialize the module."""
        async with self._initialization_lock:
            if self._is_initialized:
                return True
            
            try:
                self.logger.info(f"Initializing module: {self.config.name}")
                self.status = ModuleStatus.INITIALIZING
                
                # Check dependencies
                if not await self._check_dependencies():
                    self.logger.error(f"Dependencies not ready for module: {self.config.name}")
                    self.status = ModuleStatus.ERROR
                    return False
                
                # Perform module-specific initialization
                success = await self._initialize_impl()
                
                if success:
                    self._is_initialized = True
                    self.status = ModuleStatus.ACTIVE
                    self.health_status = HealthStatus(
                        ModuleStatus.ACTIVE,
                        "Module initialized successfully"
                    )
                    
                    # Start background tasks
                    self._start_background_tasks()
                    
                    self.logger.info(f"Module initialized successfully: {self.config.name}")
                else:
                    self.status = ModuleStatus.ERROR
                    self.health_status = HealthStatus(
                        ModuleStatus.ERROR,
                        "Module initialization failed"
                    )
                    self.logger.error(f"Module initialization failed: {self.config.name}")
                
                return success
                
            except Exception as e:
                self.status = ModuleStatus.ERROR
                self.health_status = HealthStatus(
                    ModuleStatus.ERROR,
                    f"Initialization error: {str(e)}",
                    error=str(e)
                )
                self.logger.error(f"Module initialization error: {e}")
                return False
    
    async def shutdown(self) -> bool:
        """Shutdown the module."""
        async with self._shutdown_lock:
            if self._is_shutting_down:
                return True
            
            self._is_shutting_down = True
            
            try:
                self.logger.info(f"Shutting down module: {self.config.name}")
                self.status = ModuleStatus.SHUTDOWN
                
                # Stop background tasks
                self._stop_background_tasks()
                
                # Perform module-specific shutdown
                success = await self._shutdown_impl()
                
                if success:
                    self.logger.info(f"Module shutdown completed: {self.config.name}")
                else:
                    self.logger.warning(f"Module shutdown had issues: {self.config.name}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Module shutdown error: {e}")
                return False
    
    async def pause(self) -> bool:
        """Pause the module."""
        if self.status != ModuleStatus.ACTIVE:
            return False
        
        try:
            self.status = ModuleStatus.PAUSED
            self.logger.info(f"Module paused: {self.config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to pause module: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the module."""
        if self.status != ModuleStatus.PAUSED:
            return False
        
        try:
            self.status = ModuleStatus.ACTIVE
            self.logger.info(f"Module resumed: {self.config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to resume module: {e}")
            return False
    
    # ============================================================================
    # DEPENDENCY MANAGEMENT
    # ============================================================================
    
    async def _check_dependencies(self) -> bool:
        """Check if module dependencies are ready."""
        if not self.config.dependencies:
            self._dependencies_ready = True
            return True
        
        # This would be implemented by the module registry
        # For now, assume dependencies are ready
        self._dependencies_ready = True
        return True
    
    def add_dependency(self, dependency_name: str):
        """Add a dependency to the module."""
        self.config.dependencies.add(dependency_name)
    
    def remove_dependency(self, dependency_name: str):
        """Remove a dependency from the module."""
        self.config.dependencies.discard(dependency_name)
    
    # ============================================================================
    # HEALTH MONITORING
    # ============================================================================
    
    def _setup_health_checks(self):
        """Setup periodic health checks."""
        if not self.config.enable_health_checks:
            return
        
        async def health_check_loop():
            while self.status not in [ModuleStatus.SHUTDOWN, ModuleStatus.ERROR]:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    await self._perform_health_check()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health check error: {e}")
                    await asyncio.sleep(5.0)
        
        self._health_check_task = asyncio.create_task(health_check_loop())
    
    async def _perform_health_check(self):
        """Perform a health check."""
        try:
            health_status = await self._health_check_impl()
            self.health_status = health_status
            
            if health_status.status == ModuleStatus.ERROR:
                self.logger.warning(f"Module health check failed: {health_status.message}")
            
        except Exception as e:
            self.health_status = HealthStatus(
                ModuleStatus.ERROR,
                f"Health check error: {str(e)}",
                error=str(e)
            )
    
    async def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self.health_status
    
    # ============================================================================
    # METRICS COLLECTION
    # ============================================================================
    
    def _setup_metrics_collection(self):
        """Setup metrics collection."""
        if not self.config.enable_metrics:
            return
        
        async def metrics_collection_loop():
            while self.status not in [ModuleStatus.SHUTDOWN, ModuleStatus.ERROR]:
                try:
                    await asyncio.sleep(10.0)  # Collect metrics every 10 seconds
                    await self._collect_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                    await asyncio.sleep(5.0)
        
        self._metrics_task = asyncio.create_task(metrics_collection_loop())
    
    async def _collect_metrics(self):
        """Collect current metrics."""
        try:
            # Update memory usage
            try:
                import psutil
                process = psutil.Process()
                self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            except ImportError:
                pass
            
            # Update CPU usage (simplified)
            # In a real implementation, this would use more sophisticated CPU monitoring
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
    
    def get_metrics(self) -> ModuleMetrics:
        """Get current metrics."""
        return self.metrics
    
    def record_operation(self, success: bool = True):
        """Record an operation."""
        self.metrics.operation_count += 1
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.error_count += 1
    
    # ============================================================================
    # BACKGROUND TASK MANAGEMENT
    # ============================================================================
    
    def _start_background_tasks(self):
        """Start background tasks."""
        # Health checks and metrics are started in their respective setup methods
        pass
    
    def _stop_background_tasks(self):
        """Stop background tasks."""
        if self._health_check_task:
            self._health_check_task.cancel()
        
        if self._metrics_task:
            self._metrics_task.cancel()
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status information."""
        return {
            "module_id": self.module_id,
            "name": self.config.name,
            "type": self.config.module_type.name,
            "status": self.status.name,
            "priority": self.config.priority.name,
            "enabled": self.config.enabled,
            "initialized": self._is_initialized,
            "dependencies_ready": self._dependencies_ready,
            "uptime": time.time() - self.metrics.start_time if self._is_initialized else 0,
            "health_status": {
                "status": self.health_status.status.name,
                "message": self.health_status.message,
                "timestamp": self.health_status.timestamp,
                "error": self.health_status.error
            },
            "metrics": {
                "operation_count": self.metrics.operation_count,
                "success_count": self.metrics.success_count,
                "error_count": self.metrics.error_count,
                "success_rate": self.metrics.success_rate,
                "memory_usage_mb": self.metrics.memory_usage_mb
            }
        }
    
    def is_ready(self) -> bool:
        """Check if module is ready for use."""
        return (
            self._is_initialized and
            self.status == ModuleStatus.ACTIVE and
            self._dependencies_ready
        )
    
    def get_config(self) -> ModuleConfig:
        """Get module configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update module configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config {key}: {value}")
    
    # ============================================================================
    # CONTEXT MANAGER SUPPORT
    # ============================================================================
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    # ============================================================================
    # STRING REPRESENTATION
    # ============================================================================
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name}, status={self.status.name})"
