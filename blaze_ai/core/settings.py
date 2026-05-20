import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .enums import SystemMode, PerformanceLevel, OptimizationLevel
from .health import ComponentType

# System constants
SYSTEM_NAME = "Blaze AI"
VERSION = "7.1.0"
DEFAULT_TIMEOUT = 30.0
MAX_WORKERS = 1024
CACHE_TTL = 3600  # 1 hour
ENABLE_OPTIMIZATIONS = True
ENABLE_UTILITY_OPTIMIZATIONS = True

@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.end_time is None:
            self.end_time = time.perf_counter()
        if self.duration is None:
            self.duration = self.end_time - self.start_time

@dataclass
class ComponentConfig:
    """Component configuration."""
    name: str
    component_type: ComponentType
    performance_level: PerformanceLevel
    max_workers: int = 16
    timeout_seconds: float = DEFAULT_TIMEOUT
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_ttl: int = CACHE_TTL
    enable_monitoring: bool = True
    priority: int = 5

@dataclass
class SystemConfig:
    """System configuration."""
    system_name: str = SYSTEM_NAME
    version: str = VERSION
    system_mode: SystemMode = SystemMode.DEVELOPMENT
    performance_target: PerformanceLevel = PerformanceLevel.STANDARD
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    enable_monitoring: bool = True
    enable_auto_scaling: bool = False
    enable_fault_tolerance: bool = False
    max_concurrent_operations: int = 100
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    engine_configs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "system_name": self.system_name,
            "version": self.version,
            "system_mode": self.system_mode.name,
            "performance_target": self.performance_target.value,
            "optimization_level": self.optimization_level.name,
            "enable_monitoring": self.enable_monitoring,
            "enable_auto_scaling": self.enable_auto_scaling,
            "enable_fault_tolerance": self.enable_fault_tolerance,
            "max_concurrent_operations": self.max_concurrent_operations,
            "components": {name: config.__dict__ for name, config in self.components.items()},
            "engine_configs": self.engine_configs
        }
