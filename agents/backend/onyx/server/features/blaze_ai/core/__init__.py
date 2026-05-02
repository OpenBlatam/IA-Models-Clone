"""
Blaze AI Core System v7.1.0 - Refactored for Maximum Performance

This module provides the core architectural components, interfaces, and configuration
for the Blaze AI system with advanced optimization, caching, and performance monitoring.
"""

from .enums import (
    SystemMode,
    OptimizationLevel,
    ComponentStatus,
    PerformanceLevel
)
from .health import ComponentType
from .settings import (
    PerformanceMetrics,
    ComponentConfig,
    SystemConfig,
    SYSTEM_NAME,
    VERSION
)
from .component import BlazeComponent
from .system import BlazeAISystem, ENABLE_UTILITY_OPTIMIZATIONS
from .performance import PerformanceMonitor
from .container import ServiceContainer
from .factories import (
    create_development_config,
    create_production_config,
    create_maximum_performance_config,
    create_default_config,
    initialize_system,
    create_blaze_ai_system
)

__all__ = [
    # Enums
    "SystemMode",
    "OptimizationLevel", 
    "ComponentStatus",
    "ComponentType",
    "PerformanceLevel",
    
    # Dataclasses
    "PerformanceMetrics",
    "ComponentConfig",
    "SystemConfig",
    
    # Core Classes
    "BlazeComponent",
    "BlazeAISystem",
    "PerformanceMonitor",
    "ServiceContainer",
    
    # Factory Functions
    "create_development_config",
    "create_production_config", 
    "create_maximum_performance_config",
    "create_default_config",
    "initialize_system",
    "create_blaze_ai_system",
    
    # Constants
    "ENABLE_UTILITY_OPTIMIZATIONS",
    "SYSTEM_NAME",
    "VERSION"
]
