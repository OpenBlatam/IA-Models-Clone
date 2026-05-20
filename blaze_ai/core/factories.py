from typing import List, Tuple
from .enums import SystemMode, PerformanceLevel, OptimizationLevel, ComponentStatus
from .health import ComponentType
from .settings import (
    SystemConfig, ComponentConfig, PerformanceMetrics, 
    SYSTEM_NAME, VERSION
)
from .system import BlazeAISystem

def create_default_config() -> SystemConfig:
    """Create default configuration (development)."""
    return create_development_config()

def create_development_config() -> SystemConfig:
    """Create development configuration."""
    return SystemConfig(
        system_mode=SystemMode.DEVELOPMENT,
        performance_target=PerformanceLevel.STANDARD,
        optimization_level=OptimizationLevel.STANDARD,
        enable_monitoring=True,
        enable_auto_scaling=False,
        enable_fault_tolerance=False,
        max_concurrent_operations=50
    )

def create_production_config() -> SystemConfig:
    """Create production configuration."""
    return SystemConfig(
        system_mode=SystemMode.PRODUCTION,
        performance_target=PerformanceLevel.TURBO,
        optimization_level=OptimizationLevel.ADVANCED,
        enable_monitoring=True,
        enable_auto_scaling=True,
        enable_fault_tolerance=True,
        max_concurrent_operations=200
    )

def create_maximum_performance_config() -> SystemConfig:
    """Create maximum performance configuration."""
    config = SystemConfig(
        system_mode=SystemMode.PERFORMANCE,
        performance_target=PerformanceLevel.MARAREAL,
        optimization_level=OptimizationLevel.QUANTUM,
        enable_monitoring=True,
        enable_auto_scaling=True,
        enable_fault_tolerance=True,
        max_concurrent_operations=500
    )
    
    # Add component configurations for optimization utilities
    optimization_components = [
        ("quantum_optimizer", ComponentType.UTILITY, PerformanceLevel.QUANTUM, 128),
        ("neural_turbo_engine", ComponentType.UTILITY, PerformanceLevel.TURBO, 256),
        ("marareal_engine", ComponentType.UTILITY, PerformanceLevel.MARAREAL, 512),
        ("ultra_speed_engine", ComponentType.UTILITY, PerformanceLevel.TURBO, 256),
        ("mass_efficiency_engine", ComponentType.UTILITY, PerformanceLevel.ADVANCED, 128),
        ("ultra_compact_storage", ComponentType.UTILITY, PerformanceLevel.ADVANCED, 64)
    ]
    
    for name, comp_type, perf_level, max_workers in optimization_components:
        config.components[name] = ComponentConfig(
            name=name,
            component_type=comp_type,
            performance_level=perf_level,
            max_workers=max_workers,
            enable_caching=True,
            enable_monitoring=True,
            priority=1
        )
    
    return config

async def initialize_system(config: SystemConfig) -> BlazeAISystem:
    """Initialize the Blaze AI system with the given configuration."""
    system = BlazeAISystem(config)
    success = await system.initialize()
    
    if not success:
        raise RuntimeError("Failed to initialize Blaze AI system")
    
    return system

def create_blaze_ai_system(config: Optional[SystemConfig] = None) -> BlazeAISystem:
    """Create a Blaze AI system instance."""
    if config is None:
        config = create_development_config()
    return BlazeAISystem(config)
