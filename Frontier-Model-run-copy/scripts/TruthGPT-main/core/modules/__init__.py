"""
Modular Core System
Highly modular architecture with micro-modules, plugins, and factory patterns
"""

from .base import BaseModule, ModuleRegistry, ModuleFactory
from .interfaces import (
    IOptimizer, IModel, ITrainer, IInferencer, IMonitor, IBenchmarker,
    IPlugin, IConfigurable, ILoggable, IMeasurable
)
from .plugins import PluginManager, PluginLoader
from .factories import (
    OptimizerFactory, ModelFactory, TrainerFactory, 
    InferencerFactory, MonitorFactory, BenchmarkerFactory
)
from .config import ConfigManager, ConfigValidator
from .registry import ComponentRegistry, ServiceRegistry
from .injection import DependencyInjector, ServiceContainer

__all__ = [
    # Base classes
    "BaseModule", "ModuleRegistry", "ModuleFactory",
    
    # Interfaces
    "IOptimizer", "IModel", "ITrainer", "IInferencer", "IMonitor", "IBenchmarker",
    "IPlugin", "IConfigurable", "ILoggable", "IMeasurable",
    
    # Plugin system
    "PluginManager", "PluginLoader",
    
    # Factories
    "OptimizerFactory", "ModelFactory", "TrainerFactory",
    "InferencerFactory", "MonitorFactory", "BenchmarkerFactory",
    
    # Configuration
    "ConfigManager", "ConfigValidator",
    
    # Registry
    "ComponentRegistry", "ServiceRegistry",
    
    # Dependency injection
    "DependencyInjector", "ServiceContainer"
]

