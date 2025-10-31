"""
Factory Pattern Implementation

Component creation and instantiation factories for
the AI History Comparison System.
"""

from .component_factory import ComponentFactory, ComponentRegistry
from .service_factory import ServiceFactory, ServiceRegistry
from .analyzer_factory import AnalyzerFactory, AnalyzerRegistry
from .engine_factory import EngineFactory, EngineRegistry
from .integration_factory import IntegrationFactory, IntegrationRegistry
from .config_factory import ConfigFactory, ConfigRegistry
from .plugin_factory import PluginFactory, PluginRegistry

__all__ = [
    'ComponentFactory', 'ComponentRegistry',
    'ServiceFactory', 'ServiceRegistry',
    'AnalyzerFactory', 'AnalyzerRegistry',
    'EngineFactory', 'EngineRegistry',
    'IntegrationFactory', 'IntegrationRegistry',
    'ConfigFactory', 'ConfigRegistry',
    'PluginFactory', 'PluginRegistry'
]





















