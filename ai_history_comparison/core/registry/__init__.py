"""
Registry System Module

Dependency injection container and service registry
for the AI History Comparison System.
"""

from .dependency_container import DependencyContainer, ServiceLifetime
from .service_registry import ServiceRegistry, ServiceDefinition
from .injection_context import InjectionContext, Scope
from .auto_wiring import AutoWiring, WiringStrategy
from .circular_dependency import CircularDependencyDetector
from .service_locator import ServiceLocator

__all__ = [
    'DependencyContainer', 'ServiceLifetime',
    'ServiceRegistry', 'ServiceDefinition',
    'InjectionContext', 'Scope',
    'AutoWiring', 'WiringStrategy',
    'CircularDependencyDetector',
    'ServiceLocator'
]





















