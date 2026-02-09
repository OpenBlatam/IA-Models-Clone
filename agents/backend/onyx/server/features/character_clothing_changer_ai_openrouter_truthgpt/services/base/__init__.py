"""
Base Service Classes
"""

from .base_service import BaseService, ServiceStatistics
from .service_registry import ServiceRegistry, get_service_registry, register_service, get_service
from .singleton import singleton, SingletonMeta, get_or_create_service

__all__ = [
    'BaseService',
    'ServiceStatistics',
    'ServiceRegistry',
    'get_service_registry',
    'register_service',
    'get_service',
    'singleton',
    'SingletonMeta',
    'get_or_create_service'
]

