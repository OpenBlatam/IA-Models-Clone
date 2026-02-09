"""
Microservices Module
"""

from .service_registry import ServiceRegistry, Service, ServiceEndpoint, ServiceStatus, service_registry

__all__ = [
    'ServiceRegistry',
    'Service',
    'ServiceEndpoint',
    'ServiceStatus',
    'service_registry'
]

