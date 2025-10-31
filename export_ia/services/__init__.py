"""
Microservices architecture for Export IA.
"""

from .core import ServiceRegistry, ServiceManager
from .communication import MessageBus, EventPublisher, EventSubscriber
from .discovery import ServiceDiscovery
from .gateway import APIGateway

__all__ = [
    "ServiceRegistry",
    "ServiceManager", 
    "MessageBus",
    "EventPublisher",
    "EventSubscriber",
    "ServiceDiscovery",
    "APIGateway"
]




