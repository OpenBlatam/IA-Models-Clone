"""
Ultra-Modular Architecture - Component Registry
==============================================

Ultra-modular system with independent components, microservices, and plugin architecture.
"""

from .registry import ComponentRegistry, ServiceRegistry
from .discovery import ServiceDiscovery, ComponentDiscovery
from .events import EventBus, EventHandler, Event
from .plugins import PluginManager, Plugin, PluginInterface
from .gateway import APIGateway, Route, Middleware
from .microservices import Microservice, ServiceContainer
from .communication import MessageBus, ServiceClient, ServiceServer

__version__ = "4.0.0"
__author__ = "AI Document Processor Team"
__description__ = "Ultra-modular AI document processing with microservices architecture"

__all__ = [
    # Registry
    "ComponentRegistry",
    "ServiceRegistry",
    
    # Discovery
    "ServiceDiscovery", 
    "ComponentDiscovery",
    
    # Events
    "EventBus",
    "EventHandler",
    "Event",
    
    # Plugins
    "PluginManager",
    "Plugin",
    "PluginInterface",
    
    # Gateway
    "APIGateway",
    "Route",
    "Middleware",
    
    # Microservices
    "Microservice",
    "ServiceContainer",
    
    # Communication
    "MessageBus",
    "ServiceClient",
    "ServiceServer",
]

















