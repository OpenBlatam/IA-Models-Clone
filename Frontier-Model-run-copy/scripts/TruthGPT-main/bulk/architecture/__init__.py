"""
Advanced Architecture - Enterprise-grade optimization system architecture
Provides microservices, design patterns, and advanced architectural components
"""

from .domain import DomainService, Entity, ValueObject, Repository
from .infrastructure import InfrastructureService, DatabaseService, CacheService, MessageQueue
from .application import ApplicationService, Command, Query, Event, Handler
from .presentation import PresentationService, Controller, View, API
from .patterns import DesignPattern, Factory, Builder, Observer, Strategy, Command as CommandPattern

__all__ = [
    'DomainService', 'Entity', 'ValueObject', 'Repository',
    'InfrastructureService', 'DatabaseService', 'CacheService', 'MessageQueue',
    'ApplicationService', 'Command', 'Query', 'Event', 'Handler',
    'PresentationService', 'Controller', 'View', 'API',
    'DesignPattern', 'Factory', 'Builder', 'Observer', 'Strategy', 'CommandPattern'
]
