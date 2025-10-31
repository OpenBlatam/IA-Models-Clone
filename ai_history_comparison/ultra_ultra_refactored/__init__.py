"""
Ultra Ultra Refactored AI History Comparison System
=================================================

Sistema ultra-ultra-refactorizado con:
- Microservicios y Service Mesh
- Event-Driven Architecture
- CQRS y Event Sourcing
- Hexagonal Architecture
- Plugin System avanzado
- Circuit Breaker y Resilience Patterns
- API Gateway
- Advanced Monitoring y Observability

Características principales:
- Arquitectura de microservicios
- Event sourcing y CQRS
- Hexagonal architecture
- Plugin system extensible
- Resilience patterns
- Advanced monitoring
- Service mesh
- API gateway
"""

__version__ = "3.0.0"
__author__ = "AI History Team"
__description__ = "Ultra Ultra Refactored AI History Comparison System"

# Core imports
from .core.domain.aggregates import HistoryAggregate, ComparisonAggregate, QualityAggregate
from .core.domain.events import DomainEvent, HistoryCreatedEvent, ComparisonCompletedEvent
from .core.domain.value_objects import ContentId, ModelType, QualityScore, SimilarityScore
from .core.application.commands import CreateHistoryCommand, CompareEntriesCommand
from .core.application.queries import GetHistoryQuery, GetComparisonQuery
from .core.application.handlers import CommandHandler, QueryHandler
from .core.infrastructure.event_store import EventStore
from .core.infrastructure.message_bus import MessageBus
from .core.infrastructure.plugin_registry import PluginRegistry

# Microservices
from .microservices.history_service import HistoryService
from .microservices.comparison_service import ComparisonService
from .microservices.quality_service import QualityService
from .microservices.analytics_service import AnalyticsService

# API Gateway
from .gateway.api_gateway import APIGateway
from .gateway.service_discovery import ServiceDiscovery
from .gateway.load_balancer import LoadBalancer

# Monitoring
from .monitoring.metrics import MetricsCollector
from .monitoring.tracing import DistributedTracer
from .monitoring.health import HealthChecker

__all__ = [
    # Core
    "HistoryAggregate",
    "ComparisonAggregate", 
    "QualityAggregate",
    "DomainEvent",
    "HistoryCreatedEvent",
    "ComparisonCompletedEvent",
    "ContentId",
    "ModelType",
    "QualityScore",
    "SimilarityScore",
    "CreateHistoryCommand",
    "CompareEntriesCommand",
    "GetHistoryQuery",
    "GetComparisonQuery",
    "CommandHandler",
    "QueryHandler",
    "EventStore",
    "MessageBus",
    "PluginRegistry",
    
    # Microservices
    "HistoryService",
    "ComparisonService",
    "QualityService",
    "AnalyticsService",
    
    # Gateway
    "APIGateway",
    "ServiceDiscovery",
    "LoadBalancer",
    
    # Monitoring
    "MetricsCollector",
    "DistributedTracer",
    "HealthChecker"
]




