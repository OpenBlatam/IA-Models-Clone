"""
Core Module - Módulo Core
========================

Módulo core que contiene la lógica de dominio, aplicación e infraestructura
del sistema ultra-ultra-refactorizado.
"""

from .domain.aggregates import HistoryAggregate, ComparisonAggregate, QualityAggregate
from .domain.events import DomainEvent, HistoryCreatedEvent, ComparisonCompletedEvent
from .domain.value_objects import ContentId, ModelType, QualityScore, SimilarityScore
from .application.commands import CreateHistoryCommand, CompareEntriesCommand
from .application.queries import GetHistoryQuery, GetComparisonQuery
from .application.handlers import CommandHandler, QueryHandler
from .infrastructure.event_store import EventStore
from .infrastructure.message_bus import MessageBus
from .infrastructure.plugin_registry import PluginRegistry

__all__ = [
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
    "PluginRegistry"
]




