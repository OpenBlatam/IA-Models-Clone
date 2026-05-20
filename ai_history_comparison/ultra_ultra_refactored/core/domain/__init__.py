"""
Domain Module - Módulo de Dominio
===============================

Módulo de dominio que contiene agregados, eventos, value objects
y reglas de negocio del sistema.
"""

from .aggregates import HistoryAggregate, ComparisonAggregate, QualityAggregate
from .events import (
    DomainEvent,
    HistoryCreatedEvent,
    HistoryUpdatedEvent,
    HistoryDeletedEvent,
    ComparisonCompletedEvent,
    QualityAssessedEvent,
    AnalysisCompletedEvent
)
from .value_objects import (
    ContentId,
    ModelType,
    QualityScore,
    SimilarityScore,
    ContentMetrics,
    SentimentAnalysis,
    TextComplexity,
    AnalysisResult
)
from .exceptions import (
    DomainException,
    AggregateNotFoundException,
    InvalidStateException,
    BusinessRuleViolationException
)

__all__ = [
    "HistoryAggregate",
    "ComparisonAggregate",
    "QualityAggregate",
    "DomainEvent",
    "HistoryCreatedEvent",
    "HistoryUpdatedEvent",
    "HistoryDeletedEvent",
    "ComparisonCompletedEvent",
    "QualityAssessedEvent",
    "AnalysisCompletedEvent",
    "ContentId",
    "ModelType",
    "QualityScore",
    "SimilarityScore",
    "ContentMetrics",
    "SentimentAnalysis",
    "TextComplexity",
    "AnalysisResult",
    "DomainException",
    "AggregateNotFoundException",
    "InvalidStateException",
    "BusinessRuleViolationException"
]




