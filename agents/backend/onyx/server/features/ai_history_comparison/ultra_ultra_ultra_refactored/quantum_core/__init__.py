"""
Quantum Core Module - Módulo Core Cuántico
=========================================

Módulo core cuántico que contiene la lógica de dominio, aplicación e infraestructura
del sistema ultra-ultra-ultra-refactorizado con capacidades cuánticas.
"""

from .quantum_domain.quantum_aggregates import (
    QuantumHistoryAggregate,
    QuantumComparisonAggregate,
    QuantumQualityAggregate,
    MultiverseHistoryAggregate
)
from .quantum_domain.quantum_events import (
    QuantumDomainEvent,
    QuantumHistoryCreatedEvent,
    QuantumComparisonCompletedEvent,
    MultiverseEvent,
    TemporalEvent,
    DimensionalEvent
)
from .quantum_domain.quantum_value_objects import (
    QuantumContentId,
    QuantumModelType,
    QuantumQualityScore,
    QuantumSimilarityScore,
    ConsciousnessLevel,
    DimensionalVector,
    TemporalCoordinate
)
from .quantum_application.quantum_commands import (
    QuantumCreateHistoryCommand,
    QuantumCompareEntriesCommand,
    QuantumAssessQualityCommand,
    MultiverseAnalysisCommand,
    TemporalPredictionCommand,
    DimensionalAnalysisCommand
)
from .quantum_application.quantum_queries import (
    QuantumGetHistoryQuery,
    QuantumListHistoryQuery,
    QuantumGetComparisonQuery,
    MultiverseSearchQuery,
    TemporalAnalysisQuery,
    DimensionalSearchQuery
)
from .quantum_infrastructure.quantum_event_store import QuantumEventStore
from .quantum_infrastructure.quantum_message_bus import QuantumMessageBus
from .quantum_infrastructure.quantum_plugin_registry import QuantumPluginRegistry

__all__ = [
    "QuantumHistoryAggregate",
    "QuantumComparisonAggregate",
    "QuantumQualityAggregate",
    "MultiverseHistoryAggregate",
    "QuantumDomainEvent",
    "QuantumHistoryCreatedEvent",
    "QuantumComparisonCompletedEvent",
    "MultiverseEvent",
    "TemporalEvent",
    "DimensionalEvent",
    "QuantumContentId",
    "QuantumModelType",
    "QuantumQualityScore",
    "QuantumSimilarityScore",
    "ConsciousnessLevel",
    "DimensionalVector",
    "TemporalCoordinate",
    "QuantumCreateHistoryCommand",
    "QuantumCompareEntriesCommand",
    "QuantumAssessQualityCommand",
    "MultiverseAnalysisCommand",
    "TemporalPredictionCommand",
    "DimensionalAnalysisCommand",
    "QuantumGetHistoryQuery",
    "QuantumListHistoryQuery",
    "QuantumGetComparisonQuery",
    "MultiverseSearchQuery",
    "TemporalAnalysisQuery",
    "DimensionalSearchQuery",
    "QuantumEventStore",
    "QuantumMessageBus",
    "QuantumPluginRegistry"
]




