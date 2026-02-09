"""
Quantum Domain Module - Módulo de Dominio Cuántico
================================================

Módulo de dominio cuántico que contiene agregados, eventos, value objects
y reglas de negocio del sistema con capacidades cuánticas.
"""

from .quantum_aggregates import (
    QuantumHistoryAggregate,
    QuantumComparisonAggregate,
    QuantumQualityAggregate,
    MultiverseHistoryAggregate,
    TemporalHistoryAggregate,
    DimensionalHistoryAggregate
)
from .quantum_events import (
    QuantumDomainEvent,
    QuantumHistoryCreatedEvent,
    QuantumHistoryUpdatedEvent,
    QuantumHistoryDeletedEvent,
    QuantumComparisonCompletedEvent,
    QuantumQualityAssessedEvent,
    MultiverseEvent,
    TemporalEvent,
    DimensionalEvent,
    ConsciousnessEvent,
    QuantumEntanglementEvent
)
from .quantum_value_objects import (
    QuantumContentId,
    QuantumModelType,
    QuantumQualityScore,
    QuantumSimilarityScore,
    ConsciousnessLevel,
    DimensionalVector,
    TemporalCoordinate,
    QuantumState,
    EntanglementPair,
    SuperpositionState,
    QuantumCoherence
)
from .quantum_exceptions import (
    QuantumDomainException,
    QuantumEntanglementException,
    SuperpositionCollapseException,
    QuantumDecoherenceException,
    MultiverseException,
    TemporalException,
    DimensionalException,
    ConsciousnessException
)

__all__ = [
    "QuantumHistoryAggregate",
    "QuantumComparisonAggregate",
    "QuantumQualityAggregate",
    "MultiverseHistoryAggregate",
    "TemporalHistoryAggregate",
    "DimensionalHistoryAggregate",
    "QuantumDomainEvent",
    "QuantumHistoryCreatedEvent",
    "QuantumHistoryUpdatedEvent",
    "QuantumHistoryDeletedEvent",
    "QuantumComparisonCompletedEvent",
    "QuantumQualityAssessedEvent",
    "MultiverseEvent",
    "TemporalEvent",
    "DimensionalEvent",
    "ConsciousnessEvent",
    "QuantumEntanglementEvent",
    "QuantumContentId",
    "QuantumModelType",
    "QuantumQualityScore",
    "QuantumSimilarityScore",
    "ConsciousnessLevel",
    "DimensionalVector",
    "TemporalCoordinate",
    "QuantumState",
    "EntanglementPair",
    "SuperpositionState",
    "QuantumCoherence",
    "QuantumDomainException",
    "QuantumEntanglementException",
    "SuperpositionCollapseException",
    "QuantumDecoherenceException",
    "MultiverseException",
    "TemporalException",
    "DimensionalException",
    "ConsciousnessException"
]




