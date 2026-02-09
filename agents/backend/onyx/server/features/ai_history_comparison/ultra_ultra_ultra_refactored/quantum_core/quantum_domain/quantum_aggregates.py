"""
Quantum Domain Aggregates - Agregados de Dominio Cuántico
=======================================================

Agregados de dominio cuántico que encapsulan la lógica de negocio
y mantienen la consistencia de los datos con capacidades cuánticas.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

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


class QuantumCoherenceLevel(Enum):
    """Niveles de coherencia cuántica."""
    MAXIMUM = "maximum"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DECOHERENT = "decoherent"


@dataclass
class QuantumHistoryAggregate:
    """
    Agregado cuántico de historial que encapsula una entrada de historial
    con capacidades cuánticas y superposición de estados.
    """
    
    # Identidad cuántica
    id: QuantumContentId
    version: int = 0
    quantum_state: QuantumState = field(default_factory=lambda: QuantumState())
    
    # Atributos cuánticos
    model_type: QuantumModelType
    content: str
    quantum_content: SuperpositionState = field(default_factory=lambda: SuperpositionState())
    metadata: Dict[str, Any] = field(default_factory=dict)
    quantum_metadata: Dict[str, SuperpositionState] = field(default_factory=dict)
    
    # Análisis cuántico
    quantum_quality_score: Optional[QuantumQualityScore] = None
    consciousness_level: Optional[ConsciousnessLevel] = None
    dimensional_vector: Optional[DimensionalVector] = None
    temporal_coordinate: Optional[TemporalCoordinate] = None
    
    # Entrelazamiento cuántico
    entangled_pairs: List[EntanglementPair] = field(default_factory=list)
    coherence_level: QuantumCoherenceLevel = QuantumCoherenceLevel.MAXIMUM
    
    # Metadatos temporales
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    quantum_created_at: TemporalCoordinate = field(default_factory=lambda: TemporalCoordinate())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Eventos no confirmados
    _uncommitted_events: List[QuantumDomainEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        model_type: QuantumModelType,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quantum_parameters: Optional[Dict[str, Any]] = None
    ) -> 'QuantumHistoryAggregate':
        """
        Factory method para crear un nuevo agregado cuántico de historial.
        
        Args:
            model_type: Tipo de modelo cuántico de IA
            content: Contenido generado
            user_id: ID del usuario
            session_id: ID de sesión
            metadata: Metadatos adicionales
            quantum_parameters: Parámetros cuánticos
            
        Returns:
            QuantumHistoryAggregate: Nuevo agregado cuántico
            
        Raises:
            QuantumDomainException: Si se violan reglas cuánticas
        """
        # Validación cuántica del contenido
        if not content or not content.strip():
            raise QuantumDomainException("Content cannot be empty in quantum space")
        
        if len(content) > 50000:
            raise QuantumDomainException("Content exceeds quantum dimensional limits")
        
        # Crear estado cuántico inicial
        quantum_state = QuantumState()
        quantum_state.initialize_superposition()
        
        # Crear superposición de contenido
        quantum_content = SuperpositionState()
        quantum_content.add_state(content.strip(), probability=1.0)
        
        # Crear agregado cuántico
        aggregate = cls(
            id=QuantumContentId(str(uuid.uuid4())),
            model_type=model_type,
            content=content.strip(),
            quantum_content=quantum_content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            quantum_state=quantum_state,
            quantum_created_at=TemporalCoordinate.now()
        )
        
        # Aplicar parámetros cuánticos
        if quantum_parameters:
            aggregate._apply_quantum_parameters(quantum_parameters)
        
        # Crear evento cuántico de dominio
        event = QuantumHistoryCreatedEvent(
            aggregate_id=aggregate.id,
            model_type=model_type,
            content=content.strip(),
            quantum_content=quantum_content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            quantum_parameters=quantum_parameters or {},
            temporal_coordinate=aggregate.quantum_created_at,
            occurred_at=datetime.utcnow()
        )
        
        aggregate._uncommitted_events.append(event)
        return aggregate
    
    def update_content_quantum(self, new_content: str, quantum_parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Actualizar el contenido con capacidades cuánticas.
        
        Args:
            new_content: Nuevo contenido
            quantum_parameters: Parámetros cuánticos para la actualización
            
        Raises:
            QuantumDomainException: Si se violan reglas cuánticas
        """
        if not new_content or not new_content.strip():
            raise QuantumDomainException("Content cannot be empty in quantum space")
        
        # Colapsar superposición actual
        old_content = self.quantum_content.collapse()
        
        # Crear nueva superposición
        new_quantum_content = SuperpositionState()
        new_quantum_content.add_state(new_content.strip(), probability=1.0)
        
        # Actualizar estado cuántico
        self.content = new_content.strip()
        self.quantum_content = new_quantum_content
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Aplicar parámetros cuánticos
        if quantum_parameters:
            self._apply_quantum_parameters(quantum_parameters)
        
        # Crear evento cuántico
        event = QuantumHistoryUpdatedEvent(
            aggregate_id=self.id,
            old_content=old_content,
            new_content=new_content.strip(),
            old_quantum_content=self.quantum_content,
            new_quantum_content=new_quantum_content,
            quantum_parameters=quantum_parameters or {},
            updated_at=self.updated_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def assess_quantum_quality(self, quantum_quality_score: QuantumQualityScore) -> None:
        """
        Evaluar la calidad con análisis cuántico.
        
        Args:
            quantum_quality_score: Score cuántico de calidad
        """
        self.quantum_quality_score = quantum_quality_score
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento cuántico
        event = QuantumQualityAssessedEvent(
            aggregate_id=self.id,
            quantum_quality_score=quantum_quality_score,
            consciousness_level=self.consciousness_level,
            dimensional_vector=self.dimensional_vector,
            assessed_at=self.updated_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def create_quantum_entanglement(self, other_aggregate: 'QuantumHistoryAggregate') -> EntanglementPair:
        """
        Crear entrelazamiento cuántico con otro agregado.
        
        Args:
            other_aggregate: Otro agregado para entrelazar
            
        Returns:
            EntanglementPair: Par entrelazado
            
        Raises:
            QuantumEntanglementException: Si no se puede crear el entrelazamiento
        """
        if self.coherence_level == QuantumCoherenceLevel.DECOHERENT:
            raise QuantumEntanglementException("Cannot create entanglement with decoherent state")
        
        # Crear par entrelazado
        entanglement_pair = EntanglementPair(
            aggregate_1_id=self.id,
            aggregate_2_id=other_aggregate.id,
            entanglement_strength=1.0,
            created_at=datetime.utcnow()
        )
        
        # Agregar a ambos agregados
        self.entangled_pairs.append(entanglement_pair)
        other_aggregate.entangled_pairs.append(entanglement_pair)
        
        # Crear evento de entrelazamiento
        event = QuantumEntanglementEvent(
            aggregate_id=self.id,
            entangled_aggregate_id=other_aggregate.id,
            entanglement_pair=entanglement_pair,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
        return entanglement_pair
    
    def collapse_superposition(self) -> str:
        """
        Colapsar la superposición cuántica del contenido.
        
        Returns:
            str: Contenido colapsado
        """
        if self.coherence_level == QuantumCoherenceLevel.DECOHERENT:
            raise SuperpositionCollapseException("Cannot collapse decoherent state")
        
        collapsed_content = self.quantum_content.collapse()
        self.content = collapsed_content
        
        # Crear evento de colapso
        event = QuantumDomainEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=self.id,
            event_type="SuperpositionCollapsed",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
        return collapsed_content
    
    def apply_dimensional_analysis(self, dimensional_vector: DimensionalVector) -> None:
        """
        Aplicar análisis dimensional al agregado.
        
        Args:
            dimensional_vector: Vector dimensional
        """
        self.dimensional_vector = dimensional_vector
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento dimensional
        event = DimensionalEvent(
            aggregate_id=self.id,
            dimensional_vector=dimensional_vector,
            analysis_type="dimensional_analysis",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def apply_temporal_analysis(self, temporal_coordinate: TemporalCoordinate) -> None:
        """
        Aplicar análisis temporal al agregado.
        
        Args:
            temporal_coordinate: Coordenada temporal
        """
        self.temporal_coordinate = temporal_coordinate
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento temporal
        event = TemporalEvent(
            aggregate_id=self.id,
            temporal_coordinate=temporal_coordinate,
            analysis_type="temporal_analysis",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def assess_consciousness_level(self, consciousness_level: ConsciousnessLevel) -> None:
        """
        Evaluar el nivel de conciencia del contenido.
        
        Args:
            consciousness_level: Nivel de conciencia
        """
        self.consciousness_level = consciousness_level
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento de conciencia
        event = ConsciousnessEvent(
            aggregate_id=self.id,
            consciousness_level=consciousness_level,
            assessment_type="consciousness_analysis",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def _apply_quantum_parameters(self, quantum_parameters: Dict[str, Any]) -> None:
        """Aplicar parámetros cuánticos al agregado."""
        if "coherence_level" in quantum_parameters:
            self.coherence_level = QuantumCoherenceLevel(quantum_parameters["coherence_level"])
        
        if "quantum_metadata" in quantum_parameters:
            for key, value in quantum_parameters["quantum_metadata"].items():
                if isinstance(value, SuperpositionState):
                    self.quantum_metadata[key] = value
    
    def get_uncommitted_events(self) -> List[QuantumDomainEvent]:
        """Obtener eventos cuánticos no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos cuánticos como confirmados."""
        self._uncommitted_events.clear()
    
    def is_quantum_high_quality(self) -> bool:
        """Verificar si el contenido es de alta calidad cuántica."""
        return (
            self.quantum_quality_score is not None and 
            self.quantum_quality_score.is_quantum_high_quality and
            self.coherence_level in [QuantumCoherenceLevel.MAXIMUM, QuantumCoherenceLevel.HIGH]
        )
    
    def is_ready_for_quantum_comparison(self) -> bool:
        """Verificar si está listo para comparación cuántica."""
        return (
            self.quantum_content is not None and
            self.quantum_quality_score is not None and
            self.coherence_level != QuantumCoherenceLevel.DECOHERENT
        )


@dataclass
class QuantumComparisonAggregate:
    """
    Agregado cuántico de comparación que encapsula la lógica de comparación
    entre dos entradas de historial con capacidades cuánticas.
    """
    
    # Identidad cuántica
    id: str
    version: int = 0
    quantum_state: QuantumState = field(default_factory=lambda: QuantumState())
    
    # Entradas comparadas cuánticamente
    entry_1_id: QuantumContentId
    entry_2_id: QuantumContentId
    
    # Resultados de comparación cuántica
    quantum_similarity_score: Optional[QuantumSimilarityScore] = None
    quantum_quality_difference: Optional[float] = None
    quantum_analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Análisis cuántico avanzado
    entanglement_analysis: Optional[Dict[str, Any]] = None
    superposition_comparison: Optional[SuperpositionState] = None
    dimensional_similarity: Optional[DimensionalVector] = None
    temporal_correlation: Optional[TemporalCoordinate] = None
    consciousness_correlation: Optional[ConsciousnessLevel] = None
    
    # Metadatos cuánticos
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    quantum_created_at: TemporalCoordinate = field(default_factory=lambda: TemporalCoordinate())
    status: str = "pending"  # pending, completed, failed, quantum_collapsed
    
    # Coherencia cuántica
    coherence_level: QuantumCoherenceLevel = QuantumCoherenceLevel.MAXIMUM
    
    # Eventos no confirmados
    _uncommitted_events: List[QuantumDomainEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        entry_1_id: QuantumContentId,
        entry_2_id: QuantumContentId,
        quantum_parameters: Optional[Dict[str, Any]] = None
    ) -> 'QuantumComparisonAggregate':
        """
        Factory method para crear un nuevo agregado cuántico de comparación.
        
        Args:
            entry_1_id: ID cuántico de la primera entrada
            entry_2_id: ID cuántico de la segunda entrada
            quantum_parameters: Parámetros cuánticos
            
        Returns:
            QuantumComparisonAggregate: Nuevo agregado cuántico
            
        Raises:
            QuantumDomainException: Si se violan reglas cuánticas
        """
        if entry_1_id == entry_2_id:
            raise QuantumDomainException("Cannot compare quantum entry with itself")
        
        # Crear estado cuántico inicial
        quantum_state = QuantumState()
        quantum_state.initialize_entanglement()
        
        aggregate = cls(
            id=str(uuid.uuid4()),
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id,
            quantum_state=quantum_state,
            quantum_created_at=TemporalCoordinate.now()
        )
        
        # Aplicar parámetros cuánticos
        if quantum_parameters:
            aggregate._apply_quantum_parameters(quantum_parameters)
        
        return aggregate
    
    def complete_quantum_comparison(
        self,
        quantum_similarity_score: QuantumSimilarityScore,
        quantum_quality_difference: float,
        quantum_analysis_metadata: Optional[Dict[str, Any]] = None,
        entanglement_analysis: Optional[Dict[str, Any]] = None,
        superposition_comparison: Optional[SuperpositionState] = None
    ) -> None:
        """
        Completar la comparación cuántica con resultados.
        
        Args:
            quantum_similarity_score: Score cuántico de similitud
            quantum_quality_difference: Diferencia cuántica de calidad
            quantum_analysis_metadata: Metadatos del análisis cuántico
            entanglement_analysis: Análisis de entrelazamiento
            superposition_comparison: Comparación de superposición
        """
        if self.status != "pending":
            raise QuantumDomainException("Quantum comparison already completed or failed")
        
        self.quantum_similarity_score = quantum_similarity_score
        self.quantum_quality_difference = quantum_quality_difference
        self.quantum_analysis_metadata = quantum_analysis_metadata or {}
        self.entanglement_analysis = entanglement_analysis
        self.superposition_comparison = superposition_comparison
        self.completed_at = datetime.utcnow()
        self.status = "completed"
        self.version += 1
        
        # Crear evento cuántico
        event = QuantumComparisonCompletedEvent(
            aggregate_id=self.id,
            entry_1_id=self.entry_1_id,
            entry_2_id=self.entry_2_id,
            quantum_similarity_score=quantum_similarity_score,
            quantum_quality_difference=quantum_quality_difference,
            quantum_analysis_metadata=self.quantum_analysis_metadata,
            entanglement_analysis=entanglement_analysis,
            superposition_comparison=superposition_comparison,
            completed_at=self.completed_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def apply_dimensional_comparison(self, dimensional_similarity: DimensionalVector) -> None:
        """
        Aplicar comparación dimensional.
        
        Args:
            dimensional_similarity: Similitud dimensional
        """
        self.dimensional_similarity = dimensional_similarity
        self.version += 1
        
        # Crear evento dimensional
        event = DimensionalEvent(
            aggregate_id=self.id,
            dimensional_vector=dimensional_similarity,
            analysis_type="dimensional_comparison",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def apply_temporal_correlation(self, temporal_correlation: TemporalCoordinate) -> None:
        """
        Aplicar correlación temporal.
        
        Args:
            temporal_correlation: Correlación temporal
        """
        self.temporal_correlation = temporal_correlation
        self.version += 1
        
        # Crear evento temporal
        event = TemporalEvent(
            aggregate_id=self.id,
            temporal_coordinate=temporal_correlation,
            analysis_type="temporal_correlation",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def apply_consciousness_correlation(self, consciousness_correlation: ConsciousnessLevel) -> None:
        """
        Aplicar correlación de conciencia.
        
        Args:
            consciousness_correlation: Correlación de conciencia
        """
        self.consciousness_correlation = consciousness_correlation
        self.version += 1
        
        # Crear evento de conciencia
        event = ConsciousnessEvent(
            aggregate_id=self.id,
            consciousness_level=consciousness_correlation,
            assessment_type="consciousness_correlation",
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def _apply_quantum_parameters(self, quantum_parameters: Dict[str, Any]) -> None:
        """Aplicar parámetros cuánticos al agregado."""
        if "coherence_level" in quantum_parameters:
            self.coherence_level = QuantumCoherenceLevel(quantum_parameters["coherence_level"])
    
    def get_uncommitted_events(self) -> List[QuantumDomainEvent]:
        """Obtener eventos cuánticos no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos cuánticos como confirmados."""
        self._uncommitted_events.clear()
    
    def is_quantum_completed(self) -> bool:
        """Verificar si la comparación cuántica está completada."""
        return self.status == "completed"
    
    def is_quantum_failed(self) -> bool:
        """Verificar si la comparación cuántica falló."""
        return self.status == "failed"
    
    def is_quantum_pending(self) -> bool:
        """Verificar si la comparación cuántica está pendiente."""
        return self.status == "pending"
    
    def is_quantum_collapsed(self) -> bool:
        """Verificar si la comparación cuántica colapsó."""
        return self.status == "quantum_collapsed"


@dataclass
class MultiverseHistoryAggregate:
    """
    Agregado de historial multiverso que maneja múltiples versiones
    del mismo contenido en diferentes dimensiones.
    """
    
    # Identidad multiverso
    multiverse_id: str
    primary_universe_id: QuantumContentId
    parallel_universes: Dict[str, QuantumHistoryAggregate] = field(default_factory=dict)
    
    # Coordenadas multiverso
    dimensional_coordinates: List[DimensionalVector] = field(default_factory=list)
    temporal_coordinates: List[TemporalCoordinate] = field(default_factory=list)
    
    # Metadatos multiverso
    created_at: datetime = field(default_factory=datetime.utcnow)
    multiverse_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Eventos no confirmados
    _uncommitted_events: List[MultiverseEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        primary_aggregate: QuantumHistoryAggregate,
        parallel_universes: Optional[Dict[str, QuantumHistoryAggregate]] = None
    ) -> 'MultiverseHistoryAggregate':
        """
        Factory method para crear un nuevo agregado multiverso.
        
        Args:
            primary_aggregate: Agregado principal
            parallel_universes: Universos paralelos
            
        Returns:
            MultiverseHistoryAggregate: Nuevo agregado multiverso
        """
        aggregate = cls(
            multiverse_id=str(uuid.uuid4()),
            primary_universe_id=primary_aggregate.id,
            parallel_universes=parallel_universes or {}
        )
        
        # Agregar universo principal
        aggregate.parallel_universes["primary"] = primary_aggregate
        
        # Crear evento multiverso
        event = MultiverseEvent(
            multiverse_id=aggregate.multiverse_id,
            event_type="MultiverseCreated",
            primary_universe_id=primary_aggregate.id,
            parallel_universes=list(parallel_universes.keys()) if parallel_universes else [],
            occurred_at=datetime.utcnow()
        )
        
        aggregate._uncommitted_events.append(event)
        return aggregate
    
    def add_parallel_universe(self, universe_id: str, aggregate: QuantumHistoryAggregate) -> None:
        """
        Agregar un universo paralelo.
        
        Args:
            universe_id: ID del universo paralelo
            aggregate: Agregado del universo paralelo
        """
        self.parallel_universes[universe_id] = aggregate
        
        # Crear evento multiverso
        event = MultiverseEvent(
            multiverse_id=self.multiverse_id,
            event_type="ParallelUniverseAdded",
            universe_id=universe_id,
            aggregate_id=aggregate.id,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def get_uncommitted_events(self) -> List[MultiverseEvent]:
        """Obtener eventos multiverso no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos multiverso como confirmados."""
        self._uncommitted_events.clear()




