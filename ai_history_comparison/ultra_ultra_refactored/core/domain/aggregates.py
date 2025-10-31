"""
Domain Aggregates - Agregados de Dominio
======================================

Agregados de dominio que encapsulan la lógica de negocio
y mantienen la consistencia de los datos.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from .events import (
    DomainEvent,
    HistoryCreatedEvent,
    HistoryUpdatedEvent,
    HistoryDeletedEvent,
    ComparisonCompletedEvent,
    QualityAssessedEvent
)
from .value_objects import (
    ContentId,
    ModelType,
    QualityScore,
    SimilarityScore,
    ContentMetrics,
    SentimentAnalysis,
    TextComplexity
)
from .exceptions import (
    DomainException,
    InvalidStateException,
    BusinessRuleViolationException
)


@dataclass
class HistoryAggregate:
    """
    Agregado de historial que encapsula una entrada de historial
    y su lógica de negocio asociada.
    """
    
    # Identidad
    id: ContentId
    version: int = 0
    
    # Atributos
    model_type: ModelType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[QualityScore] = None
    content_metrics: Optional[ContentMetrics] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    text_complexity: Optional[TextComplexity] = None
    
    # Metadatos
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Eventos no confirmados
    _uncommitted_events: List[DomainEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        model_type: ModelType,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'HistoryAggregate':
        """
        Factory method para crear un nuevo agregado de historial.
        
        Args:
            model_type: Tipo de modelo de IA
            content: Contenido generado
            user_id: ID del usuario
            session_id: ID de sesión
            metadata: Metadatos adicionales
            
        Returns:
            HistoryAggregate: Nuevo agregado
            
        Raises:
            BusinessRuleViolationException: Si se violan reglas de negocio
        """
        # Validar contenido
        if not content or not content.strip():
            raise BusinessRuleViolationException("Content cannot be empty")
        
        if len(content) > 50000:
            raise BusinessRuleViolationException("Content too long")
        
        # Crear agregado
        aggregate = cls(
            id=ContentId(str(uuid.uuid4())),
            model_type=model_type,
            content=content.strip(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Crear evento de dominio
        event = HistoryCreatedEvent(
            aggregate_id=aggregate.id,
            model_type=model_type,
            content=content.strip(),
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            occurred_at=datetime.utcnow()
        )
        
        aggregate._uncommitted_events.append(event)
        return aggregate
    
    def update_content(self, new_content: str) -> None:
        """
        Actualizar el contenido del historial.
        
        Args:
            new_content: Nuevo contenido
            
        Raises:
            BusinessRuleViolationException: Si se violan reglas de negocio
        """
        if not new_content or not new_content.strip():
            raise BusinessRuleViolationException("Content cannot be empty")
        
        if len(new_content) > 50000:
            raise BusinessRuleViolationException("Content too long")
        
        old_content = self.content
        self.content = new_content.strip()
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento
        event = HistoryUpdatedEvent(
            aggregate_id=self.id,
            old_content=old_content,
            new_content=new_content.strip(),
            updated_at=self.updated_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def assess_quality(self, quality_score: QualityScore) -> None:
        """
        Evaluar la calidad del contenido.
        
        Args:
            quality_score: Score de calidad
        """
        self.quality_score = quality_score
        self.updated_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento
        event = QualityAssessedEvent(
            aggregate_id=self.id,
            quality_score=quality_score,
            assessed_at=self.updated_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def analyze_content(
        self,
        content_metrics: ContentMetrics,
        sentiment_analysis: Optional[SentimentAnalysis] = None,
        text_complexity: Optional[TextComplexity] = None
    ) -> None:
        """
        Analizar el contenido y actualizar métricas.
        
        Args:
            content_metrics: Métricas de contenido
            sentiment_analysis: Análisis de sentimiento
            text_complexity: Complejidad del texto
        """
        self.content_metrics = content_metrics
        self.sentiment_analysis = sentiment_analysis
        self.text_complexity = text_complexity
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def delete(self) -> None:
        """
        Marcar el agregado como eliminado.
        """
        # Crear evento
        event = HistoryDeletedEvent(
            aggregate_id=self.id,
            deleted_at=datetime.utcnow(),
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Obtener eventos no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos como confirmados."""
        self._uncommitted_events.clear()
    
    def is_high_quality(self) -> bool:
        """Verificar si el contenido es de alta calidad."""
        return self.quality_score is not None and self.quality_score.is_high_quality
    
    def is_ready_for_comparison(self) -> bool:
        """Verificar si está listo para comparación."""
        return (
            self.content_metrics is not None and
            self.quality_score is not None
        )


@dataclass
class ComparisonAggregate:
    """
    Agregado de comparación que encapsula la lógica de comparación
    entre dos entradas de historial.
    """
    
    # Identidad
    id: str
    version: int = 0
    
    # Entradas comparadas
    entry_1_id: ContentId
    entry_2_id: ContentId
    
    # Resultados de comparación
    similarity_score: Optional[SimilarityScore] = None
    quality_difference: Optional[float] = None
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Metadatos
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, completed, failed
    
    # Eventos no confirmados
    _uncommitted_events: List[DomainEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        entry_1_id: ContentId,
        entry_2_id: ContentId
    ) -> 'ComparisonAggregate':
        """
        Factory method para crear un nuevo agregado de comparación.
        
        Args:
            entry_1_id: ID de la primera entrada
            entry_2_id: ID de la segunda entrada
            
        Returns:
            ComparisonAggregate: Nuevo agregado
            
        Raises:
            BusinessRuleViolationException: Si se violan reglas de negocio
        """
        if entry_1_id == entry_2_id:
            raise BusinessRuleViolationException("Cannot compare entry with itself")
        
        aggregate = cls(
            id=str(uuid.uuid4()),
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id
        )
        
        return aggregate
    
    def complete_comparison(
        self,
        similarity_score: SimilarityScore,
        quality_difference: float,
        analysis_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Completar la comparación con resultados.
        
        Args:
            similarity_score: Score de similitud
            quality_difference: Diferencia de calidad
            analysis_metadata: Metadatos del análisis
        """
        if self.status != "pending":
            raise InvalidStateException("Comparison already completed or failed")
        
        self.similarity_score = similarity_score
        self.quality_difference = quality_difference
        self.analysis_metadata = analysis_metadata or {}
        self.completed_at = datetime.utcnow()
        self.status = "completed"
        self.version += 1
        
        # Crear evento
        event = ComparisonCompletedEvent(
            aggregate_id=self.id,
            entry_1_id=self.entry_1_id,
            entry_2_id=self.entry_2_id,
            similarity_score=similarity_score,
            quality_difference=quality_difference,
            analysis_metadata=self.analysis_metadata,
            completed_at=self.completed_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def fail_comparison(self, error_message: str) -> None:
        """
        Marcar la comparación como fallida.
        
        Args:
            error_message: Mensaje de error
        """
        if self.status != "pending":
            raise InvalidStateException("Comparison already completed or failed")
        
        self.status = "failed"
        self.analysis_metadata["error"] = error_message
        self.completed_at = datetime.utcnow()
        self.version += 1
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Obtener eventos no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos como confirmados."""
        self._uncommitted_events.clear()
    
    def is_completed(self) -> bool:
        """Verificar si la comparación está completada."""
        return self.status == "completed"
    
    def is_failed(self) -> bool:
        """Verificar si la comparación falló."""
        return self.status == "failed"
    
    def is_pending(self) -> bool:
        """Verificar si la comparación está pendiente."""
        return self.status == "pending"


@dataclass
class QualityAggregate:
    """
    Agregado de calidad que encapsula la evaluación de calidad
    de una entrada de historial.
    """
    
    # Identidad
    id: str
    entry_id: ContentId
    version: int = 0
    
    # Evaluación de calidad
    quality_score: QualityScore
    recommendations: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Metadatos
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessor_version: str = "1.0.0"
    
    # Eventos no confirmados
    _uncommitted_events: List[DomainEvent] = field(default_factory=list)
    
    @classmethod
    def create(
        cls,
        entry_id: ContentId,
        quality_score: QualityScore,
        recommendations: Optional[List[str]] = None,
        detailed_analysis: Optional[Dict[str, Any]] = None
    ) -> 'QualityAggregate':
        """
        Factory method para crear un nuevo agregado de calidad.
        
        Args:
            entry_id: ID de la entrada evaluada
            quality_score: Score de calidad
            recommendations: Recomendaciones
            detailed_analysis: Análisis detallado
            
        Returns:
            QualityAggregate: Nuevo agregado
        """
        aggregate = cls(
            id=str(uuid.uuid4()),
            entry_id=entry_id,
            quality_score=quality_score,
            recommendations=recommendations or [],
            detailed_analysis=detailed_analysis or {}
        )
        
        # Crear evento
        event = QualityAssessedEvent(
            aggregate_id=aggregate.id,
            entry_id=entry_id,
            quality_score=quality_score,
            recommendations=aggregate.recommendations,
            detailed_analysis=aggregate.detailed_analysis,
            assessed_at=aggregate.assessed_at,
            occurred_at=datetime.utcnow()
        )
        
        aggregate._uncommitted_events.append(event)
        return aggregate
    
    def update_assessment(
        self,
        quality_score: QualityScore,
        recommendations: Optional[List[str]] = None,
        detailed_analysis: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Actualizar la evaluación de calidad.
        
        Args:
            quality_score: Nuevo score de calidad
            recommendations: Nuevas recomendaciones
            detailed_analysis: Nuevo análisis detallado
        """
        self.quality_score = quality_score
        self.recommendations = recommendations or self.recommendations
        self.detailed_analysis = detailed_analysis or self.detailed_analysis
        self.assessed_at = datetime.utcnow()
        self.version += 1
        
        # Crear evento
        event = QualityAssessedEvent(
            aggregate_id=self.id,
            entry_id=self.entry_id,
            quality_score=quality_score,
            recommendations=self.recommendations,
            detailed_analysis=self.detailed_analysis,
            assessed_at=self.assessed_at,
            occurred_at=datetime.utcnow()
        )
        
        self._uncommitted_events.append(event)
    
    def get_uncommitted_events(self) -> List[DomainEvent]:
        """Obtener eventos no confirmados."""
        return self._uncommitted_events.copy()
    
    def mark_events_as_committed(self) -> None:
        """Marcar eventos como confirmados."""
        self._uncommitted_events.clear()
    
    def is_high_quality(self) -> bool:
        """Verificar si es de alta calidad."""
        return self.quality_score.is_high_quality
    
    def needs_improvement(self) -> bool:
        """Verificar si necesita mejoras."""
        return self.quality_score.needs_improvement




