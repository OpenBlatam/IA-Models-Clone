"""
Domain Events - Eventos de Dominio
================================

Eventos de dominio que representan cambios importantes
en el estado del sistema.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .value_objects import ContentId, ModelType, QualityScore, SimilarityScore


@dataclass
class DomainEvent(ABC):
    """
    Evento base de dominio.
    
    Todos los eventos de dominio deben heredar de esta clase.
    """
    
    # Identidad del evento
    event_id: str
    aggregate_id: str
    event_type: str
    version: int = 1
    
    # Metadatos
    occurred_at: datetime
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    # Datos del evento
    event_data: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción."""
        if self.event_data is None:
            self.event_data = {}
    
    @abstractmethod
    def get_event_type(self) -> str:
        """Obtener el tipo de evento."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir evento a diccionario."""
        return {
            "event_id": self.event_id,
            "aggregate_id": self.aggregate_id,
            "event_type": self.event_type,
            "version": self.version,
            "occurred_at": self.occurred_at.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "event_data": self.event_data
        }


@dataclass
class HistoryCreatedEvent(DomainEvent):
    """
    Evento que se dispara cuando se crea una nueva entrada de historial.
    """
    
    model_type: ModelType
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "HistoryCreated"
        self.event_data.update({
            "model_type": self.model_type,
            "content": self.content,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata or {}
        })
    
    def get_event_type(self) -> str:
        return "HistoryCreated"


@dataclass
class HistoryUpdatedEvent(DomainEvent):
    """
    Evento que se dispara cuando se actualiza una entrada de historial.
    """
    
    old_content: str
    new_content: str
    updated_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "HistoryUpdated"
        self.event_data.update({
            "old_content": self.old_content,
            "new_content": self.new_content,
            "updated_at": self.updated_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "HistoryUpdated"


@dataclass
class HistoryDeletedEvent(DomainEvent):
    """
    Evento que se dispara cuando se elimina una entrada de historial.
    """
    
    deleted_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "HistoryDeleted"
        self.event_data.update({
            "deleted_at": self.deleted_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "HistoryDeleted"


@dataclass
class ComparisonCompletedEvent(DomainEvent):
    """
    Evento que se dispara cuando se completa una comparación.
    """
    
    entry_1_id: ContentId
    entry_2_id: ContentId
    similarity_score: SimilarityScore
    quality_difference: float
    analysis_metadata: Dict[str, Any]
    completed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "ComparisonCompleted"
        self.event_data.update({
            "entry_1_id": str(self.entry_1_id),
            "entry_2_id": str(self.entry_2_id),
            "similarity_score": self.similarity_score.to_dict(),
            "quality_difference": self.quality_difference,
            "analysis_metadata": self.analysis_metadata,
            "completed_at": self.completed_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "ComparisonCompleted"


@dataclass
class QualityAssessedEvent(DomainEvent):
    """
    Evento que se dispara cuando se evalúa la calidad de una entrada.
    """
    
    entry_id: ContentId
    quality_score: QualityScore
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]
    assessed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "QualityAssessed"
        self.event_data.update({
            "entry_id": str(self.entry_id),
            "quality_score": self.quality_score.to_dict(),
            "recommendations": self.recommendations,
            "detailed_analysis": self.detailed_analysis,
            "assessed_at": self.assessed_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "QualityAssessed"


@dataclass
class AnalysisCompletedEvent(DomainEvent):
    """
    Evento que se dispara cuando se completa un análisis en lote.
    """
    
    analysis_type: str
    total_entries: int
    processed_entries: int
    failed_entries: int
    results: Dict[str, Any]
    completed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "AnalysisCompleted"
        self.event_data.update({
            "analysis_type": self.analysis_type,
            "total_entries": self.total_entries,
            "processed_entries": self.processed_entries,
            "failed_entries": self.failed_entries,
            "results": self.results,
            "completed_at": self.completed_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "AnalysisCompleted"


@dataclass
class SystemHealthChangedEvent(DomainEvent):
    """
    Evento que se dispara cuando cambia el estado de salud del sistema.
    """
    
    service_name: str
    old_status: str
    new_status: str
    health_details: Dict[str, Any]
    changed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "SystemHealthChanged"
        self.event_data.update({
            "service_name": self.service_name,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "health_details": self.health_details,
            "changed_at": self.changed_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "SystemHealthChanged"


@dataclass
class PluginLoadedEvent(DomainEvent):
    """
    Evento que se dispara cuando se carga un plugin.
    """
    
    plugin_name: str
    plugin_version: str
    plugin_type: str
    loaded_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "PluginLoaded"
        self.event_data.update({
            "plugin_name": self.plugin_name,
            "plugin_version": self.plugin_version,
            "plugin_type": self.plugin_type,
            "loaded_at": self.loaded_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "PluginLoaded"


@dataclass
class CircuitBreakerOpenedEvent(DomainEvent):
    """
    Evento que se dispara cuando se abre un circuit breaker.
    """
    
    service_name: str
    failure_count: int
    failure_threshold: int
    opened_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "CircuitBreakerOpened"
        self.event_data.update({
            "service_name": self.service_name,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "opened_at": self.opened_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "CircuitBreakerOpened"


@dataclass
class CircuitBreakerClosedEvent(DomainEvent):
    """
    Evento que se dispara cuando se cierra un circuit breaker.
    """
    
    service_name: str
    recovery_time: float
    closed_at: datetime
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "CircuitBreakerClosed"
        self.event_data.update({
            "service_name": self.service_name,
            "recovery_time": self.recovery_time,
            "closed_at": self.closed_at.isoformat()
        })
    
    def get_event_type(self) -> str:
        return "CircuitBreakerClosed"


# Factory para crear eventos
class EventFactory:
    """
    Factory para crear eventos de dominio.
    """
    
    @staticmethod
    def create_history_created_event(
        aggregate_id: str,
        model_type: ModelType,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> HistoryCreatedEvent:
        """Crear evento de historial creado."""
        return HistoryCreatedEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=aggregate_id,
            model_type=model_type,
            content=content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            occurred_at=datetime.utcnow(),
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    @staticmethod
    def create_comparison_completed_event(
        aggregate_id: str,
        entry_1_id: ContentId,
        entry_2_id: ContentId,
        similarity_score: SimilarityScore,
        quality_difference: float,
        analysis_metadata: Dict[str, Any],
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> ComparisonCompletedEvent:
        """Crear evento de comparación completada."""
        return ComparisonCompletedEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=aggregate_id,
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id,
            similarity_score=similarity_score,
            quality_difference=quality_difference,
            analysis_metadata=analysis_metadata,
            completed_at=datetime.utcnow(),
            occurred_at=datetime.utcnow(),
            correlation_id=correlation_id,
            causation_id=causation_id
        )




