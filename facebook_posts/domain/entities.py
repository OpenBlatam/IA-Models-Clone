from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..models.facebook_models import (
        import uuid
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 Facebook Posts - Domain Entities
===================================

Entidades del dominio core siguiendo Clean Architecture y DDD patterns.
"""

    ContentIdentifier, PostSpecification, GenerationConfig, 
    FacebookPostContent, FacebookPostAnalysis, ContentStatus,
    PostType, ContentTone, TargetAudience, EngagementTier
)


# ===== DOMAIN VALUE OBJECTS =====

@dataclass(frozen=True)
class PostMetrics:
    """Value object para métricas del post."""
    engagement_rate: float
    virality_score: float
    quality_score: float
    reach_prediction: int
    interaction_count: int
    
    def __post_init__(self) -> Any:
        if not 0 <= self.engagement_rate <= 1:
            raise ValueError("Engagement rate must be between 0 and 1")
        if not 0 <= self.virality_score <= 1:
            raise ValueError("Virality score must be between 0 and 1")


@dataclass(frozen=True)
class PublicationWindow:
    """Value object para ventana de publicación."""
    start_time: datetime
    end_time: datetime
    optimal_time: datetime
    timezone: str = "UTC"
    
    def __post_init__(self) -> Any:
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        if not (self.start_time <= self.optimal_time <= self.end_time):
            raise ValueError("Optimal time must be within the publication window")


# ===== DOMAIN SERVICES INTERFACES =====

class ContentValidationService(ABC):
    """Servicio de dominio para validación de contenido."""
    
    @abstractmethod
    def validate_content_quality(self, content: FacebookPostContent) -> List[str]:
        """Validar calidad del contenido."""
        pass
    
    @abstractmethod
    def validate_brand_compliance(self, content: FacebookPostContent, brand_guidelines: Dict[str, Any]) -> List[str]:
        """Validar cumplimiento de marca."""
        pass


class EngagementPredictionService(ABC):
    """Servicio de dominio para predicción de engagement."""
    
    @abstractmethod
    async def predict_engagement(self, post: 'FacebookPostDomainEntity') -> PostMetrics:
        """Predecir métricas de engagement."""
        pass
    
    @abstractmethod
    async def calculate_optimal_timing(self, specification: PostSpecification) -> PublicationWindow:
        """Calcular timing óptimo de publicación."""
        pass


# ===== DOMAIN EXCEPTIONS =====

class FacebookPostDomainError(Exception):
    """Error base del dominio Facebook Posts."""
    pass


class DomainValidationError(FacebookPostDomainError):
    """Error de validación de reglas del dominio."""
    pass


class DomainStateError(FacebookPostDomainError):
    """Error de estado inválido del dominio."""
    pass


class ContentQualityError(FacebookPostDomainError):
    """Error de calidad de contenido."""
    pass


# ===== DOMAIN EVENTS =====

@dataclass(frozen=True)
class DomainEvent:
    """Evento base del dominio."""
    event_id: str
    aggregate_id: str
    event_type: str
    occurred_at: datetime
    data: Dict[str, Any]
    version: int


@dataclass(frozen=True)
class PostCreatedEvent(DomainEvent):
    """Evento: Post creado."""
    pass


@dataclass(frozen=True)
class PostAnalyzedEvent(DomainEvent):
    """Evento: Post analizado."""
    pass


@dataclass(frozen=True)
class PostApprovedEvent(DomainEvent):
    """Evento: Post aprobado."""
    pass


@dataclass(frozen=True)
class PostPublishedEvent(DomainEvent):
    """Evento: Post publicado."""
    pass


# ===== MAIN DOMAIN ENTITY =====

class FacebookPostDomainEntity:
    """Entidad del dominio - Facebook Post (Aggregate Root)."""
    
    def __init__(
        self,
        identifier: ContentIdentifier,
        specification: PostSpecification,
        generation_config: GenerationConfig,
        content: FacebookPostContent,
        analysis: Optional[FacebookPostAnalysis] = None
    ):
        
    """__init__ function."""
# Validar invariantes del dominio
        self._validate_domain_invariants(identifier, specification, content)
        
        # Estado inmutable de identidad
        self._identifier = identifier
        self._specification = specification
        self._generation_config = generation_config
        
        # Estado mutable del agregado
        self._content = content
        self._analysis = analysis
        self._status = ContentStatus.DRAFT
        
        # Metadatos del dominio
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._version = 1
        self._domain_events: List[DomainEvent] = []
        
        # Métricas y metadata
        self._performance_metrics: Optional[PostMetrics] = None
        self._publication_window: Optional[PublicationWindow] = None
        self._approval_history: List[Dict[str, Any]] = []
        
        # Registro de evento de creación
        self._add_domain_event("post_created", {
            "post_id": self._identifier.content_id,
            "topic": self._specification.topic,
            "target_audience": self._specification.target_audience.value,
            "content_length": len(self._content.text)
        })
    
    def _validate_domain_invariants(
        self, 
        identifier: ContentIdentifier, 
        specification: PostSpecification, 
        content: FacebookPostContent
    ) -> None:
        """Validar invariantes del dominio."""
        if not identifier:
            raise DomainValidationError("Content identifier is required")
        
        if not specification or not specification.topic.strip():
            raise DomainValidationError("Post specification with topic is required")
        
        if not content or not content.text.strip():
            raise DomainValidationError("Post content with text is required")
        
        # Regla de negocio: contenido no puede exceder límites de Facebook
        if len(content.get_display_text()) > 2000:
            raise DomainValidationError("Content exceeds Facebook's 2000 character limit")
        
        # Regla de negocio: contenido debe tener mínimo engagement potencial
        if len(content.text.strip()) < 10:
            raise DomainValidationError("Content too short for meaningful engagement")
    
    # ===== PROPERTIES (READ-ONLY) =====
    
    @property
    def identifier(self) -> ContentIdentifier:
        """Identificador único inmutable."""
        return self._identifier
    
    @property
    def specification(self) -> PostSpecification:
        """Especificación inmutable del post."""
        return self._specification
    
    @property
    def generation_config(self) -> GenerationConfig:
        """Configuración de generación inmutable."""
        return self._generation_config
    
    @property
    def content(self) -> FacebookPostContent:
        """Contenido actual del post."""
        return self._content
    
    @property
    def analysis(self) -> Optional[FacebookPostAnalysis]:
        """Análisis actual del post."""
        return self._analysis
    
    @property
    def status(self) -> ContentStatus:
        """Estado actual del post."""
        return self._status
    
    @property
    def created_at(self) -> datetime:
        """Fecha de creación inmutable."""
        return self._created_at
    
    @property
    def updated_at(self) -> datetime:
        """Fecha de última actualización."""
        return self._updated_at
    
    @property
    def version(self) -> int:
        """Versión actual del post."""
        return self._version
    
    @property
    def domain_events(self) -> List[DomainEvent]:
        """Eventos del dominio registrados."""
        return self._domain_events.copy()
    
    @property
    def performance_metrics(self) -> Optional[PostMetrics]:
        """Métricas de performance."""
        return self._performance_metrics
    
    @property
    def publication_window(self) -> Optional[PublicationWindow]:
        """Ventana de publicación."""
        return self._publication_window
    
    # ===== DOMAIN OPERATIONS =====
    
    def update_content(self, new_content: FacebookPostContent) -> None:
        """
        Actualizar contenido del post con validación de reglas del dominio.
        
        Reglas de negocio:
        - El contenido debe cumplir límites de Facebook
        - Invalidar análisis previo
        - Resetear estado a DRAFT
        - Incrementar versión
        """
        # Validar reglas de dominio
        if len(new_content.get_display_text()) > 2000:
            raise DomainValidationError("New content exceeds Facebook's character limit")
        
        if len(new_content.text.strip()) < 10:
            raise DomainValidationError("New content too short for engagement")
        
        # Aplicar cambios
        old_content_length = len(self._content.text)
        self._content = new_content
        self._analysis = None  # Invalidate analysis
        self._status = ContentStatus.DRAFT  # Reset status
        self._updated_at = datetime.now()
        self._version += 1
        
        # Registrar evento del dominio
        self._add_domain_event("content_updated", {
            "post_id": self._identifier.content_id,
            "old_length": old_content_length,
            "new_length": len(new_content.text),
            "version": self._version
        })
    
    def set_analysis(self, analysis: FacebookPostAnalysis) -> None:
        """
        Establecer análisis con reglas de negocio automáticas.
        
        Reglas de negocio:
        - Analysis determina status automático
        - Score >= 0.8 → APPROVED
        - Score >= 0.6 → UNDER_REVIEW
        - Score < 0.6 → Mantener DRAFT
        """
        if not analysis:
            raise DomainValidationError("Analysis cannot be null")
        
        self._analysis = analysis
        self._updated_at = datetime.now()
        
        # Aplicar reglas de negocio para status automático
        overall_score = analysis.get_overall_score()
        old_status = self._status
        
        if overall_score >= 0.8:
            self._status = ContentStatus.APPROVED
        elif overall_score >= 0.6:
            self._status = ContentStatus.UNDER_REVIEW
        # Si < 0.6, mantener estado actual (probablemente DRAFT)
        
        # Registrar evento del dominio
        self._add_domain_event("post_analyzed", {
            "post_id": self._identifier.content_id,
            "overall_score": overall_score,
            "quality_tier": analysis.quality_assessment.quality_tier.value,
            "old_status": old_status.value,
            "new_status": self._status.value,
            "confidence": analysis.confidence_level
        })
    
    def approve_for_publication(self, approver_id: str, notes: Optional[str] = None) -> None:
        """
        Aprobar post para publicación con validación de reglas del dominio.
        
        Reglas de negocio:
        - Debe tener análisis válido
        - Score mínimo de 0.6
        - Confidence mínimo de 0.6
        - Contenido dentro de límites
        """
        validation_errors = self.validate_for_publication()
        if validation_errors:
            raise DomainValidationError(f"Cannot approve post: {'; '.join(validation_errors)}")
        
        old_status = self._status
        self._status = ContentStatus.APPROVED
        self._updated_at = datetime.now()
        
        # Registrar en historial de aprobaciones
        approval_record = {
            "approver_id": approver_id,
            "approved_at": self._updated_at.isoformat(),
            "notes": notes,
            "quality_score": self._analysis.get_overall_score() if self._analysis else None,
            "version": self._version
        }
        self._approval_history.append(approval_record)
        
        self._add_domain_event("post_approved", {
            "post_id": self._identifier.content_id,
            "approver_id": approver_id,
            "old_status": old_status.value,
            "approval_score": self._analysis.get_overall_score() if self._analysis else 0,
            "notes": notes
        })
    
    def publish(self, publisher_id: str) -> None:
        """
        Publicar post con validación de reglas del dominio.
        
        Reglas de negocio:
        - Solo posts APPROVED o SCHEDULED pueden publicarse
        - Debe pasar todas las validaciones
        """
        if self._status not in [ContentStatus.APPROVED, ContentStatus.SCHEDULED]:
            raise DomainStateError(f"Cannot publish post with status: {self._status.value}")
        
        validation_errors = self.validate_for_publication()
        if validation_errors:
            raise DomainValidationError(f"Cannot publish post: {'; '.join(validation_errors)}")
        
        old_status = self._status
        self._status = ContentStatus.PUBLISHED
        self._updated_at = datetime.now()
        
        self._add_domain_event("post_published", {
            "post_id": self._identifier.content_id,
            "publisher_id": publisher_id,
            "old_status": old_status.value,
            "publication_time": self._updated_at.isoformat()
        })
    
    def set_performance_metrics(self, metrics: PostMetrics) -> None:
        """Establecer métricas de performance."""
        self._performance_metrics = metrics
        self._updated_at = datetime.now()
        
        self._add_domain_event("metrics_updated", {
            "post_id": self._identifier.content_id,
            "engagement_rate": metrics.engagement_rate,
            "virality_score": metrics.virality_score,
            "quality_score": metrics.quality_score
        })
    
    def set_publication_window(self, window: PublicationWindow) -> None:
        """Establecer ventana de publicación."""
        self._publication_window = window
        self._updated_at = datetime.now()
        
        self._add_domain_event("publication_window_set", {
            "post_id": self._identifier.content_id,
            "optimal_time": window.optimal_time.isoformat(),
            "timezone": window.timezone
        })
    
    # ===== DOMAIN QUERIES =====
    
    def is_ready_for_publication(self) -> bool:
        """Verificar si cumple reglas para publicación."""
        return (
            self._status in [ContentStatus.APPROVED, ContentStatus.SCHEDULED] and
            self._analysis is not None and
            self._analysis.get_overall_score() >= 0.6 and
            len(self.validate_for_publication()) == 0
        )
    
    def validate_for_publication(self) -> List[str]:
        """Validar reglas de dominio para publicación."""
        errors = []
        
        # Validación de contenido
        if len(self._content.get_display_text()) > 2000:
            errors.append("Content exceeds Facebook's 2000 character limit")
        
        if len(self._content.text.strip()) < 10:
            errors.append("Content too short for meaningful engagement")
        
        # Validación de análisis
        if not self._analysis:
            errors.append("Content must be analyzed before publication")
        else:
            if self._analysis.get_overall_score() < 0.5:
                errors.append("Content quality score below minimum threshold (0.5)")
            
            if self._analysis.confidence_level < 0.6:
                errors.append("Analysis confidence too low for publication (< 0.6)")
        
        # Validación de estado
        if self._status == ContentStatus.REJECTED:
            errors.append("Rejected content cannot be published")
        
        return errors
    
    def get_engagement_prediction(self) -> float:
        """Obtener predicción de engagement."""
        if self._performance_metrics:
            return self._performance_metrics.engagement_rate
        elif self._analysis:
            return self._analysis.engagement_prediction.engagement_rate
        return 0.5  # Predicción neutral por defecto
    
    def get_quality_tier(self) -> str:
        """Obtener tier de calidad."""
        if self._analysis:
            return self._analysis.quality_assessment.quality_tier.value
        return "unassessed"
    
    def get_approval_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de aprobaciones."""
        return self._approval_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de performance."""
        summary = {
            "post_id": self._identifier.content_id,
            "status": self._status.value,
            "quality_tier": self.get_quality_tier(),
            "engagement_prediction": self.get_engagement_prediction(),
            "ready_for_publication": self.is_ready_for_publication(),
            "content_length": len(self._content.text),
            "hashtags_count": len(self._content.hashtags),
            "version": self._version,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat()
        }
        
        if self._analysis:
            summary.update({
                "overall_score": self._analysis.get_overall_score(),
                "virality_score": self._analysis.engagement_prediction.virality_score,
                "brand_alignment": self._analysis.quality_assessment.brand_alignment,
                "recommendations_count": len(self._analysis.get_actionable_recommendations())
            })
        
        if self._performance_metrics:
            summary.update({
                "actual_engagement": self._performance_metrics.engagement_rate,
                "actual_reach": self._performance_metrics.reach_prediction,
                "interaction_count": self._performance_metrics.interaction_count
            })
        
        return summary
    
    # ===== DOMAIN EVENTS =====
    
    def _add_domain_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Registrar evento del dominio."""
        
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            aggregate_id=self._identifier.content_id,
            event_type=event_type,
            occurred_at=datetime.now(),
            data=data,
            version=self._version
        )
        
        self._domain_events.append(event)
        
        # Mantener solo los últimos 20 eventos
        if len(self._domain_events) > 20:
            self._domain_events = self._domain_events[-20:]
    
    def clear_domain_events(self) -> List[DomainEvent]:
        """Limpiar y retornar eventos del dominio."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    # ===== EQUALITY & HASHING =====
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FacebookPostDomainEntity):
            return False
        return self._identifier.content_id == other._identifier.content_id
    
    def __hash__(self) -> int:
        return hash(self._identifier.content_id)
    
    def __str__(self) -> str:
        return f"FacebookPostDomain({self._identifier.content_id[:8]}...)"
    
    def __repr__(self) -> str:
        return (
            f"FacebookPostDomainEntity("
            f"id={self._identifier.content_id}, "
            f"topic={self._specification.topic}, "
            f"status={self._status.value}, "
            f"quality={self.get_quality_tier()})"
        )


# ===== DOMAIN FACTORY =====

class FacebookPostDomainFactory:
    """Factory para crear entidades del dominio."""
    
    @staticmethod
    def create_new_post(
        topic: str,
        content_text: str,
        post_type: PostType = PostType.TEXT,
        tone: ContentTone = ContentTone.CASUAL,
        target_audience: TargetAudience = TargetAudience.GENERAL,
        keywords: Optional[List[str]] = None,
        hashtags: Optional[List[str]] = None,
        **config_kwargs
    ) -> FacebookPostDomainEntity:
        """Crear nueva entidad de dominio con validación completa."""
        
        # Crear identificador único
        identifier = ContentIdentifier.generate(content_text, {
            "topic": topic,
            "audience": target_audience.value
        })
        
        # Crear especificación
        specification = PostSpecification(
            topic=topic,
            post_type=post_type,
            tone=tone,
            target_audience=target_audience,
            keywords=keywords or [topic.lower()],
            target_engagement=config_kwargs.get('target_engagement', EngagementTier.HIGH)
        )
        
        # Crear configuración de generación
        generation_config = GenerationConfig(
            max_length=config_kwargs.get('max_length', 280),
            include_hashtags=config_kwargs.get('include_hashtags', True),
            include_emojis=config_kwargs.get('include_emojis', True),
            include_call_to_action=config_kwargs.get('include_call_to_action', True),
            brand_voice=config_kwargs.get('brand_voice'),
            campaign_context=config_kwargs.get('campaign_context')
        )
        
        # Crear contenido
        content = FacebookPostContent(
            text=content_text,
            hashtags=hashtags or [],
            mentions=config_kwargs.get('mentions', []),
            media_urls=config_kwargs.get('media_urls', []),
            link_url=config_kwargs.get('link_url'),
            call_to_action=config_kwargs.get('call_to_action')
        )
        
        # Crear y retornar entidad del dominio
        return FacebookPostDomainEntity(
            identifier=identifier,
            specification=specification,
            generation_config=generation_config,
            content=content
        )
    
    @staticmethod
    def create_high_performance_post(
        topic: str,
        target_audience: TargetAudience = TargetAudience.GENERAL
    ) -> FacebookPostDomainEntity:
        """Crear post optimizado para alto rendimiento."""
        
        # Contenido optimizado para engagement
        optimized_content = f"✨ Transform your {topic} strategy today! 🚀\n\nDiscover proven techniques that deliver real results. What's been your biggest challenge?\n\n👇 Share your experience below!"
        
        # Hashtags optimizados
        base_hashtag = topic.lower().replace(' ', '').replace('-', '')
        optimized_hashtags = [
            base_hashtag,
            'success',
            'growth',
            'transformation',
            'results'
        ]
        
        return FacebookPostDomainFactory.create_new_post(
            topic=topic,
            content_text=optimized_content,
            post_type=PostType.TEXT,
            tone=ContentTone.INSPIRING,
            target_audience=target_audience,
            keywords=[topic.lower(), 'success', 'growth'],
            hashtags=optimized_hashtags,
            target_engagement=EngagementTier.VIRAL,
            max_length=500,
            include_call_to_action=True,
            call_to_action="Share your experience in the comments! 👇"
        ) 