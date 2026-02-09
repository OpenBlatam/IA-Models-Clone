from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import uuid
from dataclasses import dataclass
from ..models.facebook_models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 Facebook Posts - Domain Entities
===================================

Entidades del dominio core para Facebook posts siguiendo Clean Architecture.
Aggregate Root: FacebookPostEntity
"""


    ContentIdentifier, PostSpecification, GenerationConfig, FacebookPostContent,
    FacebookPostAnalysis, ContentStatus, PostType, ContentTone, TargetAudience,
    EngagementTier, QualityTier
)


class FacebookPostDomainEntity:
    """
    Entidad del dominio principal - Facebook Post (Aggregate Root).
    Encapsula la lógica de negocio y mantiene la integridad del agregado.
    """
    
    def __init__(
        self,
        identifier: ContentIdentifier,
        specification: PostSpecification,
        generation_config: GenerationConfig,
        content: FacebookPostContent,
        analysis: Optional[FacebookPostAnalysis] = None
    ):
        
    """__init__ function."""
# Validación de invariantes del dominio
        self._validate_domain_rules(identifier, specification, content)
        
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
        self._domain_events: List[Dict[str, Any]] = []
        
        # Registro de dominio
        self._register_domain_event("post_created", {
            "post_id": self._identifier.content_id,
            "topic": self._specification.topic,
            "target_audience": self._specification.target_audience.value
        })
    
    def _validate_domain_rules(
        self, 
        identifier: ContentIdentifier, 
        specification: PostSpecification, 
        content: FacebookPostContent
    ) -> None:
        """Validar reglas de negocio del dominio."""
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
    def domain_events(self) -> List[Dict[str, Any]]:
        """Eventos del dominio registrados."""
        return self._domain_events.copy()
    
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
        old_content_hash = self._content.get_character_count()
        self._content = new_content
        self._analysis = None  # Invalidate analysis
        self._status = ContentStatus.DRAFT  # Reset status
        self._updated_at = datetime.now()
        self._version += 1
        
        # Registrar evento del dominio
        self._register_domain_event("content_updated", {
            "post_id": self._identifier.content_id,
            "old_length": old_content_hash,
            "new_length": new_content.get_character_count(),
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
        self._register_domain_event("analysis_completed", {
            "post_id": self._identifier.content_id,
            "overall_score": overall_score,
            "quality_tier": analysis.quality_assessment.quality_tier.value,
            "old_status": old_status.value,
            "new_status": self._status.value,
            "confidence": analysis.confidence_level
        })
    
    def approve_for_publication(self, user_id: Optional[str] = None) -> None:
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
        
        self._register_domain_event("post_approved", {
            "post_id": self._identifier.content_id,
            "user_id": user_id,
            "old_status": old_status.value,
            "approval_score": self._analysis.get_overall_score() if self._analysis else 0
        })
    
    def publish(self, user_id: Optional[str] = None) -> None:
        """
        Publicar post con validación de reglas del dominio.
        
        Reglas de negocio:
        - Solo posts APPROVED o SCHEDULED pueden publicarse
        - Debe pasar todas las validaciones
        """
        if self._status not in [ContentStatus.APPROVED, ContentStatus.SCHEDULED]:
            raise DomainValidationError(f"Cannot publish post with status: {self._status.value}")
        
        validation_errors = self.validate_for_publication()
        if validation_errors:
            raise DomainValidationError(f"Cannot publish post: {'; '.join(validation_errors)}")
        
        old_status = self._status
        self._status = ContentStatus.PUBLISHED
        self._updated_at = datetime.now()
        
        self._register_domain_event("post_published", {
            "post_id": self._identifier.content_id,
            "user_id": user_id,
            "old_status": old_status.value,
            "publication_time": self._updated_at.isoformat()
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
        if self._analysis:
            return self._analysis.engagement_prediction.engagement_rate
        return 0.5  # Predicción neutral por defecto
    
    def get_quality_tier(self) -> str:
        """Obtener tier de calidad."""
        if self._analysis:
            return self._analysis.quality_assessment.quality_tier.value
        return "unassessed"
    
    def get_performance_indicators(self) -> Dict[str, Any]:
        """Obtener indicadores clave de performance."""
        indicators = {
            "post_id": self._identifier.content_id,
            "status": self._status.value,
            "quality_tier": self.get_quality_tier(),
            "engagement_prediction": self.get_engagement_prediction(),
            "ready_for_publication": self.is_ready_for_publication(),
            "content_length": len(self._content.text),
            "hashtags_count": len(self._content.hashtags),
            "version": self._version
        }
        
        if self._analysis:
            indicators.update({
                "overall_score": self._analysis.get_overall_score(),
                "virality_score": self._analysis.engagement_prediction.virality_score,
                "brand_alignment": self._analysis.quality_assessment.brand_alignment,
                "recommendations_count": len(self._analysis.get_actionable_recommendations())
            })
        
        return indicators
    
    # ===== DOMAIN EVENTS =====
    
    def _register_domain_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Registrar evento del dominio."""
        event = {
            "event_type": event_type,
            "aggregate_id": self._identifier.content_id,
            "timestamp": datetime.now().isoformat(),
            "version": self._version,
            "data": data
        }
        self._domain_events.append(event)
        
        # Mantener solo los últimos 20 eventos
        if len(self._domain_events) > 20:
            self._domain_events = self._domain_events[-20:]
    
    def clear_domain_events(self) -> List[Dict[str, Any]]:
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