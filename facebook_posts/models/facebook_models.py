from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import hashlib
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 Facebook Posts Models - Onyx Integration
==========================================

Modelos avanzados para el sistema de Facebook posts integrado con Onyx.
Clean Architecture + LangChain + Performance Optimizations.
"""



# ===== ENUMS =====

class PostType(str, Enum):
    """Tipos de posts de Facebook."""
    TEXT = "text"
    IMAGE = "image" 
    VIDEO = "video"
    LINK = "link"
    CAROUSEL = "carousel"
    POLL = "poll"
    EVENT = "event"
    STORY = "story"


class ContentTone(str, Enum):
    """Tonos de comunicación."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    HUMOROUS = "humorous"
    INSPIRING = "inspiring"
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"
    CONTROVERSIAL = "controversial"


class TargetAudience(str, Enum):
    """Audiencias objetivo."""
    GENERAL = "general"
    YOUNG_ADULTS = "young_adults"
    PROFESSIONALS = "professionals"
    PARENTS = "parents"
    ENTREPRENEURS = "entrepreneurs"
    STUDENTS = "students"
    SENIORS = "seniors"
    CUSTOM = "custom"


class ContentStatus(str, Enum):
    """Estados del contenido."""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REJECTED = "rejected"


class EngagementTier(str, Enum):
    """Niveles de engagement objetivo."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VIRAL = "viral"


class QualityTier(str, Enum):
    """Niveles de calidad."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"
    PREMIUM = "premium"


# ===== CORE VALUE OBJECTS =====

@dataclass(frozen=True)
class ContentIdentifier:
    """Identificador único e inmutable del contenido."""
    content_id: str
    content_hash: str
    created_at: datetime
    fingerprint: str
    
    @classmethod
    def generate(cls, content_text: str, metadata: Optional[Dict[str, Any]] = None) -> 'ContentIdentifier':
        """Generar identificador único."""
        content_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]
        created_at = datetime.now()
        
        # Create fingerprint with metadata
        fingerprint_data = f"{content_text}{metadata or {}}{created_at.isoformat()}"
        fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()[:12]
        
        return cls(
            content_id=content_id,
            content_hash=content_hash,
            created_at=created_at,
            fingerprint=fingerprint
        )


@dataclass
class PostSpecification:
    """Especificación para generación de post."""
    topic: str
    post_type: PostType
    tone: ContentTone
    target_audience: TargetAudience
    keywords: List[str]
    target_engagement: EngagementTier = EngagementTier.HIGH
    
    def __post_init__(self) -> Any:
        if not self.topic.strip():
            raise ValueError("Topic cannot be empty")
        if len(self.topic) > 200:
            raise ValueError("Topic too long (max 200 chars)")


@dataclass
class GenerationConfig:
    """Configuración de generación."""
    max_length: int = 280
    include_hashtags: bool = True
    include_emojis: bool = True
    include_call_to_action: bool = True
    brand_voice: Optional[str] = None
    campaign_context: Optional[str] = None
    custom_instructions: Optional[str] = None
    
    def __post_init__(self) -> Any:
        if self.max_length < 50 or self.max_length > 2000:
            raise ValueError("max_length must be between 50 and 2000")


# ===== CONTENT MODELS =====

class FacebookPostContent(BaseModel):
    """Contenido del post."""
    text: str = Field(..., min_length=10, max_length=2000)
    hashtags: List[str] = Field(default_factory=list, max_items=30)
    mentions: List[str] = Field(default_factory=list, max_items=10)
    media_urls: List[str] = Field(default_factory=list, max_items=10)
    link_url: Optional[str] = None
    call_to_action: Optional[str] = None
    
    @validator('hashtags', each_item=True)
    def validate_hashtags(cls, v) -> bool:
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid hashtag format: {v}")
        return v.lower()
    
    @validator('text')
    def validate_text(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Text content cannot be empty")
        return v.strip()
    
    def get_display_text(self) -> str:
        """Obtener texto completo para mostrar."""
        text = self.text
        if self.hashtags:
            text += "\n\n" + " ".join(f"#{tag}" for tag in self.hashtags)
        return text
    
    def get_word_count(self) -> int:
        """Contar palabras."""
        return len(self.text.split())
    
    def get_character_count(self) -> int:
        """Contar caracteres del contenido completo."""
        return len(self.get_display_text())


# ===== ANALYSIS MODELS =====

class ContentMetrics(BaseModel):
    """Métricas de contenido."""
    character_count: int
    word_count: int
    hashtag_count: int
    mention_count: int
    emoji_count: int
    readability_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    
    
class EngagementPrediction(BaseModel):
    """Predicción de engagement."""
    engagement_rate: float = Field(ge=0.0, le=1.0)
    virality_score: float = Field(ge=0.0, le=1.0)
    predicted_likes: int = Field(ge=0)
    predicted_shares: int = Field(ge=0)
    predicted_comments: int = Field(ge=0)
    predicted_reach: int = Field(ge=0)
    confidence_level: float = Field(ge=0.0, le=1.0, default=0.8)


class QualityAssessment(BaseModel):
    """Evaluación de calidad."""
    overall_score: float = Field(ge=0.0, le=1.0)
    quality_tier: QualityTier
    brand_alignment: float = Field(ge=0.0, le=1.0)
    audience_relevance: float = Field(ge=0.0, le=1.0)
    trend_alignment: float = Field(ge=0.0, le=1.0)
    clarity_score: float = Field(ge=0.0, le=1.0)
    
    # Detailed insights
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)


class FacebookPostAnalysis(BaseModel):
    """Análisis comprehensivo del post."""
    content_metrics: ContentMetrics
    engagement_prediction: EngagementPrediction
    quality_assessment: QualityAssessment
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    analysis_version: str = "3.0"
    processing_time_ms: float = Field(default=0.0)
    confidence_level: float = Field(ge=0.0, le=1.0, default=0.85)
    
    # Model information
    analysis_models_used: List[str] = Field(default_factory=list)
    onyx_model_id: Optional[str] = None
    langchain_chain_id: Optional[str] = None
    
    # Advanced insights
    optimal_posting_time: Optional[datetime] = None
    hashtag_suggestions: List[str] = Field(default_factory=list)
    similar_successful_posts: List[str] = Field(default_factory=list)
    competitive_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    def get_overall_score(self) -> float:
        """Score general ponderado optimizado."""
        weights = {
            'quality': 0.35,
            'engagement': 0.30,
            'virality': 0.20,
            'audience_relevance': 0.10,
            'trend_alignment': 0.05
        }
        
        return (
            self.quality_assessment.overall_score * weights['quality'] +
            self.engagement_prediction.engagement_rate * weights['engagement'] +
            self.engagement_prediction.virality_score * weights['virality'] +
            self.quality_assessment.audience_relevance * weights['audience_relevance'] +
            self.quality_assessment.trend_alignment * weights['trend_alignment']
        )
    
    def get_actionable_recommendations(self) -> List[str]:
        """Obtener recomendaciones accionables."""
        recommendations = []
        
        # Quality-based recommendations
        if self.quality_assessment.overall_score < 0.7:
            recommendations.extend(self.quality_assessment.improvement_suggestions)
        
        # Engagement-based recommendations
        if self.engagement_prediction.engagement_rate < 0.6:
            recommendations.append("Consider adding more engaging elements like questions or polls")
        
        # Hashtag recommendations
        if len(self.hashtag_suggestions) > 0:
            recommendations.append(f"Consider using hashtags: {', '.join(self.hashtag_suggestions[:3])}")
        
        return recommendations[:5]  # Top 5 recommendations


# ===== MAIN ENTITY =====

class FacebookPostEntity(BaseModel):
    """Entidad principal refactorizada para Onyx (Aggregate Root)."""
    
    # Core identity
    identifier: ContentIdentifier
    specification: PostSpecification
    generation_config: GenerationConfig
    content: FacebookPostContent
    
    # State management
    status: ContentStatus = ContentStatus.DRAFT
    analysis: Optional[FacebookPostAnalysis] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    
    # Versioning and relationships
    version: int = 1
    parent_id: Optional[str] = None  # For variations/versions
    child_ids: List[str] = Field(default_factory=list)  # Variations created from this
    
    # Metadata and tags
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Onyx integration
    onyx_workspace_id: Optional[str] = None
    onyx_user_id: Optional[str] = None
    onyx_project_id: Optional[str] = None
    
    # LangChain integration
    langchain_trace: List[Dict[str, Any]] = Field(default_factory=list)
    langchain_session_id: Optional[str] = None
    
    # Performance tracking
    actual_metrics: Optional[Dict[str, Any]] = None
    ab_test_group: Optional[str] = None
    
    @dataclass
class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @root_validator
    def validate_consistency(cls, values) -> bool:
        """Validar consistencia entre campos."""
        content = values.get('content')
        config = values.get('generation_config')
        
        if content and config:
            if len(content.get_display_text()) > config.max_length:
                raise ValueError("Content exceeds generation config max_length")
        
        return values
    
    # ===== BUSINESS METHODS =====
    
    def update_content(self, new_content: FacebookPostContent) -> None:
        """Actualizar contenido con invalidación de análisis."""
        self.content = new_content
        self.analysis = None  # Invalidate analysis
        self.status = ContentStatus.DRAFT  # Reset status
        self.updated_at = datetime.now()
        self.version += 1
        
        self.add_langchain_trace("content_updated", {
            "new_content_length": len(new_content.text),
            "hashtags_count": len(new_content.hashtags),
            "version": self.version
        })
    
    def set_analysis(self, analysis: FacebookPostAnalysis) -> None:
        """Establecer análisis con trazabilidad."""
        self.analysis = analysis
        self.updated_at = datetime.now()
        
        self.add_langchain_trace("analysis_completed", {
            "overall_score": analysis.get_overall_score(),
            "quality_tier": analysis.quality_assessment.quality_tier.value,
            "confidence": analysis.confidence_level,
            "processing_time_ms": analysis.processing_time_ms
        })
        
        # Auto-update status based on analysis
        if analysis.get_overall_score() >= 0.8:
            self.status = ContentStatus.APPROVED
        elif analysis.get_overall_score() >= 0.6:
            self.status = ContentStatus.UNDER_REVIEW
    
    def update_status(self, new_status: ContentStatus, user_id: Optional[str] = None) -> None:
        """Actualizar estado con validación."""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status == ContentStatus.PUBLISHED:
            self.published_at = datetime.now()
        
        self.add_langchain_trace("status_changed", {
            "from": old_status.value,
            "to": new_status.value,
            "user_id": user_id,
            "timestamp": self.updated_at.isoformat()
        })
    
    def add_langchain_trace(self, step: str, data: Dict[str, Any]) -> None:
        """Agregar trazabilidad LangChain detallada."""
        self.langchain_trace.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.langchain_session_id,
            "data": data
        })
        
        # Maintain trace size (keep last 50 entries)
        if len(self.langchain_trace) > 50:
            self.langchain_trace = self.langchain_trace[-50:]
    
    # ===== COMPUTED PROPERTIES =====
    
    def is_ready_for_publication(self) -> bool:
        """Verificar si está listo para publicación."""
        return (
            self.status in [ContentStatus.APPROVED, ContentStatus.SCHEDULED] and
            self.analysis is not None and
            self.analysis.get_overall_score() >= 0.6 and
            len(self.validate_for_publication()) == 0
        )
    
    def get_engagement_score(self) -> float:
        """Score de engagement (real o predicho)."""
        if self.actual_metrics and 'engagement_rate' in self.actual_metrics:
            return self.actual_metrics['engagement_rate']
        elif self.analysis:
            return self.analysis.engagement_prediction.engagement_rate
        return 0.5
    
    def get_quality_tier(self) -> str:
        """Tier de calidad del contenido."""
        if self.analysis:
            return self.analysis.quality_assessment.quality_tier.value
        return "unassessed"
    
    def validate_for_publication(self) -> List[str]:
        """Validaciones específicas para publicación."""
        errors = []
        
        # Content validation
        display_text = self.content.get_display_text()
        if len(display_text) > 2000:
            errors.append("Content exceeds Facebook's 2000 character limit")
        
        if len(self.content.text.strip()) < 10:
            errors.append("Content is too short for meaningful engagement")
        
        # Quality thresholds
        if self.analysis:
            if self.analysis.get_overall_score() < 0.5:
                errors.append("Content quality score is below minimum threshold")
            
            if self.analysis.confidence_level < 0.6:
                errors.append("Analysis confidence is too low for publication")
        else:
            errors.append("Content must be analyzed before publication")
        
        # Status validation
        if self.status not in [ContentStatus.APPROVED, ContentStatus.SCHEDULED]:
            errors.append("Content must be approved before publication")
        
        return errors
    
    def get_display_preview(self) -> str:
        """Preview optimizado del post."""
        preview = self.content.text[:97]
        if len(self.content.text) > 97:
            preview += "..."
        
        additions = []
        if self.content.hashtags:
            additions.append(f"{len(self.content.hashtags)} hashtags")
        if self.content.media_urls:
            additions.append(f"{len(self.content.media_urls)} media")
        if self.content.call_to_action:
            additions.append("CTA")
        
        if additions:
            preview += f" [{', '.join(additions)}]"
        
        return preview
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Resumen de performance completo."""
        summary = {
            "post_id": self.identifier.content_id,
            "status": self.status.value,
            "quality_tier": self.get_quality_tier(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.updated_at.isoformat()
        }
        
        if self.analysis:
            summary.update({
                "overall_score": self.analysis.get_overall_score(),
                "engagement_prediction": self.analysis.engagement_prediction.engagement_rate,
                "virality_score": self.analysis.engagement_prediction.virality_score,
                "recommendations_count": len(self.analysis.get_actionable_recommendations())
            })
        
        if self.actual_metrics:
            summary.update({
                "actual_performance": self.actual_metrics,
                "prediction_accuracy": self.actual_metrics.get("prediction_accuracy", {})
            })
        
        return summary
    
    # ===== COMPARISON & HASHING =====
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FacebookPostEntity):
            return False
        return self.identifier.content_id == other.identifier.content_id
    
    def __hash__(self) -> int:
        return hash(self.identifier.content_id)
    
    def __str__(self) -> str:
        return f"FacebookPost({self.identifier.content_id[:8]}...)"
    
    def __repr__(self) -> str:
        return (
            f"FacebookPostEntity("
            f"id={self.identifier.content_id}, "
            f"topic={self.specification.topic}, "
            f"status={self.status.value}, "
            f"quality={self.get_quality_tier()})"
        )


# ===== FACTORY =====

class FacebookPostFactory:
    """Factory para crear posts optimizados."""
    
    @staticmethod
    def create_from_specification(
        specification: PostSpecification,
        generation_config: GenerationConfig,
        content_text: str,
        hashtags: Optional[List[str]] = None,
        **kwargs
    ) -> FacebookPostEntity:
        """Crear post completo desde especificación."""
        
        # Generate identifier
        identifier = ContentIdentifier.generate(
            content_text, 
            {"spec": specification.topic, "config": generation_config.max_length}
        )
        
        # Create content
        content = FacebookPostContent(
            text=content_text,
            hashtags=hashtags or [],
            mentions=kwargs.get('mentions', []),
            media_urls=kwargs.get('media_urls', []),
            link_url=kwargs.get('link_url'),
            call_to_action=kwargs.get('call_to_action')
        )
        
        return FacebookPostEntity(
            identifier=identifier,
            specification=specification,
            generation_config=generation_config,
            content=content,
            onyx_workspace_id=kwargs.get('workspace_id'),
            onyx_user_id=kwargs.get('user_id'),
            onyx_project_id=kwargs.get('project_id')
        )
    
    @staticmethod
    def create_high_performance_post(
        topic: str,
        audience: TargetAudience = TargetAudience.GENERAL,
        **kwargs
    ) -> FacebookPostEntity:
        """Crear post de alta performance."""
        
        spec = PostSpecification(
            topic=topic,
            post_type=PostType.TEXT,
            tone=ContentTone.INSPIRING,
            target_audience=audience,
            keywords=[topic.lower()],
            target_engagement=EngagementTier.HIGH
        )
        
        config = GenerationConfig(
            max_length=kwargs.get('max_length', 280),
            include_hashtags=True,
            include_emojis=True,
            include_call_to_action=True
        )
        
        content_text = f"✨ Discover amazing {topic} insights! Transform your approach today. What's your experience? 💭"
        hashtags = [topic.lower().replace(' ', ''), 'success', 'growth', 'transformation']
        
        return FacebookPostFactory.create_from_specification(
            specification=spec,
            generation_config=config,
            content_text=content_text,
            hashtags=hashtags,
            **kwargs
        )


# ===== REQUEST/RESPONSE MODELS =====

class FacebookPostRequest(BaseModel):
    """Request para generar Facebook post."""
    topic: str = Field(..., min_length=3, max_length=200)
    post_type: PostType = PostType.TEXT
    tone: ContentTone = ContentTone.CASUAL
    target_audience: TargetAudience = TargetAudience.GENERAL
    target_engagement: EngagementTier = EngagementTier.HIGH
    
    # Generation config
    max_length: int = Field(280, ge=50, le=2000)
    include_hashtags: bool = True
    include_emojis: bool = True
    include_call_to_action: bool = True
    
    # Advanced options
    keywords: List[str] = Field(default_factory=list, max_items=10)
    brand_voice: Optional[str] = None
    campaign_context: Optional[str] = None
    custom_instructions: Optional[str] = None
    
    # Onyx context
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None


class FacebookPostResponse(BaseModel):
    """Respuesta de generación de Facebook post."""
    success: bool = Field(..., description="Éxito de la operación")
    post: Optional[FacebookPostEntity] = Field(None, description="Post generado")
    variations: List[FacebookPostEntity] = Field(default_factory=list, description="Variaciones")
    analysis: Optional[FacebookPostAnalysis] = Field(None, description="Análisis del post")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento")
    error_message: Optional[str] = Field(None, description="Mensaje de error")
    
    # LangChain metadata
    langchain_session_id: Optional[str] = None
    generation_steps: List[Dict[str, Any]] = Field(default_factory=list)


# ===== LEGACY COMPATIBILITY =====

# Backwards compatibility aliases
FacebookPost = FacebookPostEntity
FacebookAnalysis = FacebookPostAnalysis
FacebookRequest = FacebookPostRequest
FacebookTone = ContentTone
FacebookAudience = TargetAudience
FacebookPostType = PostType 