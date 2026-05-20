from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid
import json
        import hashlib
        import re
        import re
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
📝 Core Models - Modelos Consolidados
===================================

Modelos principales consolidados para el sistema de Facebook Posts.
Incluye entidades, enums, y estructuras de datos centralizadas.
"""


# ===== ENUMS =====

class PostStatus(Enum):
    """Estados posibles de un post."""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    ARCHIVED = "archived"

class ContentType(Enum):
    """Tipos de contenido."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PROMOTIONAL = "promotional"
    NEWS = "news"
    PERSONAL = "personal"
    TECHNICAL = "technical"
    INSPIRATIONAL = "inspirational"

class AudienceType(Enum):
    """Tipos de audiencia."""
    GENERAL = "general"
    PROFESSIONALS = "professionals"
    ENTREPRENEURS = "entrepreneurs"
    STUDENTS = "students"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"

class OptimizationLevel(Enum):
    """Niveles de optimización."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ULTRA = "ultra"

class QualityTier(Enum):
    """Niveles de calidad."""
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"
    EXCEPTIONAL = "exceptional"

# ===== VALUE OBJECTS =====

@dataclass(frozen=True)
class ContentIdentifier:
    """Identificador inmutable para contenido."""
    value: str
    
    def __post_init__(self) -> Any:
        if not self.value or len(self.value.strip()) == 0:
            raise ValueError("Content identifier cannot be empty")
    
    @classmethod
    def generate(cls, content: str) -> 'ContentIdentifier':
        """Generar identificador basado en contenido."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return cls(f"content_{content_hash[:8]}")
    
    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class PostMetrics:
    """Métricas de performance de un post."""
    engagement_score: float
    quality_score: float
    readability_score: float
    sentiment_score: float
    creativity_score: float
    relevance_score: float
    
    def __post_init__(self) -> Any:
        # Validar que todos los scores estén entre 0 y 1
        for field_name, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")
    
    @property
    def overall_score(self) -> float:
        """Calcular score general ponderado."""
        weights = {
            'engagement_score': 0.25,
            'quality_score': 0.25,
            'readability_score': 0.15,
            'sentiment_score': 0.15,
            'creativity_score': 0.1,
            'relevance_score': 0.1
        }
        
        return sum(
            getattr(self, field) * weight 
            for field, weight in weights.items()
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {
            'engagement_score': self.engagement_score,
            'quality_score': self.quality_score,
            'readability_score': self.readability_score,
            'sentiment_score': self.sentiment_score,
            'creativity_score': self.creativity_score,
            'relevance_score': self.relevance_score,
            'overall_score': self.overall_score
        }

@dataclass(frozen=True)
class PublicationWindow:
    """Ventana de publicación para un post."""
    start_time: datetime
    end_time: datetime
    timezone: str = "UTC"
    
    def __post_init__(self) -> Any:
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Verificar si la ventana está activa."""
        if current_time is None:
            current_time = datetime.now()
        return self.start_time <= current_time <= self.end_time
    
    def time_until_start(self, current_time: Optional[datetime] = None) -> float:
        """Tiempo hasta el inicio en segundos."""
        if current_time is None:
            current_time = datetime.now()
        return (self.start_time - current_time).total_seconds()
    
    def time_until_end(self, current_time: Optional[datetime] = None) -> float:
        """Tiempo hasta el fin en segundos."""
        if current_time is None:
            current_time = datetime.now()
        return (self.end_time - current_time).total_seconds()

# ===== MAIN MODELS =====

@dataclass
class FacebookPost:
    """Modelo principal de post de Facebook."""
    id: str
    content: str
    status: PostStatus
    content_type: ContentType
    audience_type: AudienceType
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[PostMetrics] = None
    publication_window: Optional[PublicationWindow] = None
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    quality_tier: Optional[QualityTier] = None
    tags: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> Any:
        # Generar ID si no se proporciona
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Extraer hashtags, mentions y URLs del contenido
        if not self.hashtags:
            self.hashtags = self._extract_hashtags()
        if not self.mentions:
            self.mentions = self._extract_mentions()
        if not self.urls:
            self.urls = self._extract_urls()
    
    def _extract_hashtags(self) -> List[str]:
        """Extraer hashtags del contenido."""
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, self.content)
    
    def _extract_mentions(self) -> List[str]:
        """Extraer menciones del contenido."""
        mention_pattern = r'@\w+'
        return re.findall(mention_pattern, self.content)
    
    def _extract_urls(self) -> List[str]:
        """Extraer URLs del contenido."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, self.content)
    
    # Métodos de negocio
    def approve(self) -> None:
        """Aprobar el post."""
        if self.status == PostStatus.PENDING:
            self.status = PostStatus.APPROVED
            self.updated_at = datetime.now()
        else:
            raise ValueError(f"Cannot approve post in status {self.status}")
    
    def publish(self) -> None:
        """Publicar el post."""
        if self.status == PostStatus.APPROVED:
            self.status = PostStatus.PUBLISHED
            self.updated_at = datetime.now()
        else:
            raise ValueError(f"Cannot publish post in status {self.status}")
    
    def reject(self, reason: str = "") -> None:
        """Rechazar el post."""
        if self.status in [PostStatus.PENDING, PostStatus.DRAFT]:
            self.status = PostStatus.REJECTED
            self.updated_at = datetime.now()
            if reason:
                self.metadata['rejection_reason'] = reason
        else:
            raise ValueError(f"Cannot reject post in status {self.status}")
    
    def archive(self) -> None:
        """Archivar el post."""
        self.status = PostStatus.ARCHIVED
        self.updated_at = datetime.now()
    
    def update_content(self, new_content: str) -> None:
        """Actualizar contenido del post."""
        if self.status in [PostStatus.PUBLISHED, PostStatus.ARCHIVED]:
            raise ValueError(f"Cannot update content of {self.status} post")
        
        self.content = new_content
        self.updated_at = datetime.now()
        
        # Re-extraer elementos
        self.hashtags = self._extract_hashtags()
        self.mentions = self._extract_mentions()
        self.urls = self._extract_urls()
    
    def add_metrics(self, metrics: PostMetrics) -> None:
        """Añadir métricas al post."""
        self.metrics = metrics
        self.updated_at = datetime.now()
        
        # Actualizar quality tier basado en métricas
        overall_score = metrics.overall_score
        if overall_score >= 0.9:
            self.quality_tier = QualityTier.EXCEPTIONAL
        elif overall_score >= 0.8:
            self.quality_tier = QualityTier.EXCELLENT
        elif overall_score >= 0.6:
            self.quality_tier = QualityTier.GOOD
        else:
            self.quality_tier = QualityTier.BASIC
    
    def is_ready_for_publication(self) -> bool:
        """Verificar si el post está listo para publicación."""
        return (
            self.status == PostStatus.APPROVED and
            self.metrics is not None and
            self.metrics.overall_score >= 0.7
        )
    
    def get_word_count(self) -> int:
        """Obtener número de palabras."""
        return len(self.content.split())
    
    def get_character_count(self) -> int:
        """Obtener número de caracteres."""
        return len(self.content)
    
    def get_reading_time(self) -> float:
        """Obtener tiempo de lectura estimado en minutos."""
        words_per_minute = 200
        return self.get_word_count() / words_per_minute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'id': self.id,
            'content': self.content,
            'status': self.status.value,
            'content_type': self.content_type.value,
            'audience_type': self.audience_type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'publication_window': {
                'start_time': self.publication_window.start_time.isoformat(),
                'end_time': self.publication_window.end_time.isoformat(),
                'timezone': self.publication_window.timezone
            } if self.publication_window else None,
            'optimization_level': self.optimization_level.value,
            'quality_tier': self.quality_tier.value if self.quality_tier else None,
            'tags': self.tags,
            'hashtags': self.hashtags,
            'mentions': self.mentions,
            'urls': self.urls,
            'word_count': self.get_word_count(),
            'character_count': self.get_character_count(),
            'reading_time': self.get_reading_time()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FacebookPost':
        """Crear desde diccionario."""
        return cls(
            id=data['id'],
            content=data['content'],
            status=PostStatus(data['status']),
            content_type=ContentType(data['content_type']),
            audience_type=AudienceType(data['audience_type']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            metadata=data.get('metadata', {}),
            metrics=PostMetrics(**data['metrics']) if data.get('metrics') else None,
            publication_window=PublicationWindow(
                start_time=datetime.fromisoformat(data['publication_window']['start_time']),
                end_time=datetime.fromisoformat(data['publication_window']['end_time']),
                timezone=data['publication_window']['timezone']
            ) if data.get('publication_window') else None,
            optimization_level=OptimizationLevel(data.get('optimization_level', 'standard')),
            quality_tier=QualityTier(data['quality_tier']) if data.get('quality_tier') else None,
            tags=data.get('tags', []),
            hashtags=data.get('hashtags', []),
            mentions=data.get('mentions', []),
            urls=data.get('urls', [])
        )

@dataclass
class PostRequest:
    """Request para generar un post."""
    topic: str
    audience_type: AudienceType
    content_type: ContentType
    tone: str = "professional"
    length: Optional[int] = None
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    include_hashtags: bool = True
    include_mentions: bool = False
    include_urls: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        if not self.topic or len(self.topic.strip()) == 0:
            raise ValueError("Topic cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'topic': self.topic,
            'audience_type': self.audience_type.value,
            'content_type': self.content_type.value,
            'tone': self.tone,
            'length': self.length,
            'optimization_level': self.optimization_level.value,
            'include_hashtags': self.include_hashtags,
            'include_mentions': self.include_mentions,
            'include_urls': self.include_urls,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostRequest':
        """Crear desde diccionario."""
        return cls(
            topic=data['topic'],
            audience_type=AudienceType(data['audience_type']),
            content_type=ContentType(data['content_type']),
            tone=data.get('tone', 'professional'),
            length=data.get('length'),
            optimization_level=OptimizationLevel(data.get('optimization_level', 'standard')),
            include_hashtags=data.get('include_hashtags', True),
            include_mentions=data.get('include_mentions', False),
            include_urls=data.get('include_urls', False),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )

@dataclass
class PostResponse:
    """Response de generación de post."""
    success: bool
    post: Optional[FacebookPost] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    optimizations_applied: List[str] = field(default_factory=list)
    analytics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'success': self.success,
            'post': self.post.to_dict() if self.post else None,
            'error': self.error,
            'processing_time': self.processing_time,
            'optimizations_applied': self.optimizations_applied,
            'analytics': self.analytics
        }

# ===== FACTORY METHODS =====

class FacebookPostFactory:
    """Factory para crear posts de Facebook."""
    
    @classmethod
    def create_draft(
        cls,
        content: str,
        content_type: ContentType,
        audience_type: AudienceType,
        **kwargs
    ) -> FacebookPost:
        """Crear un post en estado draft."""
        return FacebookPost(
            id=str(uuid.uuid4()),
            content=content,
            status=PostStatus.DRAFT,
            content_type=content_type,
            audience_type=audience_type,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            **kwargs
        )
    
    @classmethod
    async def create_from_request(
        cls,
        request: PostRequest,
        content: str,
        **kwargs
    ) -> FacebookPost:
        """Crear un post desde un request."""
        return cls.create_draft(
            content=content,
            content_type=request.content_type,
            audience_type=request.audience_type,
            optimization_level=request.optimization_level,
            tags=request.tags,
            metadata=request.metadata,
            **kwargs
        )
    
    @classmethod
    def create_sample_post(cls) -> FacebookPost:
        """Crear un post de ejemplo."""
        return cls.create_draft(
            content="🚀 Exciting news! We just launched a new feature that will revolutionize your workflow. "
                   "Check it out and let us know what you think! #Innovation #Productivity #Tech",
            content_type=ContentType.PROMOTIONAL,
            audience_type=AudienceType.PROFESSIONALS,
            tags=["innovation", "productivity", "tech"]
        )

# ===== EXPORTS =====

__all__ = [
    # Enums
    'PostStatus',
    'ContentType', 
    'AudienceType',
    'OptimizationLevel',
    'QualityTier',
    
    # Value Objects
    'ContentIdentifier',
    'PostMetrics',
    'PublicationWindow',
    
    # Main Models
    'FacebookPost',
    'PostRequest',
    'PostResponse',
    
    # Factory
    'FacebookPostFactory'
] 