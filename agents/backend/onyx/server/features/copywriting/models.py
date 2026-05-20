from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, List, Dict, Any, Union, Literal
from enum import Enum
from functools import lru_cache
import time
from datetime import datetime
import uuid
import orjson
import msgspec
import polars as pl
import numpy as np
from pydantic import BaseModel, Field, validator, ConfigDict, computed_field
from pydantic_settings import BaseSettings
import structlog
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Ultra-Optimized Copywriting Models with High-Performance Libraries.

Production-ready models with advanced features: language, tone, voice, variants,
creativity, translation, use cases, website info, and comprehensive optimization.
"""


# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    MSGSPEC_AVAILABLE = True
except ImportError:
    MSGSPEC_AVAILABLE = False

try:
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Fast validation with pydantic v2

# Performance monitoring
logger = structlog.get_logger(__name__)

class CopyTone(str, Enum):
    """Enhanced copy tones with performance optimization."""
    assertive = "assertive"
    casual = "casual"
    formal = "formal"
    humorous = "humorous"
    informative = "informative"
    inspirational = "inspirational"
    professional = "professional"
    witty = "witty"
    urgent = "urgent"
    emotional = "emotional"
    friendly = "friendly"
    authoritative = "authoritative"
    conversational = "conversational"
    persuasive = "persuasive"
    storytelling = "storytelling"
    educational = "educational"
    motivational = "motivational"
    empathetic = "empathetic"
    confident = "confident"
    playful = "playful"

class VoiceStyle(str, Enum):
    """Brand voice styles for consistent communication."""
    corporate = "corporate"
    startup = "startup"
    personal = "personal"
    luxury = "luxury"
    tech = "tech"
    creative = "creative"
    health = "health"
    finance = "finance"
    education = "education"
    entertainment = "entertainment"
    nonprofit = "nonprofit"
    ecommerce = "ecommerce"
    consulting = "consulting"
    agency = "agency"
    blogger = "blogger"
    influencer = "influencer"

class ContentType(str, Enum):
    """Content types for different platforms."""
    ad_copy = "ad_copy"
    social_post = "social_post"
    email_subject = "email_subject"
    email_body = "email_body"
    blog_title = "blog_title"
    blog_content = "blog_content"
    product_description = "product_description"
    landing_page = "landing_page"
    video_script = "video_script"
    press_release = "press_release"

class Platform(str, Enum):
    """Supported platforms with specific requirements."""
    facebook = "facebook"
    instagram = "instagram"
    twitter = "twitter"
    linkedin = "linkedin"
    tiktok = "tiktok"
    youtube = "youtube"
    google_ads = "google_ads"
    email = "email"
    website = "website"
    blog = "blog"

class Language(str, Enum):
    """Supported languages with enhanced coverage."""
    es = "es"  # Spanish
    en = "en"  # English
    fr = "fr"  # French
    pt = "pt"  # Portuguese
    it = "it"  # Italian
    de = "de"  # German
    nl = "nl"  # Dutch
    ru = "ru"  # Russian
    zh = "zh"  # Chinese
    ja = "ja"  # Japanese
    ko = "ko"  # Korean
    ar = "ar"  # Arabic
    hi = "hi"  # Hindi
    tr = "tr"  # Turkish
    pl = "pl"  # Polish
    sv = "sv"  # Swedish
    da = "da"  # Danish
    no = "no"  # Norwegian
    fi = "fi"  # Finnish

class UseCase(str, Enum):
    """Specific use cases for copywriting."""
    product_launch = "product_launch"
    brand_awareness = "brand_awareness"
    lead_generation = "lead_generation"
    sales_conversion = "sales_conversion"
    customer_retention = "customer_retention"
    event_promotion = "event_promotion"
    content_marketing = "content_marketing"
    social_media = "social_media"
    email_marketing = "email_marketing"
    website_copy = "website_copy"
    ad_campaigns = "ad_campaigns"
    press_release = "press_release"
    blog_content = "blog_content"
    product_description = "product_description"
    landing_page = "landing_page"
    video_script = "video_script"
    podcast_script = "podcast_script"
    newsletter = "newsletter"
    case_study = "case_study"
    testimonial = "testimonial"
    faq = "faq"
    user_guide = "user_guide"
    announcement = "announcement"
    seasonal_campaign = "seasonal_campaign"
    influencer_outreach = "influencer_outreach"

class CreativityLevel(str, Enum):
    """Creativity levels for content generation."""
    conservative = "conservative"  # 0.1-0.3
    balanced = "balanced"         # 0.4-0.6
    creative = "creative"         # 0.7-0.8
    innovative = "innovative"     # 0.9-1.0

class FeedbackType(str, Enum):
    human = "human"
    model = "model"
    auto = "auto"
    ai_analysis = "ai_analysis"

# Fast serialization mixin
class FastSerializationMixin:
    """Mixin for ultra-fast serialization."""
    
    def to_json(self) -> bytes:
        """Ultra-fast JSON serialization."""
        if MSGSPEC_AVAILABLE:
            return msgspec.json.encode(self.model_dump())
        elif JSON_AVAILABLE:
            return orjson.dumps(self.model_dump())
        else:
            return json.dumps(self.model_dump()).encode()
    
    @classmethod
    def from_json(cls, data: bytes):
        """Ultra-fast JSON deserialization."""
        if MSGSPEC_AVAILABLE:
            return cls(**msgspec.json.decode(data))
        elif JSON_AVAILABLE:
            return cls(**orjson.loads(data))
        else:
            return cls(**json.loads(data))

class OptimizedBaseModel(BaseModel, FastSerializationMixin):
    """Base model with performance optimizations."""
    
    model_config = ConfigDict(
        # Performance optimizations
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        # Serialization optimizations
        ser_json_bytes="utf8" if JSON_AVAILABLE else False,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        } if not JSON_AVAILABLE else {}
    )

class Feedback(OptimizedBaseModel):
    """Optimized feedback model."""
    type: FeedbackType = Field(..., description="Origen del feedback")
    score: Optional[float] = Field(None, ge=0, le=1, description="Puntaje de calidad (0-1)")
    comments: Optional[str] = Field(None, max_length=1000, description="Comentarios adicionales")
    user_id: Optional[str] = Field(None, max_length=50, description="ID del usuario")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Fecha/hora del feedback")
    
    # Performance metrics
    processing_time_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en ms")

class Metric(OptimizedBaseModel):
    """Optimized metric model."""
    name: str = Field(..., max_length=100, description="Nombre de la métrica")
    value: Union[float, int, str] = Field(..., description="Valor de la métrica")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalles adicionales")
    calculated_at: Optional[datetime] = Field(default_factory=datetime.now)

class WebsiteInfo(OptimizedBaseModel):
    """Website information for context-aware copywriting."""
    website_name: str = Field(..., max_length=100, description="Nombre del sitio web")
    domain: Optional[str] = Field(None, max_length=100, description="Dominio del sitio web")
    about: Optional[str] = Field(None, max_length=1000, description="Descripción sobre la empresa/sitio")
    features: Optional[List[str]] = Field(None, max_items=20, description="Características principales")
    target_market: Optional[str] = Field(None, max_length=200, description="Mercado objetivo")
    value_proposition: Optional[str] = Field(None, max_length=300, description="Propuesta de valor")
    company_size: Optional[str] = Field(None, description="Tamaño de la empresa")
    industry: Optional[str] = Field(None, max_length=100, description="Industria")
    founded_year: Optional[int] = Field(None, ge=1800, le=2030, description="Año de fundación")
    headquarters: Optional[str] = Field(None, max_length=100, description="Sede principal")
    
    @validator('features')
    def validate_features(cls, v) -> bool:
        if v:
            return [feature.strip() for feature in v if feature.strip()][:20]
        return v

class BrandVoice(OptimizedBaseModel):
    """Enhanced brand voice configuration with advanced features."""
    tone: Optional[CopyTone] = Field(None, description="Tono principal de la marca")
    voice_style: Optional[VoiceStyle] = Field(None, description="Estilo de voz de marca")
    personality_traits: Optional[List[str]] = Field(None, max_items=15, description="Rasgos de personalidad")
    communication_style: Optional[str] = Field(None, max_length=300, description="Estilo de comunicación")
    
    # Advanced voice settings
    formality_level: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Nivel de formalidad (0=informal, 1=formal)")
    emotion_level: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Nivel emocional")
    technical_level: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Nivel técnico")
    
    # Brand guidelines
    do_use: Optional[List[str]] = Field(None, max_items=10, description="Palabras/frases a usar")
    dont_use: Optional[List[str]] = Field(None, max_items=10, description="Palabras/frases a evitar")
    preferred_phrases: Optional[List[str]] = Field(None, max_items=10, description="Frases preferidas")
    voice_examples: Optional[List[str]] = Field(None, max_items=8, description="Ejemplos de voz de marca")
    
    # Context
    target_audience_age: Optional[str] = Field(None, description="Rango de edad de audiencia")
    cultural_context: Optional[str] = Field(None, max_length=200, description="Contexto cultural")
    
    @validator('personality_traits')
    def validate_traits(cls, v) -> bool:
        if v:
            return [trait.strip().lower() for trait in v if trait.strip()][:15]
        return v

class AudienceProfile(OptimizedBaseModel):
    """Optimized audience profile."""
    demographics: Optional[Dict[str, Any]] = Field(None, description="Datos demográficos")
    interests: Optional[List[str]] = Field(None, max_items=20, description="Intereses")
    pain_points: Optional[List[str]] = Field(None, max_items=10, description="Puntos de dolor")
    goals: Optional[List[str]] = Field(None, max_items=10, description="Objetivos")
    stage: Optional[str] = Field(None, max_length=50, description="Etapa del cliente")
    
    # Enhanced fields
    age_range: Optional[str] = Field(None, description="Rango de edad")
    income_level: Optional[str] = Field(None, description="Nivel de ingresos")
    education_level: Optional[str] = Field(None, description="Nivel educativo")
    preferred_channels: Optional[List[Platform]] = Field(None, description="Canales preferidos")

class ProjectContext(OptimizedBaseModel):
    """Optimized project context."""
    project_name: Optional[str] = Field(None, max_length=100, description="Nombre del proyecto")
    project_description: Optional[str] = Field(None, max_length=500, description="Descripción del proyecto")
    industry: Optional[str] = Field(None, max_length=100, description="Industria")
    key_messages: Optional[List[str]] = Field(None, max_items=5, description="Mensajes clave")
    brand_assets: Optional[List[str]] = Field(None, max_items=10, description="Activos de marca")
    content_sources: Optional[List[str]] = Field(None, max_items=10, description="Fuentes de contenido")
    custom_variables: Optional[Dict[str, Any]] = Field(None, description="Variables personalizadas")
    
    # Enhanced context
    campaign_objective: Optional[str] = Field(None, description="Objetivo de la campaña")
    budget_range: Optional[str] = Field(None, description="Rango de presupuesto")
    timeline: Optional[str] = Field(None, description="Timeline del proyecto")
    competitors: Optional[List[str]] = Field(None, max_items=5, description="Competidores principales")

class PlatformSettings(OptimizedBaseModel):
    """Optimized platform-specific settings."""
    platform: Platform = Field(..., description="Plataforma objetivo")
    requirements: Optional[Dict[str, Any]] = Field(None, description="Requisitos específicos")
    language: Optional[Language] = Field(Language.es, description="Idioma")
    
    # Platform-specific constraints
    max_characters: Optional[int] = Field(None, description="Máximo de caracteres")
    max_hashtags: Optional[int] = Field(None, description="Máximo de hashtags")
    supports_emojis: Optional[bool] = Field(True, description="Soporte para emojis")
    supports_links: Optional[bool] = Field(True, description="Soporte para enlaces")
    
    @validator('max_characters')
    def validate_max_characters(cls, v, values) -> bool:
        """Set default character limits based on platform."""
        if v is None and 'platform' in values:
            platform_limits = {
                Platform.twitter: 280,
                Platform.instagram: 2200,
                Platform.facebook: 63206,
                Platform.linkedin: 3000,
                Platform.tiktok: 300,
            }
            return platform_limits.get(values['platform'], 1000)
        return v

class SectionFeedback(OptimizedBaseModel):
    """Optimized section-specific feedback."""
    section: str = Field(..., max_length=50, description="Sección del contenido")
    feedback: Feedback = Field(..., description="Feedback específico")
    suggestions: Optional[List[str]] = Field(None, max_items=5, description="Sugerencias de mejora")

class CopyVariantHistory(OptimizedBaseModel):
    """Optimized copy variant history."""
    variant_id: str = Field(..., max_length=50, description="ID de la variante")
    previous_versions: Optional[List[str]] = Field(None, max_items=10, description="Versiones anteriores")
    change_log: Optional[List[str]] = Field(None, max_items=20, description="Log de cambios")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(None)

class OptimizationResult(OptimizedBaseModel):
    """Optimized optimization results."""
    experiment_id: str = Field(..., max_length=50, description="ID del experimento")
    best_variant_id: str = Field(..., max_length=50, description="ID de la mejor variante")
    metrics: List[Metric] = Field(..., description="Métricas del experimento")
    recommendations: Optional[str] = Field(None, max_length=1000, description="Recomendaciones")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Nivel de confianza")
    sample_size: Optional[int] = Field(None, description="Tamaño de la muestra")

class TranslationSettings(OptimizedBaseModel):
    """Translation settings for multi-language support."""
    target_languages: List[Language] = Field(..., min_items=1, description="Idiomas objetivo")
    maintain_tone: bool = Field(True, description="Mantener tono en traducción")
    cultural_adaptation: bool = Field(True, description="Adaptación cultural")
    preserve_formatting: bool = Field(True, description="Preservar formato")
    translate_hashtags: bool = Field(False, description="Traducir hashtags")
    localize_currency: bool = Field(True, description="Localizar moneda")
    localize_dates: bool = Field(True, description="Localizar fechas")
    
class VariantSettings(OptimizedBaseModel):
    """Settings for variant generation."""
    max_variants: int = Field(5, ge=1, le=20, description="Número máximo de variantes")
    variant_diversity: float = Field(0.7, ge=0.0, le=1.0, description="Diversidad entre variantes")
    length_variation: bool = Field(True, description="Variación en longitud")
    tone_variation: bool = Field(False, description="Variación en tono")
    structure_variation: bool = Field(True, description="Variación en estructura")
    keyword_density_variation: bool = Field(True, description="Variación en densidad de palabras clave")
    
class CopywritingInput(OptimizedBaseModel):
    """Ultra-optimized copywriting input model with enhanced features."""
    
    # === CORE FIELDS ===
    product_description: str = Field(..., max_length=2000, description="Descripción del producto/servicio")
    target_platform: Platform = Field(..., description="Plataforma objetivo")
    content_type: ContentType = Field(..., description="Tipo de contenido")
    tone: CopyTone = Field(..., description="Tono deseado")
    use_case: UseCase = Field(..., description="Caso de uso específico")
    
    # === LANGUAGE & TRANSLATION ===
    language: Language = Field(Language.es, description="Idioma principal del contenido")
    translation_settings: Optional[TranslationSettings] = Field(None, description="Configuración de traducción")
    
    # === CREATIVITY & VARIANTS ===
    creativity_level: CreativityLevel = Field(CreativityLevel.balanced, description="Nivel de creatividad")
    creativity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Puntuación de creatividad personalizada")
    variant_settings: Optional[VariantSettings] = Field(None, description="Configuración de variantes")
    
    # === WEBSITE & BRAND INFO ===
    website_info: Optional[WebsiteInfo] = Field(None, description="Información del sitio web")
    brand_voice: Optional[BrandVoice] = Field(None, description="Voz de marca")
    
    # === AUDIENCE & CONTEXT ===
    target_audience: Optional[str] = Field(None, max_length=800, description="Audiencia objetivo detallada")
    audience_profile: Optional[AudienceProfile] = Field(None, description="Perfil de audiencia")
    project_context: Optional[ProjectContext] = Field(None, description="Contexto del proyecto")
    
    # === CONTENT SPECIFICATIONS ===
    key_points: Optional[List[str]] = Field(None, max_items=15, description="Puntos clave a incluir")
    features_to_highlight: Optional[List[str]] = Field(None, max_items=10, description="Características a destacar")
    call_to_action: Optional[str] = Field(None, max_length=100, description="Llamada a la acción específica")
    keywords: Optional[List[str]] = Field(None, max_items=20, description="Palabras clave SEO")
    
    # === INSTRUCTIONS & RESTRICTIONS ===
    instructions: Optional[str] = Field(None, max_length=1000, description="Instrucciones adicionales")
    restrictions: Optional[List[str]] = Field(None, max_items=15, description="Restricciones y limitaciones")
    compliance_requirements: Optional[List[str]] = Field(None, max_items=10, description="Requisitos de cumplimiento")
    
    # === PLATFORM SETTINGS ===
    platform_settings: Optional[PlatformSettings] = Field(None, description="Configuración específica de plataforma")
    max_length: Optional[int] = Field(None, ge=10, le=10000, description="Longitud máxima del contenido")
    min_length: Optional[int] = Field(None, ge=10, le=5000, description="Longitud mínima del contenido")
    
    # === OPTIMIZATION SETTINGS ===
    seo_optimization: bool = Field(False, description="Optimización SEO")
    engagement_optimization: bool = Field(True, description="Optimización para engagement")
    conversion_optimization: bool = Field(False, description="Optimización para conversión")
    
    # === TRACKING & METADATA ===
    tracking_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="ID de seguimiento")
    user_id: Optional[str] = Field(None, max_length=50, description="ID del usuario")
    session_id: Optional[str] = Field(None, max_length=50, description="ID de sesión")
    campaign_id: Optional[str] = Field(None, max_length=50, description="ID de campaña")
    
    # === PERFORMANCE SETTINGS ===
    priority: Optional[str] = Field("normal", description="Prioridad de procesamiento")
    timeout: Optional[int] = Field(30, ge=5, le=300, description="Timeout en segundos")
    
    @validator('key_points')
    def validate_key_points(cls, v) -> bool:
        if v:
            return [point.strip() for point in v if point.strip()][:15]
        return v
    
    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        if v:
            return [kw.strip().lower() for kw in v if kw.strip()][:20]
        return v
    
    @computed_field
    @property
    def effective_creativity_score(self) -> float:
        """Calculate effective creativity score."""
        if self.creativity_score is not None:
            return self.creativity_score
        
        # Map creativity level to score
        level_mapping = {
            CreativityLevel.conservative: 0.2,
            CreativityLevel.balanced: 0.6,
            CreativityLevel.creative: 0.8,
            CreativityLevel.innovative: 0.95
        }
        return level_mapping.get(self.creativity_level, 0.6)
    
    @computed_field
    @property
    def effective_max_variants(self) -> int:
        """Get effective max variants."""
        if self.variant_settings:
            return self.variant_settings.max_variants
        return 5

class CopyVariant(OptimizedBaseModel):
    """Ultra-optimized copy variant model."""
    # Core content
    headline: str = Field(..., max_length=200, description="Encabezado principal")
    primary_text: str = Field(..., max_length=5000, description="Texto principal")
    call_to_action: Optional[str] = Field(None, max_length=100, description="Llamada a la acción")
    
    # Social media specific
    hashtags: Optional[List[str]] = Field(None, max_items=30, description="Hashtags")
    mentions: Optional[List[str]] = Field(None, max_items=10, description="Menciones")
    
    # Platform optimization
    platform_tips: Optional[str] = Field(None, max_length=300, description="Tips de plataforma")
    character_count: Optional[int] = Field(None, description="Conteo de caracteres")
    word_count: Optional[int] = Field(None, description="Conteo de palabras")
    
    # Quality metrics
    evaluation_metrics: Optional[List[Metric]] = Field(None, description="Métricas de evaluación")
    readability_score: Optional[float] = Field(None, ge=0, le=100, description="Puntuación de legibilidad")
    engagement_prediction: Optional[float] = Field(None, ge=0, le=1, description="Predicción de engagement")
    
    # Feedback and history
    feedback: Optional[List[Feedback]] = Field(None, description="Feedback recibido")
    section_feedback: Optional[List[SectionFeedback]] = Field(None, description="Feedback por sección")
    history: Optional[CopyVariantHistory] = Field(None, description="Historial de cambios")
    
    # Metadata
    variant_id: str = Field(..., max_length=50, description="ID único de la variante")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(None)
    extra: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")
    
    @validator('hashtags')
    def validate_hashtags(cls, v) -> bool:
        if v:
            # Clean and format hashtags
            clean_hashtags = []
            for tag in v:
                tag = tag.strip().replace('#', '').replace(' ', '')
                if tag:
                    clean_hashtags.append(f"#{tag}")
            return clean_hashtags
        return v

class CopywritingOutput(OptimizedBaseModel):
    """Ultra-optimized copywriting output model."""
    # Generated content
    variants: List[CopyVariant] = Field(..., min_items=1, description="Variantes generadas")
    
    # Generation metadata
    model_used: Optional[str] = Field(None, max_length=100, description="Modelo utilizado")
    generation_time: Optional[float] = Field(None, description="Tiempo de generación en segundos")
    tokens_used: Optional[int] = Field(None, description="Tokens utilizados")
    
    # Performance metrics
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Métricas de rendimiento")
    optimization_suggestions: Optional[List[str]] = Field(None, description="Sugerencias de optimización")
    
    # Tracking and batching
    batch_id: Optional[str] = Field(None, max_length=50, description="ID de lote")
    tracking_id: Optional[str] = Field(None, max_length=50, description="ID de seguimiento")
    
    # Results and optimization
    optimization_results: Optional[List[OptimizationResult]] = Field(None, description="Resultados de optimización")
    best_variant_id: Optional[str] = Field(None, description="ID de la mejor variante")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Puntuación de confianza")
    
    # Metadata
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales")
    created_at: datetime = Field(default_factory=datetime.now)

class CopywritingSettings(BaseSettings):
    """Optimized settings with performance considerations."""
    
    # Security & API
    api_key: str = Field("test-secret-key", description="API key for authentication")
    allowed_cors_origins: List[str] = Field(["*"], description="Allowed CORS origins")
    
    # Performance settings
    max_concurrent_requests: int = Field(100, description="Maximum concurrent requests")
    request_timeout: int = Field(30, description="Request timeout in seconds")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    
    # AI/ML settings
    default_model: str = Field("gpt-3.5-turbo", description="Default AI model")
    max_tokens: int = Field(1000, description="Maximum tokens per request")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="AI model temperature")
    
    # Redis (for caching & rate limiting)
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")
    redis_db: int = Field(0, description="Redis database number")
    
    # Database
    database_url: Optional[str] = Field(None, description="Database connection URL")
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable performance metrics")
    enable_logging: bool = Field(True, description="Enable detailed logging")
    log_level: str = Field("INFO", description="Logging level")
    
    # External services
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    
    model_config = ConfigDict(
        env_prefix="COPYWRITING_",
        env_file=".env",
        case_sensitive=False
    )

# Performance-optimized singleton
@lru_cache(maxsize=1)
def get_settings() -> CopywritingSettings:
    """Get cached settings instance for optimal performance."""
    return CopywritingSettings()

# Fast validation functions
def validate_input_fast(data: Dict[str, Any]) -> bool:
    """Ultra-fast input validation."""
    required_fields = ["product_description", "target_platform", "content_type", "tone"]
    return all(field in data and data[field] for field in required_fields)

def calculate_metrics_fast(text: str) -> Dict[str, float]:
    """Fast metric calculation."""
    words = len(text.split())
    chars = len(text)
    
    return {
        "word_count": words,
        "character_count": chars,
        "avg_word_length": chars / words if words > 0 else 0,
        "reading_time_minutes": words / 200,  # Average reading speed
    }

# Export optimized models
__all__ = [
    "CopyTone", "ContentType", "Platform", "Language", "FeedbackType",
    "Feedback", "Metric", "BrandVoice", "AudienceProfile", "ProjectContext",
    "PlatformSettings", "SectionFeedback", "CopyVariantHistory", 
    "OptimizationResult", "CopywritingInput", "CopyVariant", 
    "CopywritingOutput", "CopywritingSettings", "get_settings",
    "validate_input_fast", "calculate_metrics_fast"
] 