"""
NLP Models - Modelos de datos para el sistema NLP
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Language(Enum):
    """Idiomas soportados."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "unknown"


class SentimentType(Enum):
    """Tipos de sentimiento."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TextType(Enum):
    """Tipos de texto."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    FORMAL = "formal"
    INFORMAL = "informal"
    UNKNOWN = "unknown"


@dataclass
class TextAnalysisResult:
    """Resultado del análisis de texto."""
    text: str
    language: Language
    sentiment: SentimentType
    confidence: float
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    text_type: TextType = TextType.UNKNOWN
    readability_score: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    character_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentResult:
    """Resultado del análisis de sentimiento."""
    text: str
    sentiment: SentimentType
    confidence: float
    positive_score: float = 0.0
    negative_score: float = 0.0
    neutral_score: float = 0.0
    mixed_score: float = 0.0
    emotional_intensity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LanguageDetectionResult:
    """Resultado de detección de idioma."""
    text: str
    detected_language: Language
    confidence: float
    alternative_languages: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TranslationResult:
    """Resultado de traducción."""
    original_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SummarizationResult:
    """Resultado de resumen."""
    original_text: str
    summary: str
    compression_ratio: float
    key_points: List[str] = field(default_factory=list)
    word_count_original: int = 0
    word_count_summary: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextGenerationResult:
    """Resultado de generación de texto."""
    prompt: str
    generated_text: str
    model_used: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EntityRecognitionResult:
    """Resultado de reconocimiento de entidades."""
    text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KeywordExtractionResult:
    """Resultado de extracción de palabras clave."""
    text: str
    keywords: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TopicModelingResult:
    """Resultado de modelado de temas."""
    text: str
    topics: List[Dict[str, Any]] = field(default_factory=list)
    dominant_topic: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextSimilarityResult:
    """Resultado de similitud de texto."""
    text1: str
    text2: str
    similarity_score: float
    similarity_type: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextClassificationResult:
    """Resultado de clasificación de texto."""
    text: str
    predicted_class: str
    confidence: float
    all_classes: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NLPAnalysisRequest:
    """Solicitud de análisis NLP."""
    text: str
    analysis_types: List[str] = field(default_factory=lambda: ["sentiment", "language", "entities"])
    language: Optional[Language] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NLPAnalysisResponse:
    """Respuesta de análisis NLP."""
    request_id: str
    results: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)




