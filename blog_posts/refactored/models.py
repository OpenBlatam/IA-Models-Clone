from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
        import re
        import orjson
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Modelos de datos para el sistema NLP ultra-optimizado.
"""


class AnalysisStatus(Enum):
    """Estados del análisis NLP."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    CACHED = "cached"
    ERROR = "error"

class QualityLevel(Enum):
    """Niveles de calidad del contenido."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    AVERAGE = "average"     # 60-74
    POOR = "poor"          # <60

@dataclass
class BasicMetrics:
    """Métricas básicas del texto."""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    character_count: int = 0
    avg_words_per_sentence: float = 0.0
    
    @classmethod
    def from_text(cls, text: str) -> 'BasicMetrics':
        """Crear métricas desde texto."""
        
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return cls(
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs),
            character_count=len(text),
            avg_words_per_sentence=len(words) / max(len(sentences), 1)
        )

@dataclass
class SentimentMetrics:
    """Métricas de análisis de sentimientos."""
    polarity: float = 0.0  # -1 (negativo) a 1 (positivo)
    subjectivity: float = 0.0  # 0 (objetivo) a 1 (subjetivo)
    confidence: float = 0.0  # 0 a 1
    label: str = "neutral"  # positive, negative, neutral
    score: float = 50.0  # 0-100 score normalizado

@dataclass  
class ReadabilityMetrics:
    """Métricas de legibilidad del texto."""
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    coleman_liau: float = 0.0
    automated_readability: float = 0.0
    score: float = 50.0  # 0-100 score normalizado
    level: str = "average"  # very_easy, easy, average, difficult, very_difficult

@dataclass
class KeywordMetrics:
    """Métricas de palabras clave."""
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    density: Dict[str, float] = field(default_factory=dict)
    total_keywords: int = 0
    avg_score: float = 0.0

@dataclass
class LanguageMetrics:
    """Métricas de detección de idioma."""
    detected_language: str = "unknown"
    confidence: float = 0.0
    all_languages: List[Tuple[str, float]] = field(default_factory=list)

@dataclass
class QualityMetrics:
    """Métricas de calidad general."""
    overall_score: float = 0.0
    level: QualityLevel = QualityLevel.AVERAGE
    content_score: float = 0.0
    structure_score: float = 0.0
    engagement_score: float = 0.0
    seo_score: float = 0.0

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del análisis."""
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    models_used: List[str] = field(default_factory=list)
    worker_id: Optional[str] = None
    memory_usage_mb: Optional[float] = None

@dataclass
class NLPAnalysisResult:
    """Resultado completo del análisis NLP."""
    
    # Identificación
    id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest())
    timestamp: datetime = field(default_factory=datetime.now)
    status: AnalysisStatus = AnalysisStatus.PENDING
    
    # Métricas de análisis
    basic: BasicMetrics = field(default_factory=BasicMetrics)
    sentiment: SentimentMetrics = field(default_factory=SentimentMetrics)
    readability: ReadabilityMetrics = field(default_factory=ReadabilityMetrics)
    keywords: KeywordMetrics = field(default_factory=KeywordMetrics)
    language: LanguageMetrics = field(default_factory=LanguageMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Metadata adicional
    options: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'basic': {
                'word_count': self.basic.word_count,
                'sentence_count': self.basic.sentence_count,
                'paragraph_count': self.basic.paragraph_count,
                'character_count': self.basic.character_count,
                'avg_words_per_sentence': self.basic.avg_words_per_sentence
            },
            'sentiment': {
                'polarity': self.sentiment.polarity,
                'subjectivity': self.sentiment.subjectivity,
                'confidence': self.sentiment.confidence,
                'label': self.sentiment.label,
                'score': self.sentiment.score
            },
            'readability': {
                'flesch_reading_ease': self.readability.flesch_reading_ease,
                'flesch_kincaid_grade': self.readability.flesch_kincaid_grade,
                'score': self.readability.score,
                'level': self.readability.level
            },
            'keywords': {
                'keywords': self.keywords.keywords,
                'total_keywords': self.keywords.total_keywords,
                'avg_score': self.keywords.avg_score
            },
            'language': {
                'detected_language': self.language.detected_language,
                'confidence': self.language.confidence
            },
            'quality': {
                'overall_score': self.quality.overall_score,
                'level': self.quality.level.value,
                'content_score': self.quality.content_score,
                'structure_score': self.quality.structure_score,
                'engagement_score': self.quality.engagement_score,
                'seo_score': self.quality.seo_score
            },
            'performance': {
                'processing_time_ms': self.performance.processing_time_ms,
                'cache_hit': self.performance.cache_hit,
                'models_used': self.performance.models_used,
                'memory_usage_mb': self.performance.memory_usage_mb
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NLPAnalysisResult':
        """Crear resultado desde diccionario."""
        result = cls()
        
        if 'id' in data:
            result.id = data['id']
        if 'timestamp' in data:
            result.timestamp = datetime.fromisoformat(data['timestamp'])
        if 'status' in data:
            result.status = AnalysisStatus(data['status'])
        
        # Básicas
        if 'basic' in data:
            basic_data = data['basic']
            result.basic = BasicMetrics(
                word_count=basic_data.get('word_count', 0),
                sentence_count=basic_data.get('sentence_count', 0),
                paragraph_count=basic_data.get('paragraph_count', 0),
                character_count=basic_data.get('character_count', 0),
                avg_words_per_sentence=basic_data.get('avg_words_per_sentence', 0.0)
            )
        
        # Sentimientos
        if 'sentiment' in data:
            sent_data = data['sentiment']
            result.sentiment = SentimentMetrics(
                polarity=sent_data.get('polarity', 0.0),
                subjectivity=sent_data.get('subjectivity', 0.0),
                confidence=sent_data.get('confidence', 0.0),
                label=sent_data.get('label', 'neutral'),
                score=sent_data.get('score', 50.0)
            )
        
        # Legibilidad
        if 'readability' in data:
            read_data = data['readability']
            result.readability = ReadabilityMetrics(
                flesch_reading_ease=read_data.get('flesch_reading_ease', 0.0),
                flesch_kincaid_grade=read_data.get('flesch_kincaid_grade', 0.0),
                score=read_data.get('score', 50.0),
                level=read_data.get('level', 'average')
            )
        
        return result
    
    def get_quality_level(self) -> QualityLevel:
        """Determinar nivel de calidad basado en score general."""
        score = self.quality.overall_score
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.AVERAGE
        else:
            return QualityLevel.POOR
    
    def add_error(self, error: str):
        """Agregar error al resultado."""
        self.errors.append(error)
        if self.status == AnalysisStatus.PROCESSING:
            self.status = AnalysisStatus.ERROR
    
    def add_warning(self, warning: str):
        """Agregar warning al resultado."""
        self.warnings.append(warning)
    
    def add_recommendation(self, recommendation: str):
        """Agregar recomendación al resultado."""
        self.recommendations.append(recommendation)
    
    def mark_completed(self) -> Any:
        """Marcar análisis como completado."""
        self.status = AnalysisStatus.COMPLETED
        self.quality.level = self.get_quality_level()

@dataclass
class AnalysisRequest:
    """Solicitud de análisis NLP."""
    text: str
    options: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical
    callback_url: Optional[str] = None
    user_id: Optional[str] = None
    
    def get_cache_key(self) -> str:
        """Generar key de cache para la solicitud."""
        try:
            content = f"{self.text}:{orjson.dumps(self.options, sort_keys=True).decode()}"
        except:
            content = f"{self.text}:{str(sorted(self.options.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate(self) -> List[str]:
        """Validar la solicitud y devolver errores."""
        errors = []
        
        if not self.text or not self.text.strip():
            errors.append("Text cannot be empty")
        
        if len(self.text) > 100000:  # 100KB máximo
            errors.append("Text too long (max 100KB)")
        
        if self.priority not in [0, 1, 2]:
            errors.append("Priority must be 0, 1, or 2")
        
        return errors 