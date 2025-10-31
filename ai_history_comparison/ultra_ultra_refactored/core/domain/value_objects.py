"""
Value Objects - Objetos de Valor
==============================

Objetos de valor inmutables que representan conceptos del dominio
sin identidad propia.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
import uuid


class ModelType(str, Enum):
    """Tipos de modelos de IA soportados."""
    GPT_4 = "gpt-4"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    LLAMA_2_70B = "llama-2-70b"
    LLAMA_2_13B = "llama-2-13b"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ContentId:
    """
    Identificador único para contenido.
    
    Value object inmutable que representa un ID de contenido.
    """
    
    value: str
    
    def __post_init__(self):
        """Validar que el ID sea válido."""
        if not self.value or not isinstance(self.value, str):
            raise ValueError("ContentId must be a non-empty string")
        
        # Validar formato UUID
        try:
            uuid.UUID(self.value)
        except ValueError:
            raise ValueError("ContentId must be a valid UUID")
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return f"ContentId('{self.value}')"
    
    @classmethod
    def generate(cls) -> 'ContentId':
        """Generar un nuevo ContentId."""
        return cls(str(uuid.uuid4()))


@dataclass(frozen=True)
class QualityScore:
    """
    Score de calidad de contenido.
    
    Value object inmutable que representa la calidad de una entrada.
    """
    
    overall_score: float
    readability_score: float
    coherence_score: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    
    def __post_init__(self):
        """Validar que los scores sean válidos."""
        scores = [
            self.overall_score,
            self.readability_score,
            self.coherence_score,
            self.relevance_score,
            self.completeness_score,
            self.accuracy_score
        ]
        
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All quality scores must be between 0.0 and 1.0")
    
    @property
    def is_high_quality(self) -> bool:
        """Verificar si la calidad es alta."""
        return self.overall_score >= 0.8
    
    @property
    def is_acceptable_quality(self) -> bool:
        """Verificar si la calidad es aceptable."""
        return self.overall_score >= 0.6
    
    @property
    def needs_improvement(self) -> bool:
        """Verificar si necesita mejoras."""
        return self.overall_score < 0.6
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {
            "overall_score": self.overall_score,
            "readability_score": self.readability_score,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "accuracy_score": self.accuracy_score
        }


@dataclass(frozen=True)
class SimilarityScore:
    """
    Score de similitud entre contenidos.
    
    Value object inmutable que representa la similitud entre dos entradas.
    """
    
    overall_similarity: float
    content_similarity: float
    semantic_similarity: float
    structural_similarity: float
    style_similarity: float
    
    def __post_init__(self):
        """Validar que los scores sean válidos."""
        scores = [
            self.overall_similarity,
            self.content_similarity,
            self.semantic_similarity,
            self.structural_similarity,
            self.style_similarity
        ]
        
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All similarity scores must be between 0.0 and 1.0")
    
    @property
    def is_highly_similar(self) -> bool:
        """Verificar si es altamente similar."""
        return self.overall_similarity >= 0.8
    
    @property
    def is_moderately_similar(self) -> bool:
        """Verificar si es moderadamente similar."""
        return 0.5 <= self.overall_similarity < 0.8
    
    @property
    def is_dissimilar(self) -> bool:
        """Verificar si es disimilar."""
        return self.overall_similarity < 0.5
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario."""
        return {
            "overall_similarity": self.overall_similarity,
            "content_similarity": self.content_similarity,
            "semantic_similarity": self.semantic_similarity,
            "structural_similarity": self.structural_similarity,
            "style_similarity": self.style_similarity
        }


@dataclass(frozen=True)
class ContentMetrics:
    """
    Métricas de contenido de una entrada.
    
    Value object inmutable que encapsula las métricas de análisis de contenido.
    """
    
    word_count: int
    sentence_count: int
    character_count: int
    paragraph_count: int
    avg_word_length: float
    avg_sentence_length: float
    unique_words: int
    vocabulary_richness: float
    
    def __post_init__(self):
        """Validar que las métricas sean válidas."""
        if self.word_count < 0:
            raise ValueError("Word count cannot be negative")
        if self.sentence_count < 0:
            raise ValueError("Sentence count cannot be negative")
        if self.character_count < 0:
            raise ValueError("Character count cannot be negative")
        if self.paragraph_count < 0:
            raise ValueError("Paragraph count cannot be negative")
        if self.avg_word_length < 0:
            raise ValueError("Average word length cannot be negative")
        if self.avg_sentence_length < 0:
            raise ValueError("Average sentence length cannot be negative")
        if self.unique_words < 0:
            raise ValueError("Unique words cannot be negative")
        if not 0.0 <= self.vocabulary_richness <= 1.0:
            raise ValueError("Vocabulary richness must be between 0.0 and 1.0")
    
    @property
    def readability_score(self) -> float:
        """Calcular score de legibilidad basado en métricas."""
        if self.sentence_count == 0 or self.word_count == 0:
            return 0.0
        
        # Fórmula simplificada de legibilidad
        avg_sentence_length = self.word_count / self.sentence_count
        avg_syllables_per_word = self.avg_word_length / 2.5  # Aproximación
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(1.0, score / 100.0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "character_count": self.character_count,
            "paragraph_count": self.paragraph_count,
            "avg_word_length": self.avg_word_length,
            "avg_sentence_length": self.avg_sentence_length,
            "unique_words": self.unique_words,
            "vocabulary_richness": self.vocabulary_richness,
            "readability_score": self.readability_score
        }


@dataclass(frozen=True)
class SentimentAnalysis:
    """
    Análisis de sentimiento de contenido.
    
    Value object inmutable que representa el análisis de sentimiento.
    """
    
    polarity: float
    subjectivity: float
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    
    def __post_init__(self):
        """Validar que los scores sean válidos."""
        if not -1.0 <= self.polarity <= 1.0:
            raise ValueError("Polarity must be between -1.0 and 1.0")
        if not 0.0 <= self.subjectivity <= 1.0:
            raise ValueError("Subjectivity must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not 0.0 <= self.positive_score <= 1.0:
            raise ValueError("Positive score must be between 0.0 and 1.0")
        if not 0.0 <= self.negative_score <= 1.0:
            raise ValueError("Negative score must be between 0.0 and 1.0")
        if not 0.0 <= self.neutral_score <= 1.0:
            raise ValueError("Neutral score must be between 0.0 and 1.0")
        
        # Validar que los scores de sentimiento sumen 1.0
        total = self.positive_score + self.negative_score + self.neutral_score
        if abs(total - 1.0) > 0.1:  # Tolerancia del 10%
            raise ValueError("Sentiment scores must sum to 1.0")
    
    @property
    def is_positive(self) -> bool:
        """Verificar si el sentimiento es positivo."""
        return self.polarity > 0.1
    
    @property
    def is_negative(self) -> bool:
        """Verificar si el sentimiento es negativo."""
        return self.polarity < -0.1
    
    @property
    def is_neutral(self) -> bool:
        """Verificar si el sentimiento es neutral."""
        return -0.1 <= self.polarity <= 0.1
    
    @property
    def dominant_sentiment(self) -> str:
        """Obtener el sentimiento dominante."""
        if self.positive_score > self.negative_score and self.positive_score > self.neutral_score:
            return "positive"
        elif self.negative_score > self.positive_score and self.negative_score > self.neutral_score:
            return "negative"
        else:
            return "neutral"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
            "confidence": self.confidence,
            "positive_score": self.positive_score,
            "negative_score": self.negative_score,
            "neutral_score": self.neutral_score,
            "dominant_sentiment": self.dominant_sentiment
        }


@dataclass(frozen=True)
class TextComplexity:
    """
    Complejidad del texto.
    
    Value object inmutable que representa la complejidad del contenido.
    """
    
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    smog_index: float
    automated_readability_index: float
    
    def __post_init__(self):
        """Validar que los índices sean válidos."""
        if not 0.0 <= self.flesch_reading_ease <= 100.0:
            raise ValueError("Flesch Reading Ease must be between 0.0 and 100.0")
        if not 0.0 <= self.flesch_kincaid_grade <= 20.0:
            raise ValueError("Flesch-Kincaid Grade must be between 0.0 and 20.0")
        if not 0.0 <= self.gunning_fog_index <= 20.0:
            raise ValueError("Gunning Fog Index must be between 0.0 and 20.0")
        if not 0.0 <= self.smog_index <= 20.0:
            raise ValueError("SMOG Index must be between 0.0 and 20.0")
        if not 0.0 <= self.automated_readability_index <= 20.0:
            raise ValueError("Automated Readability Index must be between 0.0 and 20.0")
    
    @property
    def readability_level(self) -> str:
        """Obtener el nivel de legibilidad."""
        if self.flesch_reading_ease >= 90:
            return "very_easy"
        elif self.flesch_reading_ease >= 80:
            return "easy"
        elif self.flesch_reading_ease >= 70:
            return "fairly_easy"
        elif self.flesch_reading_ease >= 60:
            return "standard"
        elif self.flesch_reading_ease >= 50:
            return "fairly_difficult"
        elif self.flesch_reading_ease >= 30:
            return "difficult"
        else:
            return "very_difficult"
    
    @property
    def is_readable(self) -> bool:
        """Verificar si el texto es legible."""
        return self.flesch_reading_ease >= 60
    
    @property
    def is_accessible(self) -> bool:
        """Verificar si el texto es accesible."""
        return self.flesch_reading_ease >= 70
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "flesch_reading_ease": self.flesch_reading_ease,
            "flesch_kincaid_grade": self.flesch_kincaid_grade,
            "gunning_fog_index": self.gunning_fog_index,
            "smog_index": self.smog_index,
            "automated_readability_index": self.automated_readability_index,
            "readability_level": self.readability_level,
            "is_readable": self.is_readable,
            "is_accessible": self.is_accessible
        }


@dataclass(frozen=True)
class AnalysisResult:
    """
    Resultado de análisis.
    
    Value object inmutable que representa el resultado de un análisis.
    """
    
    analysis_type: str
    success: bool
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: str
    
    def __post_init__(self):
        """Validar que el resultado sea válido."""
        if not self.analysis_type or not isinstance(self.analysis_type, str):
            raise ValueError("Analysis type must be a non-empty string")
        if not isinstance(self.success, bool):
            raise ValueError("Success must be a boolean")
        if not isinstance(self.results, dict):
            raise ValueError("Results must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")
        if not self.timestamp or not isinstance(self.timestamp, str):
            raise ValueError("Timestamp must be a non-empty string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "analysis_type": self.analysis_type,
            "success": self.success,
            "results": self.results,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp
        }




