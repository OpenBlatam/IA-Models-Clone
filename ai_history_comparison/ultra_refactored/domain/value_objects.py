"""
Value Objects - Objetos de Valor
===============================

Objetos de valor inmutables que representan conceptos del dominio
sin identidad propia.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List
from enum import Enum
import re


class ContentMetrics(BaseModel):
    """
    Métricas de contenido de una entrada de historial.
    
    Objeto de valor que encapsula las métricas de análisis de contenido.
    """
    word_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=0)
    character_count: int = Field(..., ge=0)
    paragraph_count: int = Field(..., ge=0)
    avg_word_length: float = Field(..., ge=0.0)
    avg_sentence_length: float = Field(..., ge=0.0)
    unique_words: int = Field(..., ge=0)
    vocabulary_richness: float = Field(..., ge=0.0, le=1.0)
    
    @validator('vocabulary_richness')
    def validate_vocabulary_richness(cls, v, values):
        """Validar que la riqueza del vocabulario sea consistente."""
        if 'unique_words' in values and 'word_count' in values:
            unique_words = values['unique_words']
            word_count = values['word_count']
            if word_count > 0:
                expected_richness = unique_words / word_count
                if abs(v - expected_richness) > 0.1:  # Tolerancia del 10%
                    pass  # Permitir discrepancias menores
        return v
    
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
    
    class Config:
        """Configuración del modelo."""
        frozen = True  # Hacer el objeto inmutable


class QualityScore(BaseModel):
    """
    Score de calidad de contenido.
    
    Objeto de valor que representa la calidad de una entrada.
    """
    overall_score: float = Field(..., ge=0.0, le=1.0)
    readability_score: float = Field(..., ge=0.0, le=1.0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    
    @validator('overall_score')
    def validate_overall_score(cls, v, values):
        """Validar que el score overall sea consistente con los scores individuales."""
        if all(key in values for key in ['readability_score', 'coherence_score', 'relevance_score', 'completeness_score', 'accuracy_score']):
            individual_scores = [
                values['readability_score'],
                values['coherence_score'],
                values['relevance_score'],
                values['completeness_score'],
                values['accuracy_score']
            ]
            expected_overall = sum(individual_scores) / len(individual_scores)
            if abs(v - expected_overall) > 0.2:  # Tolerancia del 20%
                pass  # Permitir discrepancias menores
        return v
    
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
    
    class Config:
        """Configuración del modelo."""
        frozen = True


class SimilarityScore(BaseModel):
    """
    Score de similitud entre dos entradas.
    
    Objeto de valor que representa la similitud entre contenidos.
    """
    overall_similarity: float = Field(..., ge=0.0, le=1.0)
    content_similarity: float = Field(..., ge=0.0, le=1.0)
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    structural_similarity: float = Field(..., ge=0.0, le=1.0)
    style_similarity: float = Field(..., ge=0.0, le=1.0)
    
    @validator('overall_similarity')
    def validate_overall_similarity(cls, v, values):
        """Validar que el score overall sea consistente."""
        if all(key in values for key in ['content_similarity', 'semantic_similarity', 'structural_similarity', 'style_similarity']):
            individual_scores = [
                values['content_similarity'],
                values['semantic_similarity'],
                values['structural_similarity'],
                values['style_similarity']
            ]
            expected_overall = sum(individual_scores) / len(individual_scores)
            if abs(v - expected_overall) > 0.2:  # Tolerancia del 20%
                pass  # Permitir discrepancias menores
        return v
    
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
    
    class Config:
        """Configuración del modelo."""
        frozen = True


class SentimentAnalysis(BaseModel):
    """
    Análisis de sentimiento de contenido.
    
    Objeto de valor que representa el análisis de sentimiento.
    """
    polarity: float = Field(..., ge=-1.0, le=1.0)
    subjectivity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    
    @validator('positive_score', 'negative_score', 'neutral_score')
    def validate_sentiment_scores(cls, v, values):
        """Validar que los scores de sentimiento sumen 1.0."""
        if all(key in values for key in ['positive_score', 'negative_score', 'neutral_score']):
            total = values['positive_score'] + values['negative_score'] + values['neutral_score']
            if abs(total - 1.0) > 0.1:  # Tolerancia del 10%
                pass  # Permitir discrepancias menores
        return v
    
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
    
    class Config:
        """Configuración del modelo."""
        frozen = True


class TextComplexity(BaseModel):
    """
    Complejidad del texto.
    
    Objeto de valor que representa la complejidad del contenido.
    """
    flesch_reading_ease: float = Field(..., ge=0.0, le=100.0)
    flesch_kincaid_grade: float = Field(..., ge=0.0, le=20.0)
    gunning_fog_index: float = Field(..., ge=0.0, le=20.0)
    smog_index: float = Field(..., ge=0.0, le=20.0)
    automated_readability_index: float = Field(..., ge=0.0, le=20.0)
    
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
    
    class Config:
        """Configuración del modelo."""
        frozen = True




