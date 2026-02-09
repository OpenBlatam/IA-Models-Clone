"""
Analizador de Estilo y Legibilidad
==================================

Sistema para analizar el estilo de escritura, legibilidad y calidad de documentos.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StyleAnalysis:
    """Resultado de análisis de estilo"""
    total_words: int
    total_sentences: int
    total_paragraphs: int
    avg_words_per_sentence: float
    avg_sentences_per_paragraph: float
    avg_word_length: float
    readability_score: float
    complexity: str
    tone: str
    sentiment: Dict[str, float]
    vocabulary_richness: float
    punctuation_density: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class StyleAnalyzer:
    """
    Analizador de estilo de escritura
    
    Proporciona análisis de:
    - Legibilidad
    - Complejidad
    - Tono
    - Estadísticas de escritura
    - Calidad del texto
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador de estilo
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        logger.info("StyleAnalyzer inicializado")
    
    async def analyze_writing_style(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estilo de escritura
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de estilo
        """
        # Estadísticas básicas
        words = self._extract_words(content)
        sentences = self._extract_sentences(content)
        paragraphs = self._extract_paragraphs(content)
        
        total_words = len(words)
        total_sentences = len(sentences)
        total_paragraphs = len(paragraphs)
        
        # Promedios
        avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
        avg_sentences_per_paragraph = total_sentences / total_paragraphs if total_paragraphs > 0 else 0
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Score de legibilidad (Flesch Reading Ease simplificado)
        readability_score = self._calculate_readability(
            total_words,
            total_sentences,
            avg_words_per_sentence,
            avg_word_length
        )
        
        # Complejidad
        complexity = self._assess_complexity(
            avg_words_per_sentence,
            avg_word_length,
            readability_score
        )
        
        # Análisis de sentimiento
        sentiment = await self.analyzer.analyze_sentiment(content)
        
        # Tono
        tone = self._determine_tone(sentiment, avg_words_per_sentence)
        
        # Riqueza de vocabulario
        vocabulary_richness = len(set(words)) / total_words if total_words > 0 else 0
        
        # Densidad de puntuación
        punctuation_count = len(re.findall(r'[.,!?;:]', content))
        punctuation_density = punctuation_count / total_words if total_words > 0 else 0
        
        return StyleAnalysis(
            total_words=total_words,
            total_sentences=total_sentences,
            total_paragraphs=total_paragraphs,
            avg_words_per_sentence=avg_words_per_sentence,
            avg_sentences_per_paragraph=avg_sentences_per_paragraph,
            avg_word_length=avg_word_length,
            readability_score=readability_score,
            complexity=complexity,
            tone=tone,
            sentiment=sentiment,
            vocabulary_richness=vocabulary_richness,
            punctuation_density=punctuation_density
        ).__dict__
    
    def _extract_words(self, text: str) -> List[str]:
        """Extraer palabras del texto"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extraer oraciones del texto"""
        # Dividir por puntos, signos de exclamación e interrogación
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extraer párrafos del texto"""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _calculate_readability(
        self,
        total_words: int,
        total_sentences: int,
        avg_words_per_sentence: float,
        avg_word_length: float
    ) -> float:
        """
        Calcular score de legibilidad (0-100)
        Basado en Flesch Reading Ease simplificado
        """
        if total_sentences == 0:
            return 0.0
        
        # Fórmula simplificada
        # Score más alto = más legible
        sentence_score = 100 - (avg_words_per_sentence * 1.5)
        word_score = 100 - (avg_word_length * 10)
        
        readability = (sentence_score + word_score) / 2
        return max(0, min(100, readability))
    
    def _assess_complexity(
        self,
        avg_words_per_sentence: float,
        avg_word_length: float,
        readability_score: float
    ) -> str:
        """Evaluar complejidad del texto"""
        if readability_score < 30:
            return "very_complex"
        elif readability_score < 50:
            return "complex"
        elif readability_score < 70:
            return "moderate"
        elif readability_score < 90:
            return "simple"
        else:
            return "very_simple"
    
    def _determine_tone(
        self,
        sentiment: Dict[str, float],
        avg_words_per_sentence: float
    ) -> str:
        """Determinar tono del texto"""
        if not sentiment:
            return "neutral"
        
        # Determinar sentimiento dominante
        positive = sentiment.get("positive", 0)
        negative = sentiment.get("negative", 0)
        neutral_score = sentiment.get("neutral", 0)
        
        if positive > negative and positive > neutral_score:
            tone = "positive"
        elif negative > positive and negative > neutral_score:
            tone = "negative"
        else:
            tone = "neutral"
        
        # Ajustar por longitud de oraciones
        if avg_words_per_sentence > 25:
            tone += "_formal"
        elif avg_words_per_sentence < 10:
            tone += "_casual"
        
        return tone
    
    async def assess_quality(
        self,
        content: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluar calidad general del documento
        
        Args:
            content: Contenido del documento
            criteria: Criterios de evaluación (opcional)
        
        Returns:
            Diccionario con evaluación de calidad
        """
        style = await self.analyze_writing_style(content)
        
        quality_score = 0
        max_score = 100
        feedback = []
        
        # Legibilidad (30 puntos)
        readability = style["readability_score"]
        if readability >= 70:
            quality_score += 30
            feedback.append("✅ Excelente legibilidad")
        elif readability >= 50:
            quality_score += 20
            feedback.append("⚠️ Legibilidad moderada")
        else:
            quality_score += 10
            feedback.append("❌ Legibilidad baja - considera simplificar")
        
        # Complejidad apropiada (20 puntos)
        complexity = style["complexity"]
        if complexity == "moderate":
            quality_score += 20
            feedback.append("✅ Complejidad apropiada")
        elif complexity in ["simple", "very_simple"]:
            quality_score += 15
            feedback.append("⚠️ Texto muy simple - podría ser más profundo")
        else:
            quality_score += 10
            feedback.append("⚠️ Texto muy complejo - considera simplificar")
        
        # Riqueza de vocabulario (20 puntos)
        vocab_richness = style["vocabulary_richness"]
        if vocab_richness >= 0.5:
            quality_score += 20
            feedback.append("✅ Vocabulario rico y variado")
        elif vocab_richness >= 0.3:
            quality_score += 15
            feedback.append("⚠️ Vocabulario moderadamente variado")
        else:
            quality_score += 10
            feedback.append("⚠️ Vocabulario repetitivo")
        
        # Estructura (15 puntos)
        paragraphs = style["total_paragraphs"]
        sentences_per_para = style["avg_sentences_per_paragraph"]
        if paragraphs > 0 and 3 <= sentences_per_para <= 8:
            quality_score += 15
            feedback.append("✅ Estructura bien organizada")
        else:
            quality_score += 10
            feedback.append("⚠️ Revisa la estructura de párrafos")
        
        # Longitud apropiada (15 puntos)
        word_count = style["total_words"]
        if 200 <= word_count <= 2000:
            quality_score += 15
            feedback.append("✅ Longitud apropiada")
        elif word_count < 200:
            quality_score += 10
            feedback.append("⚠️ Documento muy corto")
        else:
            quality_score += 10
            feedback.append("⚠️ Documento muy largo - considera dividir")
        
        return {
            "quality_score": quality_score,
            "max_score": max_score,
            "percentage": (quality_score / max_score) * 100,
            "grade": self._score_to_grade(quality_score / max_score),
            "feedback": feedback,
            "style_analysis": style
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convertir score a calificación"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
















