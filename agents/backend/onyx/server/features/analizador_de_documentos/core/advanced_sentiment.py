"""
Análisis de Sentimientos Avanzado
==================================

Sistema mejorado de análisis de sentimientos con contexto y emociones.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class EmotionAnalysis:
    """Análisis de emociones"""
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    dominant_emotion: str = "neutral"


@dataclass
class AdvancedSentimentResult:
    """Resultado de análisis de sentimiento avanzado"""
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1 a 1
    confidence: float
    emotions: EmotionAnalysis
    contextual_sentiment: Dict[str, float]  # Sentimiento por sección
    intensity: float  # 0-1
    polarity_scores: Dict[str, float]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class AdvancedSentimentAnalyzer:
    """
    Analizador de sentimientos avanzado
    
    Proporciona:
    - Análisis de emociones
    - Sentimiento contextual por secciones
    - Intensidad de sentimiento
    - Análisis de polaridad
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar analizador avanzado
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        self._emotion_keywords = self._load_emotion_keywords()
        logger.info("AdvancedSentimentAnalyzer inicializado")
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Cargar keywords de emociones"""
        return {
            "joy": ["alegría", "feliz", "contento", "satisfecho", "éxito", "triumfo", "celebración", "diversión"],
            "sadness": ["triste", "deprimido", "desolado", "melancólico", "pena", "dolor", "luto", "desesperación"],
            "anger": ["enfadado", "irritado", "furioso", "rabia", "indignado", "molesto", "airado", "colérico"],
            "fear": ["miedo", "terror", "pánico", "ansiedad", "preocupación", "nervioso", "asustado", "aterrorizado"],
            "surprise": ["sorpresa", "sorprendido", "asombro", "increíble", "impresionante", "inesperado", "impactante"],
            "disgust": ["disgusto", "repugnancia", "asco", "desagrado", "repulsión", "aversión", "rechazo"]
        }
    
    async def analyze_advanced_sentiment(
        self,
        content: str,
        split_into_sections: bool = True
    ) -> AdvancedSentimentResult:
        """
        Analizar sentimiento avanzado
        
        Args:
            content: Contenido del documento
            split_into_sections: Si True, analiza por secciones
        
        Returns:
            AdvancedSentimentResult con análisis completo
        """
        # Análisis básico de sentimiento
        basic_sentiment = await self.analyzer.analyze_sentiment(content)
        
        # Análisis de emociones
        emotions = self._analyze_emotions(content)
        
        # Análisis contextual
        contextual_sentiment = {}
        if split_into_sections:
            sections = self._split_into_sections(content)
            for i, section in enumerate(sections):
                section_sentiment = await self.analyzer.analyze_sentiment(section)
                contextual_sentiment[f"section_{i+1}"] = self._calculate_sentiment_score(section_sentiment)
        
        # Calcular score general
        sentiment_score = self._calculate_sentiment_score(basic_sentiment)
        
        # Determinar sentimiento general
        if sentiment_score > 0.1:
            overall_sentiment = "positive"
        elif sentiment_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Calcular intensidad
        intensity = abs(sentiment_score)
        
        # Scores de polaridad
        polarity_scores = {
            "positive": basic_sentiment.get("positive", 0),
            "negative": basic_sentiment.get("negative", 0),
            "neutral": basic_sentiment.get("neutral", 0)
        }
        
        # Confianza
        confidence = max(polarity_scores.values())
        
        return AdvancedSentimentResult(
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            confidence=confidence,
            emotions=emotions,
            contextual_sentiment=contextual_sentiment,
            intensity=intensity,
            polarity_scores=polarity_scores
        )
    
    def _analyze_emotions(self, content: str) -> EmotionAnalysis:
        """Analizar emociones en el texto"""
        content_lower = content.lower()
        emotion_scores = {}
        
        for emotion, keywords in self._emotion_keywords.items():
            score = 0
            for keyword in keywords:
                # Contar ocurrencias
                count = content_lower.count(keyword)
                score += count
            
            # Normalizar (máximo 10 keywords = score máximo)
            emotion_scores[emotion] = min(1.0, score / 10.0)
        
        # Determinar emoción dominante
        dominant = max(emotion_scores.items(), key=lambda x: x[1]) if emotion_scores else ("neutral", 0.0)
        dominant_emotion = dominant[0] if dominant[1] > 0.1 else "neutral"
        
        return EmotionAnalysis(
            joy=emotion_scores.get("joy", 0.0),
            sadness=emotion_scores.get("sadness", 0.0),
            anger=emotion_scores.get("anger", 0.0),
            fear=emotion_scores.get("fear", 0.0),
            surprise=emotion_scores.get("surprise", 0.0),
            disgust=emotion_scores.get("disgust", 0.0),
            dominant_emotion=dominant_emotion
        )
    
    def _split_into_sections(self, content: str, max_section_length: int = 500) -> List[str]:
        """Dividir contenido en secciones"""
        # Dividir por párrafos
        paragraphs = content.split('\n\n')
        
        sections = []
        current_section = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_length = len(paragraph)
            if current_length + para_length > max_section_length and current_section:
                sections.append('\n\n'.join(current_section))
                current_section = [paragraph]
                current_length = para_length
            else:
                current_section.append(paragraph)
                current_length += para_length
        
        if current_section:
            sections.append('\n\n'.join(current_section))
        
        return sections if sections else [content]
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, float]) -> float:
        """Calcular score de sentimiento (-1 a 1)"""
        positive = sentiment.get("positive", 0)
        negative = sentiment.get("negative", 0)
        neutral = sentiment.get("neutral", 0)
        
        if positive + negative > 0:
            return (positive - negative) / (positive + negative + neutral)
        return 0.0
    
    async def compare_sentiment_over_time(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comparar sentimiento a lo largo del tiempo
        
        Args:
            documents: Lista de documentos con timestamp y content
        
        Returns:
            Análisis de evolución de sentimiento
        """
        results = []
        
        for doc in documents:
            sentiment_result = await self.analyze_advanced_sentiment(doc["content"])
            results.append({
                "timestamp": doc.get("timestamp"),
                "sentiment_score": sentiment_result.sentiment_score,
                "overall_sentiment": sentiment_result.overall_sentiment,
                "intensity": sentiment_result.intensity,
                "dominant_emotion": sentiment_result.emotions.dominant_emotion
            })
        
        # Calcular tendencia
        if len(results) > 1:
            scores = [r["sentiment_score"] for r in results]
            trend = "increasing" if scores[-1] > scores[0] else "decreasing" if scores[-1] < scores[0] else "stable"
        else:
            trend = "stable"
        
        return {
            "results": results,
            "trend": trend,
            "average_sentiment": sum(r["sentiment_score"] for r in results) / len(results) if results else 0.0
        }
















