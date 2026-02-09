"""
Document Sentiment Advanced - Análisis de Sentimiento Avanzado
===============================================================

Análisis de sentimiento avanzado con múltiples dimensiones.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class AdvancedSentiment:
    """Análisis de sentimiento avanzado."""
    overall_sentiment: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1.0 a 1.0
    confidence: float
    emotions: Dict[str, float] = field(default_factory=dict)  # 'joy', 'sadness', 'anger', etc.
    sentiment_by_section: Dict[str, str] = field(default_factory=dict)
    sentiment_trend: str = "stable"  # 'increasing', 'decreasing', 'stable'
    intensity: float = 0.0  # 0.0 a 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedSentimentAnalyzer:
    """Analizador de sentimiento avanzado."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
        
        # Diccionarios de sentimientos (simplificados)
        self.positive_words = {
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
            'happy', 'joy', 'pleasure', 'satisfaction', 'success', 'achievement'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'failure',
            'sad', 'angry', 'frustrated', 'disappointed', 'problem', 'issue'
        }
        
        self.emotion_words = {
            'joy': {'happy', 'joyful', 'delighted', 'cheerful', 'ecstatic'},
            'sadness': {'sad', 'depressed', 'melancholy', 'sorrowful', 'unhappy'},
            'anger': {'angry', 'furious', 'irritated', 'annoyed', 'enraged'},
            'fear': {'afraid', 'scared', 'worried', 'anxious', 'terrified'},
            'surprise': {'surprised', 'shocked', 'amazed', 'astonished', 'stunned'}
        }
    
    async def analyze_sentiment_advanced(
        self,
        content: str,
        analyze_by_section: bool = True
    ) -> AdvancedSentiment:
        """
        Analizar sentimiento avanzado.
        
        Args:
            content: Contenido del documento
            analyze_by_section: Analizar por secciones
        
        Returns:
            AdvancedSentiment con análisis completo
        """
        # Análisis básico
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0.0
        
        # Determinar sentimiento general
        if sentiment_score > 0.2:
            overall_sentiment = "positive"
        elif sentiment_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Calcular confianza
        confidence = min(1.0, total_sentiment_words / 10.0) if total_sentiment_words > 0 else 0.3
        
        # Analizar emociones
        emotions = self._analyze_emotions(content)
        
        # Análisis por secciones
        sentiment_by_section = {}
        if analyze_by_section:
            sections = self._split_into_sections(content)
            for section_name, section_content in sections.items():
                section_sentiment = await self.analyze_sentiment_advanced(
                    section_content, analyze_by_section=False
                )
                sentiment_by_section[section_name] = section_sentiment.overall_sentiment
        
        # Determinar tendencia
        sentiment_trend = self._determine_trend(sentiment_by_section)
        
        # Calcular intensidad
        intensity = abs(sentiment_score)
        
        return AdvancedSentiment(
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            confidence=confidence,
            emotions=emotions,
            sentiment_by_section=sentiment_by_section,
            sentiment_trend=sentiment_trend,
            intensity=intensity,
            metadata={
                "positive_words": positive_count,
                "negative_words": negative_count,
                "total_words": len(words)
            }
        )
    
    def _analyze_emotions(self, content: str) -> Dict[str, float]:
        """Analizar emociones."""
        words = content.lower().split()
        emotion_scores = {}
        
        for emotion, emotion_words in self.emotion_words.items():
            count = sum(1 for word in words if word in emotion_words)
            total_words = len(words)
            score = count / total_words if total_words > 0 else 0.0
            emotion_scores[emotion] = min(1.0, score * 10)  # Normalizar
        
        return emotion_scores
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Dividir contenido en secciones."""
        sections = {}
        
        # Buscar secciones por títulos
        lines = content.split('\n')
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            # Detectar títulos (Markdown o texto)
            if line.strip().startswith('#') or (line.strip().isupper() and len(line.strip()) < 100):
                if current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = line.strip().lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections if sections else {"Full Document": content}
    
    def _determine_trend(self, sentiment_by_section: Dict[str, str]) -> str:
        """Determinar tendencia de sentimiento."""
        if not sentiment_by_section or len(sentiment_by_section) < 2:
            return "stable"
        
        sentiments = list(sentiment_by_section.values())
        
        # Convertir a números
        sentiment_nums = []
        for sent in sentiments:
            if sent == "positive":
                sentiment_nums.append(1)
            elif sent == "negative":
                sentiment_nums.append(-1)
            else:
                sentiment_nums.append(0)
        
        # Calcular tendencia
        first_half = sentiment_nums[:len(sentiment_nums)//2]
        second_half = sentiment_nums[len(sentiment_nums)//2:]
        
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        
        if avg_second > avg_first + 0.2:
            return "increasing"
        elif avg_second < avg_first - 0.2:
            return "decreasing"
        else:
            return "stable"


__all__ = [
    "AdvancedSentimentAnalyzer",
    "AdvancedSentiment"
]



