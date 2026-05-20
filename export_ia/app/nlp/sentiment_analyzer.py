"""
Sentiment Analyzer - Analizador de sentimientos
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import SentimentResult, SentimentType

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analizador de sentimientos."""
    
    def __init__(self):
        self._initialized = False
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "brilliant", "outstanding", "perfect", "love", "like",
            "happy", "joy", "pleasure", "satisfied", "content", "positive"
        }
        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "disgusting", "hate",
            "dislike", "angry", "sad", "disappointed", "frustrated", "negative",
            "worst", "pathetic", "useless", "stupid", "dumb", "annoying"
        }
    
    async def initialize(self):
        """Inicializar el analizador."""
        self._initialized = True
        logger.info("Sentiment Analyzer inicializado")
    
    async def shutdown(self):
        """Cerrar el analizador."""
        self._initialized = False
        logger.info("Sentiment Analyzer cerrado")
    
    async def analyze(self, text: str) -> SentimentResult:
        """Analizar sentimiento del texto."""
        if not self._initialized:
            await self.initialize()
        
        # Convertir a minúsculas
        text_lower = text.lower()
        
        # Contar palabras positivas y negativas
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Calcular puntuaciones
        total_words = len(text.split())
        positive_score = positive_count / total_words if total_words > 0 else 0
        negative_score = negative_count / total_words if total_words > 0 else 0
        neutral_score = 1 - positive_score - negative_score
        
        # Determinar sentimiento
        if positive_score > negative_score and positive_score > 0.1:
            sentiment = SentimentType.POSITIVE
            confidence = positive_score
        elif negative_score > positive_score and negative_score > 0.1:
            sentiment = SentimentType.NEGATIVE
            confidence = negative_score
        elif abs(positive_score - negative_score) < 0.05:
            sentiment = SentimentType.MIXED
            confidence = 0.5
        else:
            sentiment = SentimentType.NEUTRAL
            confidence = neutral_score
        
        # Calcular intensidad emocional
        emotional_intensity = (positive_score + negative_score) * 2
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            mixed_score=0.0,
            emotional_intensity=emotional_intensity
        )
    
    async def health_check(self) -> bool:
        """Verificar salud del analizador."""
        return self._initialized




