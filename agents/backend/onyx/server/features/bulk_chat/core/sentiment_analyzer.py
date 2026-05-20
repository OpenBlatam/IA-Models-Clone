"""
Sentiment Analyzer - Analizador de Sentimientos
===============================================

Sistema avanzado de análisis de sentimientos con detección de emociones y polaridad.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class Sentiment(Enum):
    """Sentimiento."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Emotion(Enum):
    """Emoción."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


@dataclass
class SentimentResult:
    """Resultado de análisis de sentimiento."""
    sentiment: Sentiment
    polarity: float  # -1.0 a 1.0
    confidence: float  # 0.0 a 1.0
    emotions: Dict[str, float] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.now)


class SentimentAnalyzer:
    """Analizador de sentimientos."""
    
    def __init__(self):
        # Diccionarios de palabras (simplificado, en producción usar modelos ML)
        self.positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "happy", "love", "like", "best", "perfect", "awesome", "brilliant",
            "success", "win", "beautiful", "delightful", "pleased", "satisfied",
        }
        
        self.negative_words = {
            "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
            "fail", "error", "problem", "issue", "difficult", "disappointed",
            "angry", "frustrated", "sad", "upset", "annoying", "broken",
        }
        
        self.emotion_keywords = {
            Emotion.JOY: {"happy", "joy", "excited", "delighted", "celebrate", "smile"},
            Emotion.SADNESS: {"sad", "unhappy", "depressed", "cry", "tears", "lonely"},
            Emotion.ANGER: {"angry", "mad", "furious", "rage", "annoyed", "irritated"},
            Emotion.FEAR: {"afraid", "scared", "worried", "anxious", "fear", "nervous"},
            Emotion.SURPRISE: {"surprised", "shocked", "amazed", "wow", "unexpected"},
            Emotion.DISGUST: {"disgusted", "revolted", "sick", "gross", "nasty"},
            Emotion.TRUST: {"trust", "confident", "reliable", "secure", "safe"},
            Emotion.ANTICIPATION: {"excited", "eager", "hopeful", "expecting", "waiting"},
        }
    
    def analyze(self, text: str) -> SentimentResult:
        """Analizar sentimiento de texto."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Calcular polaridad
        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_count = sum(1 for w in words if w in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            polarity = 0.0
            sentiment = Sentiment.NEUTRAL
            confidence = 0.3
        else:
            polarity = (positive_count - negative_count) / total_sentiment_words
            polarity = max(-1.0, min(1.0, polarity))
            
            if polarity > 0.1:
                sentiment = Sentiment.POSITIVE
            elif polarity < -0.1:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL
            
            confidence = min(1.0, total_sentiment_words / max(len(words), 1) * 2)
        
        # Detectar emociones
        emotions = self._detect_emotions(text_lower, words)
        
        # Extraer keywords
        keywords = self._extract_keywords(words)
        
        return SentimentResult(
            sentiment=sentiment,
            polarity=polarity,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords,
        )
    
    def _detect_emotions(self, text: str, words: List[str]) -> Dict[str, float]:
        """Detectar emociones."""
        emotions = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            matches = sum(1 for w in words if w in keywords)
            if matches > 0:
                score = min(1.0, matches / len(keywords) * 2)
                emotions[emotion.value] = score
        
        return emotions
    
    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extraer palabras clave."""
        all_sentiment_words = self.positive_words | self.negative_words
        keywords = [w for w in words if w in all_sentiment_words]
        return list(set(keywords))[:10]  # Top 10 únicos
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analizar lote de textos."""
        results = []
        for text in texts:
            result = self.analyze(text)
            results.append(result)
        return results
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Obtener resumen de sentimientos."""
        sentiment_counts: Dict[str, int] = defaultdict(int)
        total_polarity = 0.0
        emotion_totals: Dict[str, float] = defaultdict(float)
        
        for result in results:
            sentiment_counts[result.sentiment.value] += 1
            total_polarity += result.polarity
            
            for emotion, score in result.emotions.items():
                emotion_totals[emotion] += score
        
        avg_polarity = total_polarity / len(results) if results else 0.0
        
        # Normalizar emociones
        if emotion_totals:
            max_emotion_score = max(emotion_totals.values())
            if max_emotion_score > 0:
                emotion_totals = {
                    k: v / max_emotion_score
                    for k, v in emotion_totals.items()
                }
        
        return {
            "total_analyses": len(results),
            "sentiment_distribution": dict(sentiment_counts),
            "average_polarity": avg_polarity,
            "dominant_emotions": dict(sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)[:5]),
        }
















