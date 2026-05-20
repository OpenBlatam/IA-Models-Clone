"""
Sentiment Analyzer - Sistema avanzado de análisis de sentimientos
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentType(Enum):
    """Tipos de sentimiento"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class SentimentResult:
    """Resultado de análisis de sentimiento"""
    sentiment: SentimentType
    score: float  # -1 a 1
    confidence: float  # 0 a 1
    positive_words: List[str]
    negative_words: List[str]
    neutral_words: List[str]


class AdvancedSentimentAnalyzer:
    """Analizador avanzado de sentimientos"""

    def __init__(self):
        """Inicializar analizador"""
        # Diccionarios de palabras
        self.positive_words = {
            'bueno', 'excelente', 'genial', 'fantástico', 'maravilloso',
            'perfecto', 'increíble', 'asombroso', 'feliz', 'alegre',
            'satisfecho', 'contento', 'agradable', 'positivo', 'éxito',
            'good', 'excellent', 'great', 'fantastic', 'wonderful',
            'perfect', 'amazing', 'happy', 'pleased', 'satisfied',
            'positive', 'success', 'love', 'like', 'enjoy'
        }
        
        self.negative_words = {
            'malo', 'terrible', 'horrible', 'pésimo', 'mal',
            'triste', 'deprimido', 'enojado', 'frustrado', 'negativo',
            'problema', 'error', 'fallo', 'fracaso', 'odio',
            'bad', 'terrible', 'horrible', 'awful', 'sad',
            'angry', 'frustrated', 'negative', 'problem', 'error',
            'failure', 'hate', 'dislike', 'worst'
        }
        
        # Intensificadores
        self.intensifiers = {
            'muy', 'extremadamente', 'súper', 'ultra', 'completamente',
            'totalmente', 'absolutamente', 'realmente', 'verdaderamente',
            'very', 'extremely', 'super', 'ultra', 'completely',
            'totally', 'absolutely', 'really', 'truly'
        }
        
        # Negaciones
        self.negations = {
            'no', 'nunca', 'jamás', 'tampoco', 'nada',
            'not', 'never', 'nothing', 'neither', 'none'
        }

    def analyze(
        self,
        text: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Analizar sentimiento del texto.

        Args:
            text: Texto a analizar
            detailed: Si incluir detalles

        Returns:
            Análisis de sentimiento
        """
        words = self._tokenize(text.lower())
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        positive_words_found = []
        negative_words_found = []
        neutral_words_found = []
        
        # Analizar cada palabra
        for i, word in enumerate(words):
            # Verificar si es palabra positiva
            if word in self.positive_words:
                # Verificar negación
                if i > 0 and words[i-1] in self.negations:
                    negative_count += 1
                    negative_words_found.append(f"{words[i-1]} {word}")
                else:
                    positive_count += 1
                    positive_words_found.append(word)
            
            # Verificar si es palabra negativa
            elif word in self.negative_words:
                # Verificar negación
                if i > 0 and words[i-1] in self.negations:
                    positive_count += 1
                    positive_words_found.append(f"{words[i-1]} {word}")
                else:
                    negative_count += 1
                    negative_words_found.append(word)
            
            else:
                neutral_count += 1
                if detailed:
                    neutral_words_found.append(word)
        
        # Calcular score (-1 a 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            score = (positive_count - negative_count) / total_sentiment_words
        else:
            score = 0.0
        
        # Determinar sentimiento
        if score > 0.3:
            sentiment = SentimentType.POSITIVE
        elif score < -0.3:
            sentiment = SentimentType.NEGATIVE
        elif abs(score) < 0.1:
            sentiment = SentimentType.NEUTRAL
        else:
            sentiment = SentimentType.MIXED
        
        # Calcular confianza
        confidence = min(abs(score), 1.0) if total_sentiment_words > 0 else 0.0
        
        result = {
            "sentiment": sentiment.value,
            "score": score,
            "confidence": confidence,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count
        }
        
        if detailed:
            result["positive_words"] = positive_words_found
            result["negative_words"] = negative_words_found
            result["neutral_words"] = neutral_words_found
        
        return result

    def analyze_by_sentences(self, text: str) -> Dict[str, Any]:
        """
        Analizar sentimiento por oraciones.

        Args:
            text: Texto

        Returns:
            Análisis por oraciones
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_sentiments = []
        for sentence in sentences:
            sentiment = self.analyze(sentence, detailed=False)
            sentence_sentiments.append({
                "sentence": sentence,
                "sentiment": sentiment
            })
        
        # Calcular sentimiento general
        avg_score = sum(s["sentiment"]["score"] for s in sentence_sentiments) / len(sentence_sentiments) if sentence_sentiments else 0.0
        
        return {
            "overall_sentiment": self._score_to_sentiment(avg_score),
            "overall_score": avg_score,
            "sentence_count": len(sentences),
            "sentences": sentence_sentiments
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizar texto"""
        return re.findall(r'\b\w+\b', text.lower())

    def _score_to_sentiment(self, score: float) -> str:
        """Convertir score a sentimiento"""
        if score > 0.3:
            return SentimentType.POSITIVE.value
        elif score < -0.3:
            return SentimentType.NEGATIVE.value
        elif abs(score) < 0.1:
            return SentimentType.NEUTRAL.value
        else:
            return SentimentType.MIXED.value






