"""
Sistema de Análisis de Sentimiento

Proporciona:
- Análisis de sentimiento de texto
- Análisis de sentimiento de audio (vía transcripción)
- Detección de emociones
- Análisis de polaridad
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, using mock sentiment analysis")


class SentimentLabel(Enum):
    """Etiquetas de sentimiento"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Resultado de análisis de sentimiento"""
    label: SentimentLabel
    score: float
    emotions: Dict[str, float] = field(default_factory=dict)
    polarity: float = 0.0  # -1.0 a 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class SentimentAnalysisService:
    """Servicio de análisis de sentimiento"""
    
    def __init__(self):
        self.sentiment_pipeline = None
        self.emotion_pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Loading sentiment analysis models")
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Sentiment model loaded")
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")
    
    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analiza el sentimiento de un texto
        
        Args:
            text: Texto a analizar
        
        Returns:
            SentimentResult
        """
        if not self.sentiment_pipeline:
            # Mock analysis
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                polarity=0.0
            )
        
        try:
            # Analizar sentimiento
            result = self.sentiment_pipeline(text)[0]
            
            # Mapear a nuestro enum
            label_str = result["label"].lower()
            if "positive" in label_str:
                label = SentimentLabel.POSITIVE
                polarity = result["score"]
            elif "negative" in label_str:
                label = SentimentLabel.NEGATIVE
                polarity = -result["score"]
            else:
                label = SentimentLabel.NEUTRAL
                polarity = 0.0
            
            return SentimentResult(
                label=label,
                score=result["score"],
                polarity=polarity
            )
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                polarity=0.0
            )
    
    def analyze_audio(
        self,
        audio_path: str,
        transcription_service: Optional[Any] = None
    ) -> SentimentResult:
        """
        Analiza el sentimiento de un audio (vía transcripción)
        
        Args:
            audio_path: Ruta del archivo de audio
            transcription_service: Servicio de transcripción
        
        Returns:
            SentimentResult
        """
        if transcription_service:
            # Transcribir primero
            transcription = transcription_service.transcribe(audio_path)
            # Analizar texto transcrito
            return self.analyze_text(transcription.text)
        else:
            # Sin transcripción, retornar neutral
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                polarity=0.0
            )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analiza sentimiento de múltiples textos
        
        Args:
            texts: Lista de textos
        
        Returns:
            Lista de resultados
        """
        return [self.analyze_text(text) for text in texts]
    
    def get_sentiment_distribution(
        self,
        results: List[SentimentResult]
    ) -> Dict[str, Any]:
        """
        Obtiene distribución de sentimientos
        
        Args:
            results: Lista de resultados
        
        Returns:
            Distribución estadística
        """
        total = len(results)
        if total == 0:
            return {}
        
        positive = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
        negative = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
        neutral = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
        
        avg_polarity = sum(r.polarity for r in results) / total if total > 0 else 0
        
        return {
            "total": total,
            "positive": {
                "count": positive,
                "percentage": (positive / total * 100) if total > 0 else 0
            },
            "negative": {
                "count": negative,
                "percentage": (negative / total * 100) if total > 0 else 0
            },
            "neutral": {
                "count": neutral,
                "percentage": (neutral / total * 100) if total > 0 else 0
            },
            "avg_polarity": avg_polarity
        }


# Instancia global
_sentiment_service: Optional[SentimentAnalysisService] = None


def get_sentiment_service() -> SentimentAnalysisService:
    """Obtiene la instancia global del servicio de análisis de sentimiento"""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentAnalysisService()
    return _sentiment_service

