"""
Rutas para Análisis de Sentimientos Avanzado
=============================================

Endpoints para análisis avanzado de sentimientos y emociones.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_sentiment import AdvancedSentimentAnalyzer
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/sentiment",
    tags=["Advanced Sentiment Analysis"]
)


class AdvancedSentimentRequest(BaseModel):
    """Request para análisis avanzado de sentimiento"""
    content: str = Field(..., description="Contenido del documento")
    split_into_sections: bool = Field(True, description="Dividir en secciones")


class CompareSentimentRequest(BaseModel):
    """Request para comparar sentimiento en el tiempo"""
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de documentos con timestamp y content"
    )


@router.post("/advanced")
async def analyze_advanced_sentiment(
    request: AdvancedSentimentRequest,
    analyzer = Depends(get_analyzer)
):
    """Analizar sentimiento avanzado con emociones"""
    try:
        sentiment_analyzer = AdvancedSentimentAnalyzer(analyzer)
        result = await sentiment_analyzer.analyze_advanced_sentiment(
            request.content,
            request.split_into_sections
        )
        
        return {
            "overall_sentiment": result.overall_sentiment,
            "sentiment_score": result.sentiment_score,
            "confidence": result.confidence,
            "emotions": {
                "joy": result.emotions.joy,
                "sadness": result.emotions.sadness,
                "anger": result.emotions.anger,
                "fear": result.emotions.fear,
                "surprise": result.emotions.surprise,
                "disgust": result.emotions.disgust,
                "dominant_emotion": result.emotions.dominant_emotion
            },
            "contextual_sentiment": result.contextual_sentiment,
            "intensity": result.intensity,
            "polarity_scores": result.polarity_scores,
            "timestamp": result.timestamp
        }
    except Exception as e:
        logger.error(f"Error analizando sentimiento avanzado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_sentiment_over_time(
    request: CompareSentimentRequest,
    analyzer = Depends(get_analyzer)
):
    """Comparar sentimiento a lo largo del tiempo"""
    try:
        sentiment_analyzer = AdvancedSentimentAnalyzer(analyzer)
        result = await sentiment_analyzer.compare_sentiment_over_time(
            request.documents
        )
        
        return result
    except Exception as e:
        logger.error(f"Error comparando sentimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















