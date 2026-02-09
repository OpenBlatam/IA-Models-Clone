"""
API de Análisis de Sentimiento

Endpoints para:
- Analizar sentimiento de texto
- Analizar sentimiento de audio
- Análisis en batch
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.sentiment_analysis import get_sentiment_service
from services.audio_transcription import get_transcription_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sentiment",
    tags=["sentiment"]
)


@router.post("/analyze-text")
async def analyze_text_sentiment(
    text: str = Body(..., description="Texto a analizar"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de un texto.
    """
    try:
        sentiment_service = get_sentiment_service()
        result = sentiment_service.analyze_text(text)
        
        return {
            "text": text,
            "sentiment": result.label.value,
            "score": result.score,
            "polarity": result.polarity,
            "emotions": result.emotions
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )


@router.post("/analyze-audio")
async def analyze_audio_sentiment(
    file: UploadFile = File(..., description="Archivo de audio"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de un archivo de audio (vía transcripción).
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            sentiment_service = get_sentiment_service()
            transcription_service = get_transcription_service()
            
            result = sentiment_service.analyze_audio(tmp_path, transcription_service)
            
            return {
                "file": file.filename,
                "sentiment": result.label.value,
                "score": result.score,
                "polarity": result.polarity,
                "emotions": result.emotions
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error analyzing audio sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing audio sentiment: {str(e)}"
        )


@router.post("/analyze-batch")
async def analyze_batch_sentiment(
    texts: List[str] = Body(..., description="Lista de textos a analizar"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza el sentimiento de múltiples textos.
    """
    try:
        sentiment_service = get_sentiment_service()
        results = sentiment_service.analyze_batch(texts)
        
        distribution = sentiment_service.get_sentiment_distribution(results)
        
        return {
            "results": [
                {
                    "text": text,
                    "sentiment": result.label.value,
                    "score": result.score,
                    "polarity": result.polarity
                }
                for text, result in zip(texts, results)
            ],
            "distribution": distribution,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing batch sentiment: {str(e)}"
        )

