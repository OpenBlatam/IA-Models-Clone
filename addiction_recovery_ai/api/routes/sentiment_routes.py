"""
Sentiment analysis routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.sentiment_service import SentimentService
except ImportError:
    from ...services.sentiment_service import SentimentService

router = APIRouter()

sentiment = SentimentService()


@router.post("/sentiment/analyze")
async def analyze_sentiment(
    text: str = Body(...),
    context: Optional[Dict] = Body(None)
):
    """Analiza el sentimiento de un texto"""
    try:
        analysis = sentiment.analyze_sentiment(text, context)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando sentimiento: {str(e)}")


@router.post("/sentiment/journal-entry")
async def analyze_journal_entry(
    user_id: str = Body(...),
    entry_text: str = Body(...),
    entry_date: str = Body(...)
):
    """Analiza una entrada de diario"""
    try:
        analysis = sentiment.analyze_journal_entry(user_id, entry_text, entry_date)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando entrada: {str(e)}")


@router.get("/sentiment/trend/{user_id}")
async def get_emotional_trend(user_id: str):
    """Obtiene tendencia emocional del usuario"""
    try:
        sentiment_data = []
        trend = sentiment.track_emotional_trend(user_id, sentiment_data)
        return JSONResponse(content=trend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo tendencia: {str(e)}")



