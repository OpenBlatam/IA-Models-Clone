"""
AI Sentiment Analysis Router
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from schemas import ContentInput, SentimentResult
except ImportError:
    logging.warning("schemas module not available")
    ContentInput = Dict[str, Any]
    SentimentResult = Dict[str, Any]

try:
    from services import analyze_sentiment
except ImportError:
    logging.warning("services module not available")
    async def analyze_sentiment(*args, **kwargs): return {}

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def analyze_sentiment_endpoint(input_data: ContentInput) -> JSONResponse:
    """Analyze sentiment of content using AI/ML"""
    logger.info(f"Sentiment analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await analyze_sentiment(input_data.content)
        logger.info(f"Sentiment analysis completed - Sentiment: {result.get('dominant_sentiment', 'unknown')}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






