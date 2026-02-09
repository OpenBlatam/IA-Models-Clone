"""
AI Topic Extraction Router
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from schemas import TopicExtractionInput, TopicResult
except ImportError:
    logging.warning("schemas module not available")
    TopicExtractionInput = Dict[str, Any]
    TopicResult = Dict[str, Any]

try:
    from services import extract_topics
except ImportError:
    logging.warning("services module not available")
    async def extract_topics(*args, **kwargs): return {}

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def extract_topics_endpoint(input_data: TopicExtractionInput) -> JSONResponse:
    """Extract topics from a collection of texts"""
    logger.info(f"Topic extraction requested - Texts: {len(input_data.texts)}, Topics: {input_data.num_topics}")
    
    try:
        result = await extract_topics(input_data.texts, input_data.num_topics)
        logger.info(f"Topic extraction completed - Topics found: {result.get('num_topics', 0)}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






