"""
AI Semantic Similarity Router
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from schemas import SimilarityInput, SemanticSimilarityResult
except ImportError:
    logging.warning("schemas module not available")
    SimilarityInput = Dict[str, Any]
    SemanticSimilarityResult = Dict[str, Any]

try:
    from services import calculate_semantic_similarity
except ImportError:
    logging.warning("services module not available")
    async def calculate_semantic_similarity(*args, **kwargs): return {}

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def calculate_semantic_similarity_endpoint(input_data: SimilarityInput) -> JSONResponse:
    """Calculate semantic similarity between two texts using AI/ML"""
    logger.info(f"Semantic similarity requested - Text1: {len(input_data.text1)}, Text2: {len(input_data.text2)}")
    
    try:
        result = await calculate_semantic_similarity(input_data.text1, input_data.text2)
        logger.info(f"Semantic similarity completed - Score: {result.get('similarity_score', 0):.3f}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Semantic similarity error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






