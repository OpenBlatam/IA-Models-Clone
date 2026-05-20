"""
AI Plagiarism Detection Router
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from schemas import PlagiarismDetectionInput, PlagiarismResult
except ImportError:
    logging.warning("schemas module not available")
    PlagiarismDetectionInput = Dict[str, Any]
    PlagiarismResult = Dict[str, Any]

try:
    from services import detect_plagiarism
except ImportError:
    logging.warning("services module not available")
    async def detect_plagiarism(*args, **kwargs): return {}

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/", response_model=Dict[str, Any])
async def detect_plagiarism_endpoint(input_data: PlagiarismDetectionInput) -> JSONResponse:
    """Detect potential plagiarism in content"""
    logger.info(f"Plagiarism detection requested - Content: {len(input_data.content)}, References: {len(input_data.reference_texts)}")
    
    try:
        result = await detect_plagiarism(
            input_data.content,
            input_data.reference_texts,
            input_data.threshold
        )
        logger.info(f"Plagiarism detection completed - Plagiarized: {result.get('is_plagiarized', False)}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Plagiarism detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






