"""
AI/ML Router - Consolidated AI/ML endpoints
"""

import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from schemas import (
        ContentInput, LanguageResult, EntityResult,
        SummaryInput, ReadabilityResult, ComprehensiveAnalysisResult,
        BatchAnalysisResult, BatchAnalysisInput
    )
except ImportError:
    logging.warning("schemas module not available")
    ContentInput = Dict[str, Any]
    LanguageResult = Dict[str, Any]
    EntityResult = Dict[str, Any]
    SummaryInput = Dict[str, Any]
    ReadabilityResult = Dict[str, Any]
    ComprehensiveAnalysisResult = Dict[str, Any]
    BatchAnalysisResult = Dict[str, Any]
    BatchAnalysisInput = Dict[str, Any]

try:
    from services import (
        detect_language, extract_entities, generate_summary,
        analyze_readability_advanced, comprehensive_analysis,
        batch_analyze_content
    )
except ImportError:
    logging.warning("services module not available")
    async def detect_language(*args, **kwargs): return {}
    async def extract_entities(*args, **kwargs): return {}
    async def generate_summary(*args, **kwargs): return {}
    async def analyze_readability_advanced(*args, **kwargs): return {}
    async def comprehensive_analysis(*args, **kwargs): return {}
    async def batch_analyze_content(*args, **kwargs): return []

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ai", tags=["AI/ML"])


@router.post("/language", response_model=Dict[str, Any])
async def detect_language_endpoint(input_data: ContentInput) -> JSONResponse:
    """Detect language of content"""
    logger.info(f"Language detection requested - Length: {len(input_data.content)}")
    
    try:
        result = await detect_language(input_data.content)
        logger.info(f"Language detection completed - Language: {result.get('language', 'unknown')}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/entities", response_model=Dict[str, Any])
async def extract_entities_endpoint(input_data: ContentInput) -> JSONResponse:
    """Extract named entities from content"""
    logger.info(f"Entity extraction requested - Length: {len(input_data.content)}")
    
    try:
        result = await extract_entities(input_data.content)
        logger.info(f"Entity extraction completed - Entities: {result.get('entity_count', 0)}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/summary", response_model=Dict[str, Any])
async def generate_summary_endpoint(input_data: SummaryInput) -> JSONResponse:
    """Generate summary of content using AI/ML"""
    logger.info(f"Text summarization requested - Length: {len(input_data.content) if hasattr(input_data, 'content') else 'unknown'}")
    
    try:
        content = input_data.content if hasattr(input_data, 'content') else input_data.get('content', '')
        max_length = input_data.max_length if hasattr(input_data, 'max_length') else input_data.get('max_length', 150)
        
        result = await generate_summary(content, max_length)
        logger.info(f"Text summarization completed - Summary length: {result.get('summary_length', 0)}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text summarization error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/readability", response_model=Dict[str, Any])
async def analyze_readability_advanced_endpoint(input_data: ContentInput) -> JSONResponse:
    """Advanced readability analysis using AI/ML"""
    logger.info(f"Advanced readability analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await analyze_readability_advanced(input_data.content)
        logger.info(f"Advanced readability analysis completed - Grade level: {result.get('grade_level', 0):.1f}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Advanced readability analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/comprehensive", response_model=Dict[str, Any])
async def comprehensive_analysis_endpoint(input_data: ContentInput) -> JSONResponse:
    """Perform comprehensive analysis combining all AI/ML features"""
    logger.info(f"Comprehensive analysis requested - Length: {len(input_data.content)}")
    
    try:
        result = await comprehensive_analysis(input_data.content)
        logger.info(f"Comprehensive analysis completed - Hash: {result.get('text_hash', 'unknown')}")
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=Dict[str, Any])
async def batch_analyze_content_endpoint(input_data: BatchAnalysisInput) -> JSONResponse:
    """Analyze multiple texts in batch for efficiency"""
    logger.info(f"Batch analysis requested - Texts: {len(input_data.texts)}")
    
    try:
        results = await batch_analyze_content(input_data.texts)
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        
        logger.info(f"Batch analysis completed - Successful: {successful}, Failed: {failed}")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "results": results,
                "total_processed": len(input_data.texts),
                "successful_analyses": successful,
                "failed_analyses": failed,
                "timestamp": time.time()
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

