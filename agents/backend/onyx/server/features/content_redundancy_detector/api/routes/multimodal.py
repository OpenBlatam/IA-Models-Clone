"""
Multimodal Router - Multimodal content analysis endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from multimodal_engine import multimodal_engine, MultimodalInput
except ImportError:
    logging.warning("multimodal_engine module not available")
    multimodal_engine = None
    MultimodalInput = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal", tags=["Multimodal"])


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_multimodal_content(input_data: Dict[str, Any]) -> JSONResponse:
    """Analyze multimodal content (text, image, audio, video)"""
    logger.info(f"Multimodal analysis requested - Type: {input_data.get('content_type', 'unknown')}")
    
    if not multimodal_engine:
        raise HTTPException(status_code=503, detail="Multimodal engine not available")
    
    try:
        if MultimodalInput:
            multimodal_input = MultimodalInput(**input_data)
            result = await multimodal_engine.analyze_multimodal(multimodal_input)
        else:
            result = await multimodal_engine.analyze_multimodal(input_data)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "content_type": result.content_type if hasattr(result, 'content_type') else input_data.get('content_type'),
                "content_hash": result.content_hash if hasattr(result, 'content_hash') else None,
                "analysis_results": result.analysis_results if hasattr(result, 'analysis_results') else result,
                "cross_modal_insights": result.cross_modal_insights if hasattr(result, 'cross_modal_insights') else None,
                "confidence_scores": result.confidence_scores if hasattr(result, 'confidence_scores') else None,
                "processing_time": result.processing_time if hasattr(result, 'processing_time') else None,
                "timestamp": result.timestamp if hasattr(result, 'timestamp') else None
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/image", response_model=Dict[str, Any])
async def analyze_image_content(input_data: Dict[str, Any]) -> JSONResponse:
    """Analyze image content specifically"""
    logger.info("Image analysis requested")
    
    if not multimodal_engine:
        raise HTTPException(status_code=503, detail="Multimodal engine not available")
    
    try:
        result = await multimodal_engine.analyze_image(input_data)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except Exception as e:
        logger.error(f"Error in image analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/audio", response_model=Dict[str, Any])
async def analyze_audio_content(input_data: Dict[str, Any]) -> JSONResponse:
    """Analyze audio content specifically"""
    logger.info("Audio analysis requested")
    
    if not multimodal_engine:
        raise HTTPException(status_code=503, detail="Multimodal engine not available")
    
    try:
        result = await multimodal_engine.analyze_audio(input_data)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/video", response_model=Dict[str, Any])
async def analyze_video_content(input_data: Dict[str, Any]) -> JSONResponse:
    """Analyze video content specifically"""
    logger.info("Video analysis requested")
    
    if not multimodal_engine:
        raise HTTPException(status_code=503, detail="Multimodal engine not available")
    
    try:
        result = await multimodal_engine.analyze_video(input_data)
        
        return JSONResponse(content={
            "success": True,
            "data": result,
            "error": None
        })
    except Exception as e:
        logger.error(f"Error in video analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






