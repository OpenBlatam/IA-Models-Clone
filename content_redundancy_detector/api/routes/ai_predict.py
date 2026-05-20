"""
AI Predict Router - AI/ML prediction endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

try:
    from ai_ml_engine import ai_ml_engine
except ImportError:
    logging.warning("ai_ml_engine module not available")
    ai_ml_engine = None

from ...core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ai/predict", tags=["AI Predict"])


@router.post("/similarity", response_model=Dict[str, Any])
async def predict_similarity_ai(prediction_data: Dict[str, Any]) -> JSONResponse:
    """Predict similarity using AI/ML models"""
    logger.info("AI similarity prediction requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        text1 = prediction_data.get("text1", "")
        text2 = prediction_data.get("text2", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 are required")
        
        result = await ai_ml_engine.predict_similarity(text1, text2, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI similarity prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quality", response_model=Dict[str, Any])
async def predict_quality_ai(prediction_data: Dict[str, Any]) -> JSONResponse:
    """Predict content quality using AI/ML models"""
    logger.info("AI quality prediction requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        content = prediction_data.get("content", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_quality(content, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI quality prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/sentiment", response_model=Dict[str, Any])
async def predict_sentiment_ai(prediction_data: Dict[str, Any]) -> JSONResponse:
    """Predict sentiment using AI/ML models"""
    logger.info("AI sentiment prediction requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        content = prediction_data.get("content", "")
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_sentiment(content, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI sentiment prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/topics", response_model=Dict[str, Any])
async def predict_topics_ai(prediction_data: Dict[str, Any]) -> JSONResponse:
    """Predict topics using AI/ML models"""
    logger.info("AI topic prediction requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        content = prediction_data.get("content", "")
        num_topics = prediction_data.get("num_topics", 5)
        model_id = prediction_data.get("model_id", "default")
        
        if not content:
            raise ValueError("Content is required")
        
        result = await ai_ml_engine.predict_topics(content, num_topics, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI topic prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cluster", response_model=Dict[str, Any])
async def cluster_content_ai(clustering_data: Dict[str, Any]) -> JSONResponse:
    """Cluster content using AI/ML models"""
    logger.info("AI content clustering requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        contents = clustering_data.get("contents", [])
        num_clusters = clustering_data.get("num_clusters", 3)
        model_id = clustering_data.get("model_id", "default")
        
        if not contents or len(contents) < 2:
            raise ValueError("At least 2 contents are required for clustering")
        
        result = await ai_ml_engine.cluster_content(contents, num_clusters, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI clustering error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_ai_response(generation_data: Dict[str, Any]) -> JSONResponse:
    """Generate AI response using AI models"""
    logger.info("AI response generation requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        prompt = generation_data.get("prompt", "")
        model_id = generation_data.get("model_id", "default")
        
        if not prompt:
            raise ValueError("Prompt is required")
        
        result = await ai_ml_engine.generate_ai_response(prompt, model_id)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "prediction": result.prediction,
                "confidence": result.confidence,
                "model_name": result.model_name,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            },
            "error": None
        })
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models", response_model=Dict[str, Any])
async def get_ai_models() -> JSONResponse:
    """Get all AI/ML models"""
    logger.info("AI models list requested")
    
    if not ai_ml_engine:
        raise HTTPException(status_code=503, detail="AI/ML engine not available")
    
    try:
        ml_models = ai_ml_engine.get_models()
        ai_models = ai_ml_engine.get_ai_models()
        model_stats = ai_ml_engine.get_model_stats()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "ml_models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "type": model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type),
                        "version": model.version,
                        "accuracy": model.accuracy
                    }
                    for model in ml_models
                ],
                "ai_models": ai_models,
                "stats": model_stats
            },
            "error": None
        })
    except Exception as e:
        logger.error(f"Get AI models error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")






