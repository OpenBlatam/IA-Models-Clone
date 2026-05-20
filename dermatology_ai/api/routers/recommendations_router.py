"""
Recommendations Router - Handles all recommendation endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json
import time

from ...api.services_locator import get_service
from ...utils.logger import logger
from ...utils.exceptions import ValidationError

router = APIRouter(prefix="/dermatology", tags=["recommendations"])


@router.post("/get-recommendations")
async def get_recommendations(
    file: UploadFile = File(..., description="Imagen de piel para análisis"),
    include_routine: bool = Form(True, description="Incluir rutina completa")
):
    """Obtiene recomendaciones de skincare basadas en análisis de piel"""
    start_time = time.time()
    
    try:
        image_processor = get_service("image_processor")
        skin_analyzer = get_service("skin_analyzer")
        skincare_recommender = get_service("skincare_recommender")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise ValidationError("El archivo debe ser una imagen")
        
        # Leer y procesar imagen
        image_bytes = await file.read()
        processed_image, is_valid, message = image_processor.process_for_analysis(image_bytes)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Analizar
        analysis_result = skin_analyzer.analyze_image(processed_image)
        
        # Generar recomendaciones
        recommendations = skincare_recommender.generate_recommendations(analysis_result)
        
        duration = time.time() - start_time
        logger.log_api_request("/get-recommendations", "POST", 200, duration)
        
        response = {
            "success": True,
            "analysis": analysis_result,
            "recommendations": recommendations if include_routine else {
                "specific_recommendations": recommendations.get("specific_recommendations", []),
                "tips": recommendations.get("tips", [])
            },
            "processing_time": round(duration, 2)
        }
        
        return JSONResponse(content=response)
    
    except ValidationError as e:
        duration = time.time() - start_time
        logger.log_api_request("/get-recommendations", "POST", 400, duration)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        duration = time.time() - start_time
        logger.log_api_request("/get-recommendations", "POST", 500, duration)
        logger.error(f"Error generando recomendaciones: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generando recomendaciones: {str(e)}")


@router.post("/recommendations/intelligent")
async def get_intelligent_recommendations(
    user_id: str = Form(...),
    analysis_data: str = Form(...),
    context: Optional[str] = Form(None)
):
    """Genera recomendaciones inteligentes basadas en ML"""
    try:
        intelligent_recommender = get_service("intelligent_recommender")
        
        analysis_dict = json.loads(analysis_data)
        context_dict = json.loads(context) if context else {}
        
        recommendations = intelligent_recommender.generate_recommendations(
            user_id, analysis_dict, context_dict
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/smart-recommendations/generate")
async def generate_smart_recommendations(
    user_id: str = Form(...),
    preferences: str = Form(...)
):
    """Genera recomendaciones inteligentes"""
    try:
        smart_recommender = get_service("smart_recommender")
        
        preferences_dict = json.loads(preferences)
        recommendations = smart_recommender.generate_smart_recommendations(
            user_id, preferences_dict
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/ml-recommendations/generate")
async def generate_ml_recommendations(
    user_id: str = Form(...),
    analysis_features: str = Form(...),
    top_k: int = Form(10)
):
    """Genera recomendaciones basadas en ML"""
    try:
        import numpy as np
        ml_recommender = get_service("ml_recommender")
        
        features_array = np.array(json.loads(analysis_features))
        recommendations = ml_recommender.generate_ml_recommendations(
            user_id, features_array, top_k
        )
        
        return JSONResponse(content={
            "success": True,
            "recommendations": [r.to_dict() for r in recommendations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




