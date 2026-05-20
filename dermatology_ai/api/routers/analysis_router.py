"""
Analysis Router - Handles image, video, texture, and advanced analysis endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import time
import hashlib
import json
import numpy as np
import cv2
from PIL import Image
import io

from ...api.services_locator import get_service
from ...utils.logger import logger
from ...utils.exceptions import (
    ImageProcessingError, VideoProcessingError,
    AnalysisError, ValidationError
)

router = APIRouter(prefix="/dermatology", tags=["analysis"])


@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(..., description="Imagen de piel para analizar"),
    enhance: bool = Form(True, description="Mejorar imagen antes de análisis"),
    use_advanced: bool = Form(True, description="Usar análisis avanzado"),
    use_cache: bool = Form(True, description="Usar cache para mejorar rendimiento")
):
    """Analiza una imagen de piel y proporciona métricas de calidad"""
    start_time = time.time()
    
    try:
        logger.info(f"Recibida solicitud de análisis de imagen: {file.filename}")
        
        # Get services from locator
        skin_analyzer = get_service("skin_analyzer")
        image_processor = get_service("image_processor")
        history_tracker = get_service("history_tracker")
        db_manager = get_service("db_manager")
        alert_system = get_service("alert_system")
        advanced_validator = get_service("advanced_validator")
        webhook_manager = get_service("webhook_manager")
        notification_service = get_service("notification_service")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise ValidationError("El archivo debe ser una imagen (JPG, PNG, etc.)")
        
        # Leer imagen
        image_bytes = await file.read()
        logger.debug(f"Imagen leída: {len(image_bytes)} bytes")
        
        # Validación avanzada
        is_valid_advanced, validation_info = advanced_validator.validate_image_comprehensive(
            image_bytes, filename=file.filename
        )
        
        if not is_valid_advanced:
            errors = validation_info.get("errors", [])
            raise ValidationError(f"Validación fallida: {'; '.join(errors)}")
        
        # Procesar imagen
        processed_image, is_valid, message = image_processor.process_for_analysis(image_bytes)
        
        if not is_valid:
            raise ImageProcessingError(message)
        
        # Mejorar si se solicita
        if enhance:
            logger.debug("Mejorando imagen")
            processed_image = image_processor.enhance_for_analysis(processed_image)
        
        # Configurar analizador
        if use_advanced != skin_analyzer.use_advanced:
            skin_analyzer.use_advanced = use_advanced
        
        # Analizar
        logger.debug("Iniciando análisis de piel")
        analysis_result = skin_analyzer.analyze_image(processed_image, use_cache=use_cache)
        analysis_result["analysis_type"] = "image"
        
        # Generar hash de imagen para tracking
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Guardar en historial
        try:
            record_id = history_tracker.save_analysis(
                analysis_result,
                image_hash=image_hash,
                metadata={"enhanced": enhance, "advanced": use_advanced}
            )
            analysis_result["record_id"] = record_id
            
            db_manager.save_analysis(
                record_id,
                user_id=None,
                analysis_result=analysis_result,
                image_hash=image_hash,
                metadata={"enhanced": enhance, "advanced": use_advanced}
            )
            
            # Verificar alertas
            alerts = alert_system.check_analysis_alerts(analysis_result)
            if alerts:
                analysis_result["alerts"] = [
                    {
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message
                    }
                    for alert in alerts
                ]
        except Exception as e:
            logger.warning(f"Error guardando en historial: {str(e)}")
        
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 200, duration)
        
        return JSONResponse(content={
            "success": True,
            "analysis": analysis_result,
            "message": message,
            "processing_time": round(duration, 2),
            "settings": {
                "enhanced": enhance,
                "advanced_analysis": use_advanced,
                "cached": use_cache
            }
        })
    
    except ValidationError as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 400, duration)
        raise HTTPException(status_code=400, detail=str(e))
    
    except (ImageProcessingError, AnalysisError) as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 400, duration)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-image", "POST", 500, duration)
        logger.error(f"Error inesperado analizando imagen: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analizando imagen: {str(e)}")


@router.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(..., description="Video de piel para analizar"),
    max_frames: int = Form(30, description="Máximo número de frames a procesar")
):
    """Analiza un video de piel y proporciona análisis agregado"""
    start_time = time.time()
    
    try:
        logger.info(f"Recibida solicitud de análisis de video: {file.filename}")
        
        video_processor = get_service("video_processor")
        skin_analyzer = get_service("skin_analyzer")
        
        # Validar tipo de archivo
        if not file.content_type or not file.content_type.startswith('video/'):
            raise ValidationError("El archivo debe ser un video (MP4, AVI, etc.)")
        
        # Leer video
        video_bytes = await file.read()
        logger.debug(f"Video leído: {len(video_bytes)} bytes")
        
        # Validar video
        is_valid, message = video_processor.validate_video(video_bytes)
        if not is_valid:
            raise VideoProcessingError(message)
        
        # Extraer frames
        video_processor.max_frames = max_frames
        logger.debug(f"Extrayendo hasta {max_frames} frames")
        frames = video_processor.extract_frames(video_bytes)
        
        if not frames:
            raise VideoProcessingError("No se pudieron extraer frames del video")
        
        logger.debug(f"Frames extraídos: {len(frames)}")
        
        # Analizar frames
        analysis_result = skin_analyzer.analyze_video(frames)
        
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 200, duration)
        
        return JSONResponse(content={
            "success": True,
            "analysis": analysis_result,
            "frames_analyzed": len(frames),
            "processing_time": round(duration, 2)
        })
    
    except (ValidationError, VideoProcessingError) as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 400, duration)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        duration = time.time() - start_time
        logger.log_api_request("/analyze-video", "POST", 500, duration)
        logger.error(f"Error inesperado analizando video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analizando video: {str(e)}")


@router.post("/texture-ml/analyze")
async def analyze_texture_ml(
    user_id: str = Form(...),
    image_id: str = Form(...),
    image_file: UploadFile = File(...)
):
    """Analiza textura usando ML avanzado"""
    try:
        advanced_texture_ml = get_service("advanced_texture_ml")
        
        # Leer imagen
        image_bytes = await image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Convertir a escala de grises si es necesario
        if len(image_array.shape) == 3:
            image_array = np.mean(image_array, axis=2)
        
        analysis = advanced_texture_ml.analyze_texture(user_id, image_id, image_array)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/image-analysis/advanced")
async def advanced_image_analysis(
    file: UploadFile = File(...),
    analysis_type: str = Form("all")
):
    """Análisis avanzado de imagen"""
    try:
        image_analysis_advanced = get_service("image_analysis_advanced")
        
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = {}
        
        if analysis_type in ["all", "texture"]:
            results["texture"] = image_analysis_advanced.analyze_texture_features(image)
        
        if analysis_type in ["all", "color"]:
            results["color"] = image_analysis_advanced.analyze_color_features(image)
        
        if analysis_type in ["all", "geometric"]:
            results["geometric"] = image_analysis_advanced.analyze_geometric_features(image)
        
        return JSONResponse(content={"success": True, "analysis": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

