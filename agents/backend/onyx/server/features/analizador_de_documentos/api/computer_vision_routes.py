"""
Rutas para Computer Vision
===========================

Endpoints para computer vision avanzado.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.computer_vision import get_computer_vision, AdvancedComputerVision

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/computer-vision",
    tags=["Computer Vision"]
)


@router.post("/detect-objects")
async def detect_objects(
    image_path: str = Field(..., description="Ruta de la imagen"),
    confidence_threshold: float = Field(0.5, description="Umbral de confianza"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Detectar objetos en imagen"""
    try:
        objects = cv.detect_objects(image_path, confidence_threshold)
        
        return {
            "image_path": image_path,
            "objects": [
                {
                    "object_id": obj.object_id,
                    "class_name": obj.class_name,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox,
                    "attributes": obj.attributes
                }
                for obj in objects
            ],
            "count": len(objects)
        }
    except Exception as e:
        logger.error(f"Error detectando objetos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-faces")
async def recognize_faces(
    image_path: str = Field(..., description="Ruta de la imagen"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Reconocer caras en imagen"""
    try:
        faces = cv.recognize_faces(image_path)
        
        return {
            "image_path": image_path,
            "faces": faces,
            "count": len(faces)
        }
    except Exception as e:
        logger.error(f"Error reconociendo caras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-text")
async def extract_text_from_image(
    image_path: str = Field(..., description="Ruta de la imagen"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Extraer texto de imagen"""
    try:
        result = cv.extract_text_from_image(image_path)
        
        return result
    except Exception as e:
        logger.error(f"Error extrayendo texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-scene")
async def analyze_scene(
    image_path: str = Field(..., description="Ruta de la imagen"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Analizar escena"""
    try:
        analysis = cv.analyze_scene(image_path)
        
        return analysis
    except Exception as e:
        logger.error(f"Error analizando escena: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-image")
async def segment_image(
    image_path: str = Field(..., description="Ruta de la imagen"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Segmentar imagen"""
    try:
        segmentation = cv.segment_image(image_path)
        
        return segmentation
    except Exception as e:
        logger.error(f"Error segmentando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-quality")
async def analyze_image_quality(
    image_path: str = Field(..., description="Ruta de la imagen"),
    cv: AdvancedComputerVision = Depends(get_computer_vision)
):
    """Analizar calidad de imagen"""
    try:
        quality = cv.analyze_image_quality(image_path)
        
        return quality
    except Exception as e:
        logger.error(f"Error analizando calidad: {e}")
        raise HTTPException(status_code=500, detail=str(e))



