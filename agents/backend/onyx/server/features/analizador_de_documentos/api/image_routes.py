"""
Rutas para Análisis de Imágenes
================================

Endpoints para análisis de imágenes en documentos.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pathlib import Path
import tempfile
import os

from ..core.image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/images",
    tags=["Image Analysis"]
)


# Instancia global
_image_analyzer: Optional[ImageAnalyzer] = None


def get_image_analyzer() -> ImageAnalyzer:
    """Dependency para obtener analizador de imágenes"""
    global _image_analyzer
    if _image_analyzer is None:
        _image_analyzer = ImageAnalyzer()
    return _image_analyzer


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    extract_text: bool = Form(True),
    detect_objects: bool = Form(True),
    analyzer: ImageAnalyzer = Depends(get_image_analyzer)
):
    """Analizar imagen"""
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Analizar
            result = await analyzer.analyze_image(
                tmp_path,
                extract_text=extract_text,
                detect_objects=detect_objects
            )
            
            return {
                "image_id": result.image_id,
                "width": result.width,
                "height": result.height,
                "format": result.format,
                "size_bytes": result.size_bytes,
                "objects": result.objects,
                "text": result.text,
                "labels": result.labels,
                "colors": result.colors,
                "confidence": result.confidence,
                "timestamp": result.timestamp
            }
        finally:
            # Limpiar
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error analizando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-from-pdf")
async def extract_images_from_pdf(
    file: UploadFile = File(...),
    analyzer: ImageAnalyzer = Depends(get_image_analyzer)
):
    """Extraer y analizar imágenes de PDF"""
    try:
        # Guardar PDF temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Extraer y analizar imágenes
            images = await analyzer.extract_images_from_pdf(tmp_path)
            
            return {
                "total_images": len(images),
                "images": images
            }
        finally:
            # Limpiar
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error extrayendo imágenes de PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















