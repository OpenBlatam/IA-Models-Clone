"""
Rutas para OCR
==============

Endpoints para procesamiento OCR de imágenes y PDFs.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pathlib import Path
import tempfile
import os

from ..core.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/ocr",
    tags=["OCR Processing"]
)


# Instancia global del procesador OCR
_ocr_processor: Optional[OCRProcessor] = None


def get_ocr_processor() -> OCRProcessor:
    """Dependency para obtener procesador OCR"""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor(engine="auto")
    return _ocr_processor


@router.post("/image")
async def process_image_ocr(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    processor: OCRProcessor = Depends(get_ocr_processor)
):
    """Procesar imagen con OCR"""
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Procesar con OCR
            result = await processor.process_image(tmp_path, language)
            
            return {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "pages": result.pages,
                "metadata": result.metadata,
                "timestamp": result.timestamp
            }
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error procesando imagen con OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf")
async def process_pdf_ocr(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    processor: OCRProcessor = Depends(get_ocr_processor)
):
    """Procesar PDF escaneado con OCR"""
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Procesar con OCR
            result = await processor.process_pdf(tmp_path, language)
            
            return {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "pages": result.pages,
                "metadata": result.metadata,
                "timestamp": result.timestamp
            }
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error procesando PDF con OCR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















