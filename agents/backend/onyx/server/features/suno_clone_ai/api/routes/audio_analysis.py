"""
API de Análisis de Audio Avanzado

Endpoints para:
- Analizar audio completo
- Analizar segmento
- Comparar audios
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.audio_analysis import get_audio_analyzer
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/audio-analysis",
    tags=["audio-analysis"]
)


@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(..., description="Archivo de audio"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza un archivo de audio completo.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            analyzer = get_audio_analyzer()
            result = analyzer.analyze(tmp_path)
            
            return {
                "bpm": result.bpm,
                "key": result.key,
                "energy": result.energy,
                "tempo": result.tempo,
                "spectral_centroid": result.spectral_centroid,
                "zero_crossing_rate": result.zero_crossing_rate,
                "beats": result.beats[:20],  # Primeros 20 beats
                "mfcc": result.mfcc
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing audio: {str(e)}"
        )


@router.post("/analyze-segment")
async def analyze_segment(
    file: UploadFile = File(..., description="Archivo de audio"),
    start_time: float = Body(..., description="Tiempo de inicio (segundos)"),
    end_time: float = Body(..., description="Tiempo de fin (segundos)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza un segmento específico del audio.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            analyzer = get_audio_analyzer()
            result = analyzer.analyze_segment(tmp_path, start_time, end_time)
            
            return {
                "bpm": result.bpm,
                "key": result.key,
                "energy": result.energy,
                "tempo": result.tempo,
                "spectral_centroid": result.spectral_centroid
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error analyzing segment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing segment: {str(e)}"
        )


@router.post("/compare")
async def compare_audio(
    file1: UploadFile = File(..., description="Primer archivo de audio"),
    file2: UploadFile = File(..., description="Segundo archivo de audio"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Compara dos archivos de audio.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file1.filename)[1]) as tmp1:
            content1 = await file1.read()
            tmp1.write(content1)
            tmp1_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file2.filename)[1]) as tmp2:
            content2 = await file2.read()
            tmp2.write(content2)
            tmp2_path = tmp2.name
        
        try:
            analyzer = get_audio_analyzer()
            comparison = analyzer.compare_audio(tmp1_path, tmp2_path)
            return comparison
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
    
    except Exception as e:
        logger.error(f"Error comparing audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing audio: {str(e)}"
        )

