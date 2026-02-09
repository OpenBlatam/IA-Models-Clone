"""
API de Transcripción de Audio

Endpoints para:
- Transcribir audio
- Detectar idioma
- Resumir transcripción
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File

from services.audio_transcription import get_transcription_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/transcription",
    tags=["transcription"]
)


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Archivo de audio"),
    language: Optional[str] = Query(None, description="Idioma (opcional, auto-detect si None)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Transcribe un archivo de audio a texto.
    """
    try:
        # Guardar archivo temporalmente
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            transcription_service = get_transcription_service()
            result = transcription_service.transcribe(tmp_path, language=language)
            
            return {
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "confidence": result.confidence,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": seg.confidence
                    }
                    for seg in result.segments
                ]
            }
        finally:
            # Limpiar archivo temporal
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}"
        )


@router.post("/detect-language")
async def detect_language(
    file: UploadFile = File(..., description="Archivo de audio"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Detecta el idioma de un archivo de audio.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            transcription_service = get_transcription_service()
            language = transcription_service.detect_language(tmp_path)
            
            return {
                "language": language,
                "file": file.filename
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error detecting language: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting language: {str(e)}"
        )


@router.post("/summarize")
async def summarize_transcription(
    transcription: Dict[str, Any] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Genera un resumen de una transcripción.
    """
    try:
        from services.audio_transcription import TranscriptionResult, TranscriptionSegment
        
        # Convertir dict a TranscriptionResult
        result = TranscriptionResult(
            text=transcription.get("text", ""),
            language=transcription.get("language", "en"),
            segments=[
                TranscriptionSegment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", ""),
                    confidence=seg.get("confidence", 0)
                )
                for seg in transcription.get("segments", [])
            ],
            duration=transcription.get("duration", 0)
        )
        
        transcription_service = get_transcription_service()
        summary = transcription_service.summarize_transcription(result)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error summarizing transcription: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error summarizing transcription: {str(e)}"
        )

