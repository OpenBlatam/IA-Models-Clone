"""
API de Generación de Letras

Endpoints para:
- Generar letras
- Generar letras desde audio
- Obtener letras existentes
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.lyrics_generator import get_lyrics_generator
from services.audio_transcription import get_transcription_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/lyrics",
    tags=["lyrics"]
)


@router.post("/generate")
async def generate_lyrics(
    theme: str = Body(..., description="Tema de la canción"),
    style: Optional[str] = Body(None, description="Estilo musical"),
    language: str = Body("en", description="Idioma"),
    num_verses: int = Body(3, ge=1, le=10, description="Número de versos"),
    include_chorus: bool = Body(True, description="Incluir coro"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Genera letras de canción con IA.
    """
    try:
        generator = get_lyrics_generator()
        lyrics = generator.generate_lyrics(
            theme=theme,
            style=style,
            language=language,
            num_verses=num_verses,
            include_chorus=include_chorus
        )
        
        return {
            "title": lyrics.title,
            "verses": lyrics.verses,
            "chorus": lyrics.chorus,
            "bridge": lyrics.bridge,
            "language": lyrics.language,
            "style": lyrics.style,
            "theme": lyrics.theme
        }
    except Exception as e:
        logger.error(f"Error generating lyrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating lyrics: {str(e)}"
        )


@router.post("/generate-from-audio")
async def generate_lyrics_from_audio(
    file: UploadFile = File(..., description="Archivo de audio"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Genera letras basadas en un archivo de audio (vía transcripción).
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            generator = get_lyrics_generator()
            transcription_service = get_transcription_service()
            
            lyrics = generator.generate_from_music(tmp_path, transcription_service)
            
            return {
                "title": lyrics.title,
                "verses": lyrics.verses,
                "chorus": lyrics.chorus,
                "bridge": lyrics.bridge,
                "language": lyrics.language,
                "style": lyrics.style,
                "theme": lyrics.theme
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error generating lyrics from audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating lyrics from audio: {str(e)}"
        )

