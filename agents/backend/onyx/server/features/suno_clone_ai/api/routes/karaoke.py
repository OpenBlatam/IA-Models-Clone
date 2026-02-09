"""
API de Karaoke

Endpoints para:
- Crear pista de karaoke
- Evaluar rendimiento
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.karaoke import get_karaoke_service
from services.lyrics_sync import get_lyrics_synchronizer
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/karaoke",
    tags=["karaoke"]
)


@router.post("/create-track")
async def create_karaoke_track(
    file: UploadFile = File(..., description="Archivo de audio"),
    method: str = Body("center", description="Método de eliminación de voces"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea una pista de karaoke eliminando voces.
    """
    try:
        import tempfile
        import os
        import uuid
        
        # Guardar archivo de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name
        
        # Crear archivo de salida
        output_filename = f"karaoke_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join("/tmp", output_filename)
        
        try:
            karaoke_service = get_karaoke_service()
            result = karaoke_service.create_karaoke_track(input_path, output_path, method)
            
            # Leer archivo de salida
            with open(output_path, "rb") as f:
                output_content = f.read()
            
            # Limpiar
            os.unlink(input_path)
            os.unlink(output_path)
            
            from fastapi.responses import Response
            return Response(
                content=output_content,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename={output_filename}"
                }
            )
        
        except Exception as e:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise
    
    except Exception as e:
        logger.error(f"Error creating karaoke track: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating karaoke track: {str(e)}"
        )


@router.post("/score")
async def score_performance(
    original_file: UploadFile = File(..., description="Audio original de referencia"),
    user_file: UploadFile = File(..., description="Audio del usuario"),
    lyrics: Optional[str] = Body(None, description="Letras (opcional)"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Evalúa el rendimiento del usuario en karaoke.
    """
    try:
        import tempfile
        import os
        
        # Guardar archivos
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_file.filename)[1]) as tmp1:
            content1 = await original_file.read()
            tmp1.write(content1)
            original_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_file.filename)[1]) as tmp2:
            content2 = await user_file.read()
            tmp2.write(content2)
            user_path = tmp2.name
        
        try:
            karaoke_service = get_karaoke_service()
            
            # Sincronizar letras si se proporcionan
            synced_lyrics = None
            if lyrics:
                synchronizer = get_lyrics_synchronizer()
                synced_lyrics = synchronizer.sync_lyrics(original_path, lyrics)
            
            score = karaoke_service.score_performance(
                original_path,
                user_path,
                synced_lyrics
            )
            
            return {
                "accuracy": score.accuracy,
                "timing_score": score.timing_score,
                "pitch_score": score.pitch_score,
                "total_score": score.total_score
            }
        
        finally:
            if os.path.exists(original_path):
                os.unlink(original_path)
            if os.path.exists(user_path):
                os.unlink(user_path)
    
    except Exception as e:
        logger.error(f"Error scoring performance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scoring performance: {str(e)}"
        )


@router.post("/sync-lyrics")
async def sync_lyrics(
    file: UploadFile = File(..., description="Archivo de audio"),
    lyrics: str = Body(..., description="Texto de las letras"),
    method: str = Body("energy", description="Método de sincronización"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Sincroniza letras con audio.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            synchronizer = get_lyrics_synchronizer()
            result = synchronizer.sync_lyrics(tmp_path, lyrics, method)
            
            return {
                "words": [
                    {
                        "word": w.word,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "confidence": w.confidence
                    }
                    for w in result.words
                ],
                "total_duration": result.total_duration
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error syncing lyrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error syncing lyrics: {str(e)}"
        )

