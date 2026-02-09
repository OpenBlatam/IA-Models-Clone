"""
API para procesamiento por lotes
"""

import logging
import uuid
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from core.music_generator import get_music_generator
    from core.chat_processor import get_chat_processor
    from services.song_service import SongService
    from services.notification_service import get_notification_service
    from config.settings import settings
except ImportError:
    from .core.music_generator import get_music_generator
    from .core.chat_processor import get_chat_processor
    from .services.song_service import SongService
    from .services.notification_service import get_notification_service
    from .config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()
song_service = SongService()


class BatchGenerationRequest(BaseModel):
    """Request para generación por lotes"""
    prompts: List[str] = Field(..., description="Lista de prompts a generar", min_items=1, max_items=50)
    user_id: Optional[str] = Field(None, description="ID del usuario")
    duration: Optional[int] = Field(None, description="Duración en segundos para todas las canciones")
    notify_on_completion: Optional[bool] = Field(True, description="Enviar notificaciones al completar")


class BatchGenerationResponse(BaseModel):
    """Response de generación por lotes"""
    batch_id: str
    total_songs: int
    song_ids: List[str]
    status: str
    message: str


@router.post("/batch/generate", response_model=BatchGenerationResponse)
async def batch_generate(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Genera múltiples canciones en lote"""
    try:
        batch_id = str(uuid.uuid4())
        song_ids = []
        
        # Procesar cada prompt
        chat_processor = get_chat_processor()
        
        for prompt in request.prompts:
            song_id = str(uuid.uuid4())
            song_ids.append(song_id)
            
            # Extraer información de la canción
            song_info = chat_processor.extract_song_info(prompt)
            if request.duration:
                song_info["duration"] = request.duration
            
            # Agregar tarea de background
            background_tasks.add_task(
                _generate_song_batch,
                song_id,
                song_info,
                request.user_id,
                batch_id,
                request.notify_on_completion
            )
        
        return BatchGenerationResponse(
            batch_id=batch_id,
            total_songs=len(song_ids),
            song_ids=song_ids,
            status="processing",
            message=f"Batch generation started for {len(song_ids)} songs"
        )
        
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_song_batch(
    song_id: str,
    song_info: Dict,
    user_id: Optional[str],
    batch_id: str,
    notify: bool
):
    """Genera una canción como parte de un lote"""
    import time
    from api.song_api import _generate_song_background
    
    start_time = time.time()
    
    try:
        # Notificar inicio si está habilitado
        if notify and user_id:
            notification_service = get_notification_service()
            await notification_service.notify_generation_started(user_id, song_id)
        
        # Generar canción (reutilizar función existente)
        await _generate_song_background(song_id, song_info, user_id)
        
        # Notificar completado
        if notify and user_id:
            song = song_service.get_song(song_id)
            if song:
                audio_url = f"/suno/songs/{song_id}/download"
                await notification_service.notify_song_completed(user_id, song_id, audio_url)
        
        logger.info(f"Batch song {song_id} completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error generating batch song {song_id}: {e}")
        if notify and user_id:
            notification_service = get_notification_service()
            await notification_service.notify_song_failed(user_id, song_id, str(e))


@router.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Obtiene el estado de un lote de generación"""
    # Nota: Esto requeriría tracking adicional en la base de datos
    # Por ahora, retornamos un placeholder
    return {
        "batch_id": batch_id,
        "status": "processing",
        "message": "Batch status tracking not fully implemented yet"
    }

