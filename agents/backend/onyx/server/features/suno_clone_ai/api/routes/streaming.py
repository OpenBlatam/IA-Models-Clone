"""
API de Streaming de Audio

Endpoints para:
- Crear stream
- Obtener chunks de audio
- Controlar stream (pause, resume, stop, seek)
- Estadísticas de streaming
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Response
from fastapi.responses import StreamingResponse

from services.audio_streaming import get_audio_streamer, StreamConfig
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/streaming",
    tags=["streaming"]
)


@router.post("/create")
async def create_stream(
    audio_path: str = Body(..., description="Ruta del archivo de audio"),
    sample_rate: int = Body(44100, description="Sample rate"),
    channels: int = Body(2, description="Canales"),
    format: str = Body("wav", description="Formato"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea un stream de audio.
    """
    try:
        import uuid
        stream_id = str(uuid.uuid4())
        
        config = StreamConfig(
            sample_rate=sample_rate,
            channels=channels,
            format=format
        )
        
        streamer = get_audio_streamer()
        result = await streamer.create_stream(stream_id, audio_path, config)
        
        return result
    except Exception as e:
        logger.error(f"Error creating stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating stream: {str(e)}"
        )


@router.get("/stream/{stream_id}")
async def stream_audio(
    stream_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Stream de audio en tiempo real.
    """
    try:
        streamer = get_audio_streamer()
        
        async def generate():
            async for chunk in streamer.stream_chunks(stream_id):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=stream_{stream_id}.wav"
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error streaming audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error streaming audio: {str(e)}"
        )


@router.post("/{stream_id}/pause")
async def pause_stream(
    stream_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Pausa un stream.
    """
    try:
        streamer = get_audio_streamer()
        streamer.pause_stream(stream_id)
        return {"message": "Stream paused", "stream_id": stream_id}
    except Exception as e:
        logger.error(f"Error pausing stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error pausing stream: {str(e)}"
        )


@router.post("/{stream_id}/resume")
async def resume_stream(
    stream_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Reanuda un stream.
    """
    try:
        streamer = get_audio_streamer()
        streamer.resume_stream(stream_id)
        return {"message": "Stream resumed", "stream_id": stream_id}
    except Exception as e:
        logger.error(f"Error resuming stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resuming stream: {str(e)}"
        )


@router.post("/{stream_id}/stop")
async def stop_stream(
    stream_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Detiene un stream.
    """
    try:
        streamer = get_audio_streamer()
        streamer.stop_stream(stream_id)
        return {"message": "Stream stopped", "stream_id": stream_id}
    except Exception as e:
        logger.error(f"Error stopping stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping stream: {str(e)}"
        )


@router.post("/{stream_id}/seek")
async def seek_stream(
    stream_id: str,
    position: float = Body(..., description="Posición en segundos"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Busca una posición en el stream.
    """
    try:
        streamer = get_audio_streamer()
        streamer.seek_stream(stream_id, position)
        return {
            "message": "Stream seeked",
            "stream_id": stream_id,
            "position": position
        }
    except Exception as e:
        logger.error(f"Error seeking stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error seeking stream: {str(e)}"
        )


@router.get("/{stream_id}/stats")
async def get_stream_stats(
    stream_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un stream.
    """
    try:
        streamer = get_audio_streamer()
        stats = streamer.get_stream_stats(stream_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stream stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

