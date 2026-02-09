"""
API de DJ Automático

Endpoints para:
- Crear mixes
- Analizar pistas
- Generar playlists
- Obtener recomendaciones
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, UploadFile, File

from services.auto_dj import get_auto_dj_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auto-dj",
    tags=["auto-dj"]
)


@router.post("/analyze")
async def analyze_track(
    file: UploadFile = File(..., description="Archivo de audio"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza una pista para DJ.
    """
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            dj_service = get_auto_dj_service()
            track_info = dj_service.analyze_track(tmp_path)
            
            return {
                "track_id": track_info.track_id,
                "bpm": track_info.bpm,
                "key": track_info.key,
                "energy": track_info.energy,
                "duration": track_info.duration
            }
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"Error analyzing track: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing track: {str(e)}"
        )


@router.post("/create-mix")
async def create_mix(
    track_paths: List[str] = Body(..., description="Lista de rutas de pistas"),
    transition_type: str = Body("crossfade", description="Tipo de transición"),
    transition_duration: float = Body(2.0, description="Duración de transición"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Crea un mix automático.
    """
    try:
        dj_service = get_auto_dj_service()
        dj_set = dj_service.create_mix(track_paths, transition_type, transition_duration)
        
        return {
            "set_id": dj_set.set_id,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "bpm": t.bpm,
                    "key": t.key,
                    "duration": t.duration
                }
                for t in dj_set.tracks
            ],
            "transitions": [
                {
                    "from_track": t.from_track,
                    "to_track": t.to_track,
                    "type": t.transition_type,
                    "duration": t.duration,
                    "start_time": t.start_time
                }
                for t in dj_set.transitions
            ],
            "total_duration": dj_set.total_duration
        }
    except Exception as e:
        logger.error(f"Error creating mix: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating mix: {str(e)}"
        )


@router.post("/recommendations")
async def get_mix_recommendations(
    current_track_path: str = Body(..., description="Ruta de pista actual"),
    available_track_paths: List[str] = Body(..., description="Pistas disponibles"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Obtiene recomendaciones de pistas para mix.
    """
    try:
        dj_service = get_auto_dj_service()
        
        current_track = dj_service.analyze_track(current_track_path)
        available_tracks = [dj_service.analyze_track(path) for path in available_track_paths]
        
        recommendations = dj_service.get_mix_recommendations(current_track, available_tracks)
        
        return {
            "recommendations": [
                {
                    "track_id": t.track_id,
                    "bpm": t.bpm,
                    "key": t.key,
                    "energy": t.energy
                }
                for t in recommendations
            ]
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting recommendations: {str(e)}"
        )

