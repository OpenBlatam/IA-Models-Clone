"""
API endpoints para análisis musical

⚠️ DEPRECATED: This file is deprecated and kept only for backward compatibility.
Please use the modular router from `music_api_refactored.py` instead.

This monolithic file contains 482+ endpoints and should not be used for new development.
All new endpoints should be added to the appropriate router in the `routes/` directory.
"""

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging

# Use factories instead of direct imports
from .factories import (
    get_spotify_service,
    get_music_analyzer,
    get_music_coach,
    get_comparison_service,
    get_export_service,
    get_history_service,
    get_analytics_service,
    get_webhook_service,
    get_favorites_service,
    get_tagging_service,
    get_playlist_service,
    get_intelligent_recommender,
    get_dashboard_service,
    get_notification_service,
    get_service
)
from ..services.webhook_service import WebhookEvent
from ..utils.exceptions import (
    SpotifyAPIException,
    TrackNotFoundException,
    InvalidTrackIDException,
    AnalysisException
)
from ..utils.cache import cache_manager
import time
from ..models.schemas import (
    TrackSearchRequest,
    TrackAnalysisRequest,
    MusicAnalysisResponse,
    CoachingAnalysisResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/music", tags=["Music Analysis"])

# Services are now retrieved via DI factories (lazy loading)
# This eliminates direct instantiation and improves testability

# Optional services are retrieved via get_service() when needed
# This avoids importing and instantiating services that may not be available


@router.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "Music Analyzer AI",
        "version": "2.18.0",
        "status": "running",
        "description": "Sistema de análisis musical con integración a Spotify"
    }


@router.get("/health")
async def health_check():
    """Verificación de salud del servicio"""
    try:
        # Get service from DI
        spotify_service = get_spotify_service()
        # Verificar conexión con Spotify
        spotify_service._get_access_token()
        return {
            "status": "healthy",
            "spotify_connection": "ok"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "spotify_connection": "error",
            "error": str(e)
        }


@router.post("/search", response_model=dict)
async def search_track(request: TrackSearchRequest):
    """
    Busca canciones en Spotify
    
    - **query**: Nombre de la canción o artista
    - **limit**: Número máximo de resultados (1-50)
    """
    try:
        spotify_service = get_spotify_service()
        tracks = spotify_service.search_track(request.query, request.limit)
        
        # Formatear resultados
        results = []
        for track in tracks:
            artists = [artist["name"] for artist in track.get("artists", [])]
            results.append({
                "id": track.get("id"),
                "name": track.get("name"),
                "artists": artists,
                "album": track.get("album", {}).get("name"),
                "duration_ms": track.get("duration_ms"),
                "preview_url": track.get("preview_url"),
                "external_urls": track.get("external_urls", {}),
                "popularity": track.get("popularity", 0)
            })
        
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "total": len(results)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching track: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar canción: {str(e)}")


@router.post("/analyze", response_model=dict)
async def analyze_track(request: TrackAnalysisRequest):
    """
    Analiza una canción completa
    
    - **track_id**: ID de la canción en Spotify (opcional si se proporciona track_name)
    - **track_name**: Nombre de la canción para buscar (opcional si se proporciona track_id)
    - **include_coaching**: Incluir análisis de coaching
    """
    try:
        # Obtener track_id
        track_id = request.track_id
        
        if not track_id and request.track_name:
            # Buscar la canción
            tracks = spotify_service.search_track(request.track_name, limit=1)
            if not tracks:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró la canción: {request.track_name}"
                )
            track_id = tracks[0]["id"]
        elif not track_id:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar track_id o track_name"
            )
        
        # Obtener datos completos de Spotify
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        
        # Analizar la música
        analysis = music_analyzer.analyze_track(spotify_data)
        
        response = {
            "success": True,
            "track_basic_info": analysis["track_basic_info"],
            "musical_analysis": analysis["musical_analysis"],
            "technical_analysis": analysis["technical_analysis"],
            "composition_analysis": analysis["composition_analysis"],
            "performance_analysis": analysis["performance_analysis"],
            "educational_insights": analysis["educational_insights"]
        }
        
        # Agregar coaching si se solicita
        if request.include_coaching:
            music_coach = get_music_coach()
            if music_coach:
                coaching = music_coach.generate_coaching_analysis(analysis)
                response["coaching"] = coaching
        
        # Guardar en historial
        try:
            user_id = None  # Se puede obtener del request si hay autenticación
            history_service = get_history_service()
            analytics_service = get_analytics_service()
            if history_service:
                history_service.add_analysis(
                    track_id=track_id,
                    track_name=analysis["track_basic_info"]["name"],
                    artists=analysis["track_basic_info"]["artists"],
                    analysis=response,
                    user_id=user_id
                )
            if analytics_service:
                analytics_service.track_analysis(track_id, user_id)
        except Exception as e:
            logger.warning(f"Error saving to history: {e}")
        
        # Disparar webhook de análisis completado
        try:
            webhook_service = get_webhook_service()
            if webhook_service:
                import asyncio
                asyncio.create_task(webhook_service.trigger_webhook(
                    WebhookEvent.ANALYSIS_COMPLETED,
                    {
                        "track_id": track_id,
                        "track_name": analysis["track_basic_info"]["name"],
                        "analysis_id": track_id
                    }
                ))
        except Exception as e:
            logger.warning(f"Error triggering webhook: {e}")
        
        return response
        
    except HTTPException:
        raise
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing track: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar canción: {str(e)}")


@router.get("/analyze/{track_id}", response_model=dict)
async def analyze_track_by_id(
    track_id: str,
    include_coaching: bool = Query(True, description="Incluir análisis de coaching")
):
    """
    Analiza una canción por su ID de Spotify
    
    - **track_id**: ID de la canción en Spotify
    - **include_coaching**: Incluir análisis de coaching
    """
    try:
        # Obtener datos completos de Spotify
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        
        # Analizar la música
        analysis = music_analyzer.analyze_track(spotify_data)
        
        response = {
            "success": True,
            "track_basic_info": analysis["track_basic_info"],
            "musical_analysis": analysis["musical_analysis"],
            "technical_analysis": analysis["technical_analysis"],
            "composition_analysis": analysis["composition_analysis"],
            "performance_analysis": analysis["performance_analysis"],
            "educational_insights": analysis["educational_insights"]
        }
        
        # Agregar coaching si se solicita
        if include_coaching:
            coaching = music_coach.generate_coaching_analysis(analysis)
            response["coaching"] = coaching
        
        return response
        
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing track {track_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar canción: {str(e)}")


@router.post("/coaching", response_model=dict)
async def get_coaching_analysis(request: TrackAnalysisRequest):
    """
    Obtiene análisis de coaching para una canción
    
    - **track_id**: ID de la canción en Spotify
    - **track_name**: Nombre de la canción para buscar
    """
    try:
        # Obtener track_id
        track_id = request.track_id
        
        if not track_id and request.track_name:
            # Buscar la canción
            tracks = spotify_service.search_track(request.track_name, limit=1)
            if not tracks:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró la canción: {request.track_name}"
                )
            track_id = tracks[0]["id"]
        elif not track_id:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar track_id o track_name"
            )
        
        # Obtener datos completos de Spotify
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        
        # Analizar la música
        analysis = music_analyzer.analyze_track(spotify_data)
        
        # Generar coaching
        coaching = music_coach.generate_coaching_analysis(analysis)
        
        return {
            "success": True,
            "track_basic_info": analysis["track_basic_info"],
            "coaching": coaching
        }
        
    except HTTPException:
        raise
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting coaching analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener coaching: {str(e)}")


@router.get("/track/{track_id}/info", response_model=dict)
async def get_track_info(track_id: str):
    """Obtiene información básica de una canción"""
    try:
        track_info = spotify_service.get_track(track_id)
        return {
            "success": True,
            "track": track_info
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting track info: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener información: {str(e)}")


@router.get("/track/{track_id}/audio-features", response_model=dict)
async def get_audio_features(track_id: str):
    """Obtiene las características de audio de una canción"""
    try:
        features = spotify_service.get_track_audio_features(track_id)
        return {
            "success": True,
            "audio_features": features
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting audio features: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener características: {str(e)}")


@router.get("/track/{track_id}/audio-analysis", response_model=dict)
async def get_audio_analysis(track_id: str):
    """Obtiene el análisis de audio detallado de una canción"""
    try:
        analysis = spotify_service.get_track_audio_analysis(track_id)
        return {
            "success": True,
            "audio_analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting audio analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener análisis: {str(e)}")


@router.post("/compare", response_model=dict)
async def compare_tracks(track_ids: List[str]):
    """
    Compara múltiples canciones
    
    - **track_ids**: Lista de IDs de canciones de Spotify (mínimo 2, máximo 10)
    """
    try:
        if len(track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 canciones para comparar")
        
        if len(track_ids) > 10:
            raise HTTPException(status_code=400, detail="No se pueden comparar más de 10 canciones a la vez")
        
        comparison = comparison_service.compare_tracks(track_ids)
        
        return {
            "success": True,
            **comparison
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error comparing tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar canciones: {str(e)}")


@router.get("/track/{track_id}/recommendations", response_model=dict)
async def get_recommendations(
    track_id: str,
    limit: int = Query(20, ge=1, le=100, description="Número de recomendaciones")
):
    """
    Obtiene recomendaciones de canciones similares
    
    - **track_id**: ID de la canción en Spotify
    - **limit**: Número de recomendaciones (1-100)
    """
    try:
        recommendations = spotify_service.get_recommendations(track_id, limit)
        
        # Formatear resultados
        results = []
        for track in recommendations:
            artists = [artist["name"] for artist in track.get("artists", [])]
            results.append({
                "id": track.get("id"),
                "name": track.get("name"),
                "artists": artists,
                "album": track.get("album", {}).get("name"),
                "duration_ms": track.get("duration_ms"),
                "popularity": track.get("popularity", 0),
                "preview_url": track.get("preview_url"),
                "external_urls": track.get("external_urls", {})
            })
        
        return {
            "success": True,
            "seed_track_id": track_id,
            "recommendations": results,
            "total": len(results)
        }
    except InvalidTrackIDException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SpotifyAPIException as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener recomendaciones: {str(e)}")


@router.get("/cache/stats", response_model=dict)
async def get_cache_stats():
    """Obtiene estadísticas del cache"""
    try:
        stats = cache_manager.get_stats()
        return {
            "success": True,
            "cache": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.delete("/cache/clear", response_model=dict)
async def clear_cache(prefix: Optional[str] = None):
    """Limpia el cache"""
    try:
        cache_manager.clear(prefix)
        return {
            "success": True,
            "message": f"Cache cleared: {prefix or 'all'}"
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error al limpiar cache: {str(e)}")


@router.post("/export/{track_id}", response_model=dict)
async def export_analysis(
    track_id: str,
    format: str = Query("json", regex="^(json|text|markdown)$"),
    include_coaching: bool = Query(True)
):
    """
    Exporta un análisis a diferentes formatos
    
    - **track_id**: ID de la canción
    - **format**: Formato de exportación (json, text, markdown)
    - **include_coaching**: Incluir análisis de coaching
    """
    try:
        start_time = time.time()
        
        # Obtener análisis
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        if include_coaching:
            coaching = music_coach.generate_coaching_analysis(analysis)
            analysis["coaching"] = coaching
        
        # Exportar según formato
        if format == "json":
            content = export_service.export_to_json(analysis, include_coaching)
            content_type = "application/json"
        elif format == "text":
            content = export_service.export_to_text(analysis, include_coaching)
            content_type = "text/plain"
        else:  # markdown
            content = export_service.export_to_markdown(analysis, include_coaching)
            content_type = "text/markdown"
        
        response_time = time.time() - start_time
        analytics_service.track_request(f"/export/{track_id}", "POST", response_time=response_time)
        
        return {
            "success": True,
            "format": format,
            "content": content,
            "content_type": content_type,
            "track_id": track_id
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar análisis: {str(e)}")


@router.get("/history", response_model=dict)
async def get_history(
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Obtiene el historial de análisis"""
    try:
        history = history_service.get_history(user_id, limit)
        return {
            "success": True,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.get("/history/stats", response_model=dict)
async def get_history_stats(user_id: Optional[str] = Query(None)):
    """Obtiene estadísticas del historial"""
    try:
        stats = history_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting history stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.delete("/history/{analysis_id}", response_model=dict)
async def delete_history_entry(
    analysis_id: str,
    user_id: Optional[str] = Query(None)
):
    """Elimina una entrada del historial"""
    try:
        success = history_service.delete_analysis(analysis_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        return {
            "success": True,
            "message": "Análisis eliminado del historial"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history entry: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar entrada: {str(e)}")


@router.get("/analytics", response_model=dict)
async def get_analytics():
    """Obtiene estadísticas y métricas del sistema"""
    try:
        stats = analytics_service.get_stats()
        return {
            "success": True,
            "analytics": stats
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener analytics: {str(e)}")


@router.post("/analytics/reset", response_model=dict)
async def reset_analytics():
    """Resetea las estadísticas de analytics"""
    try:
        analytics_service.reset_stats()
        return {
            "success": True,
            "message": "Analytics reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al resetear analytics: {str(e)}")


@router.post("/favorites", response_model=dict)
async def add_favorite(
    track_id: str,
    track_name: str,
    artists: List[str],
    user_id: str = Query(..., description="ID del usuario"),
    notes: Optional[str] = Query(None, description="Notas opcionales")
):
    """Agrega una canción a favoritos"""
    try:
        success = favorites_service.add_favorite(user_id, track_id, track_name, artists, notes)
        if not success:
            raise HTTPException(status_code=400, detail="La canción ya está en favoritos")
        return {
            "success": True,
            "message": "Canción agregada a favoritos"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar favorito: {str(e)}")


@router.delete("/favorites/{track_id}", response_model=dict)
async def remove_favorite(track_id: str, user_id: str = Query(...)):
    """Elimina una canción de favoritos"""
    try:
        success = favorites_service.remove_favorite(user_id, track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Canción no encontrada en favoritos")
        return {
            "success": True,
            "message": "Canción eliminada de favoritos"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar favorito: {str(e)}")


@router.get("/favorites", response_model=dict)
async def get_favorites(user_id: str = Query(...)):
    """Obtiene los favoritos de un usuario"""
    try:
        favorites = favorites_service.get_favorites(user_id)
        return {
            "success": True,
            "favorites": favorites,
            "total": len(favorites)
        }
    except Exception as e:
        logger.error(f"Error getting favorites: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener favoritos: {str(e)}")


@router.get("/favorites/stats", response_model=dict)
async def get_favorites_stats(user_id: str = Query(...)):
    """Obtiene estadísticas de favoritos"""
    try:
        stats = favorites_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting favorites stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.post("/tags", response_model=dict)
async def add_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
    tags: List[str] = Query(...),
    user_id: Optional[str] = Query(None)
):
    """Agrega tags a un recurso"""
    try:
        success = tagging_service.add_tags(resource_id, resource_type, tags, user_id)
        return {
            "success": True,
            "message": "Tags agregados",
            "tags": tagging_service.get_tags(resource_id, resource_type)
        }
    except Exception as e:
        logger.error(f"Error adding tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar tags: {str(e)}")


@router.delete("/tags", response_model=dict)
async def remove_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
    tags: List[str] = Query(...)
):
    """Elimina tags de un recurso"""
    try:
        success = tagging_service.remove_tags(resource_id, resource_type, tags)
        if not success:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        return {
            "success": True,
            "message": "Tags eliminados",
            "tags": tagging_service.get_tags(resource_id, resource_type)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar tags: {str(e)}")


@router.get("/tags/{resource_id}", response_model=dict)
async def get_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$")
):
    """Obtiene los tags de un recurso"""
    try:
        tags = tagging_service.get_tags(resource_id, resource_type)
        return {
            "success": True,
            "tags": tags
        }
    except Exception as e:
        logger.error(f"Error getting tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener tags: {str(e)}")


@router.get("/tags/search", response_model=dict)
async def search_by_tags(
    tags: List[str] = Query(...),
    resource_type: Optional[str] = Query(None, regex="^(track|analysis|playlist)$")
):
    """Busca recursos por tags"""
    try:
        results = tagging_service.search_by_tags(tags, resource_type)
        return {
            "success": True,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching by tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar por tags: {str(e)}")


@router.get("/tags/popular", response_model=dict)
async def get_popular_tags(
    limit: int = Query(20, ge=1, le=100),
    resource_type: Optional[str] = Query(None, regex="^(track|analysis|playlist)$")
):
    """Obtiene los tags más populares"""
    try:
        tags = tagging_service.get_popular_tags(limit, resource_type)
        return {
            "success": True,
            "tags": tags
        }
    except Exception as e:
        logger.error(f"Error getting popular tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener tags populares: {str(e)}")


@router.post("/webhooks", response_model=dict)
async def register_webhook(
    url: str,
    events: List[str],
    secret: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Registra un nuevo webhook"""
    try:
        # Validar eventos
        valid_events = [WebhookEvent(e) for e in events if e in [ev.value for ev in WebhookEvent]]
        
        if not valid_events:
            raise HTTPException(status_code=400, detail="No hay eventos válidos")
        
        webhook_id = webhook_service.register_webhook(url, valid_events, secret, user_id)
        
        return {
            "success": True,
            "webhook_id": webhook_id,
            "message": "Webhook registrado exitosamente"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Eventos inválidos")
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error al registrar webhook: {str(e)}")


@router.delete("/webhooks/{webhook_id}", response_model=dict)
async def unregister_webhook(webhook_id: str):
    """Elimina un webhook"""
    try:
        success = webhook_service.unregister_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook no encontrado")
        return {
            "success": True,
            "message": "Webhook eliminado"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar webhook: {str(e)}")


@router.get("/webhooks", response_model=dict)
async def list_webhooks(user_id: Optional[str] = None):
    """Lista todos los webhooks"""
    try:
        webhooks = webhook_service.list_webhooks(user_id)
        return {
            "success": True,
            "webhooks": webhooks,
            "total": len(webhooks)
        }
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al listar webhooks: {str(e)}")


@router.post("/auth/register", response_model=dict)
async def register_user(
    username: str,
    email: str,
    password: str
):
    """Registra un nuevo usuario"""
    try:
        user = auth_service.register_user(username, email, password)
        token = auth_service.generate_token(user)
        
        return {
            "success": True,
            "user": user,
            "token": token,
            "message": "Usuario registrado exitosamente"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Error al registrar usuario: {str(e)}")


@router.post("/auth/login", response_model=dict)
async def login(username: str, password: str):
    """Autentica un usuario y genera token"""
    try:
        user = auth_service.authenticate(username, password)
        
        if not user:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        token = auth_service.generate_token(user)
        
        return {
            "success": True,
            "user": user,
            "token": token,
            "message": "Login exitoso"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar sesión: {str(e)}")


@router.get("/auth/me", response_model=dict)
async def get_current_user(
    authorization: Optional[str] = Header(None)
):
    """Obtiene información del usuario actual"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token no proporcionado")
        
        token = authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        
        user = auth_service.get_user(payload["user_id"])
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        return {
            "success": True,
            "user": user
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener usuario: {str(e)}")


@router.post("/playlists", response_model=dict)
async def create_playlist(
    name: str,
    user_id: str = Query(...),
    description: Optional[str] = None,
    is_public: bool = False
):
    """Crea una nueva playlist"""
    try:
        playlist_id = playlist_service.create_playlist(user_id, name, description, is_public)
        return {
            "success": True,
            "playlist_id": playlist_id,
            "message": "Playlist creada exitosamente"
        }
    except Exception as e:
        logger.error(f"Error creating playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al crear playlist: {str(e)}")


@router.get("/playlists", response_model=dict)
async def get_playlists(
    user_id: Optional[str] = None,
    public_only: bool = False
):
    """Obtiene playlists"""
    try:
        if public_only:
            playlists = playlist_service.get_public_playlists()
        elif user_id:
            playlists = playlist_service.get_user_playlists(user_id)
        else:
            raise HTTPException(status_code=400, detail="Debe proporcionar user_id o public_only=true")
        
        return {
            "success": True,
            "playlists": playlists,
            "total": len(playlists)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playlists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener playlists: {str(e)}")


@router.get("/playlists/{playlist_id}", response_model=dict)
async def get_playlist(playlist_id: str):
    """Obtiene una playlist específica"""
    try:
        playlist = playlist_service.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist no encontrada")
        
        return {
            "success": True,
            "playlist": playlist
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener playlist: {str(e)}")


@router.post("/playlists/{playlist_id}/tracks", response_model=dict)
async def add_track_to_playlist(
    playlist_id: str,
    track_id: str,
    track_name: str,
    artists: List[str]
):
    """Agrega una canción a una playlist"""
    try:
        success = playlist_service.add_track_to_playlist(playlist_id, track_id, track_name, artists)
        if not success:
            raise HTTPException(status_code=400, detail="La canción ya está en la playlist o la playlist no existe")
        
        return {
            "success": True,
            "message": "Canción agregada a la playlist"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding track to playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar canción: {str(e)}")


@router.delete("/playlists/{playlist_id}/tracks/{track_id}", response_model=dict)
async def remove_track_from_playlist(playlist_id: str, track_id: str):
    """Elimina una canción de una playlist"""
    try:
        success = playlist_service.remove_track_from_playlist(playlist_id, track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Canción no encontrada en la playlist")
        
        return {
            "success": True,
            "message": "Canción eliminada de la playlist"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing track from playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar canción: {str(e)}")


@router.delete("/playlists/{playlist_id}", response_model=dict)
async def delete_playlist(playlist_id: str, user_id: str = Query(...)):
    """Elimina una playlist"""
    try:
        success = playlist_service.delete_playlist(playlist_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Playlist no encontrada o sin permisos")
        
        return {
            "success": True,
            "message": "Playlist eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar playlist: {str(e)}")


@router.post("/recommendations/intelligent", response_model=dict)
async def intelligent_recommendations(
    track_id: str,
    limit: int = Query(10, ge=1, le=50),
    method: str = Query("similarity", regex="^(similarity|mood|genre)$")
):
    """Recomendaciones inteligentes basadas en ML"""
    try:
        # Obtener track de referencia
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        target_features = spotify_data.get("audio_features", {})
        
        # Obtener recomendaciones de Spotify como base
        spotify_recommendations = spotify_service.get_recommendations(track_id, limit=50)
        
        # Preparar tracks para análisis
        tracks_with_features = []
        for track in spotify_recommendations[:30]:  # Limitar a 30 para análisis
            track_id_rec = track.get("id")
            try:
                features = spotify_service.get_track_audio_features(track_id_rec)
                tracks_with_features.append({
                    "track": track,
                    "audio_features": features
                })
            except:
                continue
        
        # Aplicar método de recomendación
        if method == "similarity":
            recommendations = intelligent_recommender.recommend_similar_tracks(
                target_features, tracks_with_features, limit
            )
        elif method == "mood":
            # Obtener mood del track objetivo
            from ..services.emotion_analyzer import EmotionAnalyzer
            emotion_analyzer = EmotionAnalyzer()
            emotions = emotion_analyzer.analyze_emotions(target_features)
            target_mood = emotions["primary_emotion"]
            
            recommendations = intelligent_recommender.recommend_by_mood(
                target_mood, tracks_with_features, limit
            )
        else:  # genre
            # Obtener género del track objetivo
            from ..services.genre_detector import GenreDetector
            genre_detector = GenreDetector()
            genre_analysis = genre_detector.detect_genre(target_features)
            target_genre = genre_analysis["primary_genre"]
            
            recommendations = intelligent_recommender.recommend_by_genre(
                target_genre, tracks_with_features, limit
            )
        
        return {
            "success": True,
            "seed_track_id": track_id,
            "method": method,
            "recommendations": recommendations,
            "total": len(recommendations)
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting intelligent recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener recomendaciones: {str(e)}")


@router.post("/recommendations/playlist", response_model=dict)
async def recommend_playlist(
    user_preferences: Dict[str, Any],
    playlist_length: int = Query(20, ge=5, le=100)
):
    """Genera una playlist recomendada basada en preferencias"""
    try:
        # Obtener tracks de ejemplo (podría venir de favoritos, historial, etc.)
        # Por ahora usamos recomendaciones de Spotify como base
        seed_track_id = user_preferences.get("seed_track_id")
        
        if seed_track_id:
            spotify_recommendations = spotify_service.get_recommendations(seed_track_id, limit=100)
        else:
            # Si no hay seed, usar búsqueda genérica
            spotify_recommendations = []
        
        # Preparar tracks con features
        available_tracks = []
        for track in spotify_recommendations[:50]:
            track_id = track.get("id")
            try:
                features = spotify_service.get_track_audio_features(track_id)
                available_tracks.append({
                    "track": track,
                    "audio_features": features
                })
            except:
                continue
        
        # Generar playlist recomendada
        recommended_playlist = intelligent_recommender.recommend_playlist(
            user_preferences, available_tracks, playlist_length
        )
        
        return {
            "success": True,
            "playlist": recommended_playlist,
            "total": len(recommended_playlist)
        }
    except Exception as e:
        logger.error(f"Error generating recommended playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar playlist: {str(e)}")


@router.get("/dashboard", response_model=dict)
async def get_dashboard(user_id: Optional[str] = None):
    """Obtiene dashboard completo de métricas"""
    try:
        dashboard = dashboard_service.get_comprehensive_dashboard(user_id)
        return {
            "success": True,
            "dashboard": dashboard
        }
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener dashboard: {str(e)}")


@router.get("/notifications", response_model=dict)
async def get_notifications(
    user_id: str = Query(...),
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=100)
):
    """Obtiene notificaciones de un usuario"""
    try:
        notifications = notification_service.get_notifications(user_id, unread_only, limit)
        unread_count = notification_service.get_unread_count(user_id)
        
        return {
            "success": True,
            "notifications": notifications,
            "unread_count": unread_count,
            "total": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener notificaciones: {str(e)}")


@router.put("/notifications/{notification_id}/read", response_model=dict)
async def mark_notification_read(
    notification_id: str,
    user_id: str = Query(...)
):
    """Marca una notificación como leída"""
    try:
        success = notification_service.mark_as_read(user_id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación marcada como leída"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar notificación: {str(e)}")


@router.put("/notifications/read-all", response_model=dict)
async def mark_all_notifications_read(user_id: str = Query(...)):
    """Marca todas las notificaciones como leídas"""
    try:
        count = notification_service.mark_all_as_read(user_id)
        return {
            "success": True,
            "message": f"{count} notificaciones marcadas como leídas",
            "count": count
        }
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar notificaciones: {str(e)}")


@router.delete("/notifications/{notification_id}", response_model=dict)
async def delete_notification(
    notification_id: str,
    user_id: str = Query(...)
):
    """Elimina una notificación"""
    try:
        success = notification_service.delete_notification(user_id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar notificación: {str(e)}")


@router.get("/notifications/stats", response_model=dict)
async def get_notification_stats(user_id: str = Query(...)):
    """Obtiene estadísticas de notificaciones"""
    try:
        stats = notification_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.2.0 ==========

@router.get("/trends/popularity", response_model=dict)
async def get_popularity_trends(track_id: str = Query(...)):
    """Analiza tendencias de popularidad de un track"""
    try:
        trends = trends_analyzer.analyze_popularity_trends(track_id)
        return {
            "success": True,
            "track_id": track_id,
            "trends": trends
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing popularity trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar tendencias: {str(e)}")


@router.post("/trends/artists", response_model=dict)
async def analyze_artist_trends(artist_ids: List[str]):
    """Analiza tendencias de múltiples artistas"""
    try:
        if len(artist_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 artistas")
        
        trends = trends_analyzer.analyze_artist_trends(artist_ids)
        return {
            "success": True,
            "trends": trends
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing artist trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar tendencias de artistas: {str(e)}")


@router.post("/predict/success", response_model=dict)
async def predict_commercial_success(track_id: str = Query(...)):
    """Predice el éxito comercial de un track"""
    try:
        prediction = trends_analyzer.predict_commercial_success(track_id)
        return {
            "success": True,
            "prediction": prediction
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting success: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir éxito: {str(e)}")


@router.get("/rhythmic/patterns", response_model=dict)
async def get_rhythmic_patterns(track_id: str = Query(...)):
    """Analiza patrones rítmicos avanzados de un track"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        patterns = trends_analyzer.analyze_rhythmic_patterns(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "rhythmic_patterns": patterns
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing rhythmic patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar patrones rítmicos: {str(e)}")


@router.get("/collaborations/analyze", response_model=dict)
async def analyze_collaborations(track_id: str = Query(...)):
    """Analiza las colaboraciones en un track"""
    try:
        analysis = collaboration_analyzer.analyze_track_collaborations(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing collaborations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar colaboraciones: {str(e)}")


@router.post("/collaborations/network", response_model=dict)
async def analyze_artist_network(artist_ids: List[str]):
    """Analiza la red de colaboraciones de artistas"""
    try:
        if len(artist_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 artistas")
        
        network = collaboration_analyzer.analyze_artist_network(artist_ids)
        return {
            "success": True,
            "network": network
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing artist network: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar red de artistas: {str(e)}")


@router.post("/versions/compare", response_model=dict)
async def compare_versions(track_ids: List[str]):
    """Compara diferentes versiones de una canción"""
    try:
        if len(track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tracks")
        
        if len(track_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 tracks")
        
        comparison = collaboration_analyzer.compare_versions(track_ids)
        return {
            "success": True,
            "comparison": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar versiones: {str(e)}")


@router.get("/alerts/check", response_model=dict)
async def check_alerts(
    track_id: str = Query(...),
    previous_popularity: Optional[int] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """Verifica alertas para un track"""
    try:
        track_info = spotify_service.get_track(track_id)
        current_popularity = track_info.get("popularity", 0)
        
        # Alertas de popularidad
        popularity_alerts = alert_service.check_popularity_alerts(
            track_id, current_popularity, previous_popularity
        )
        
        # Obtener audio features para alertas de tendencias
        audio_features = spotify_service.get_track_audio_features(track_id)
        
        # Detectar género y emoción
        from ..services.genre_detector import GenreDetector
        from ..services.emotion_analyzer import EmotionAnalyzer
        genre_detector = GenreDetector()
        emotion_analyzer = EmotionAnalyzer()
        
        genre = genre_detector.detect_genre(audio_features)
        emotion = emotion_analyzer.analyze_emotions(audio_features)
        
        # Alertas de oportunidades
        trend_alerts = alert_service.check_trend_opportunities(
            track_id, audio_features,
            genre.get("primary_genre", "Unknown"),
            emotion.get("primary_emotion", "Unknown")
        )
        
        # Alertas de colaboraciones
        artists = track_info.get("artists", [])
        collaboration_alerts = alert_service.check_collaboration_alerts(track_id, artists)
        
        all_alerts = popularity_alerts + trend_alerts + collaboration_alerts
        
        return {
            "success": True,
            "track_id": track_id,
            "alerts_count": len(all_alerts),
            "alerts": all_alerts
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error al verificar alertas: {str(e)}")


@router.get("/alerts", response_model=dict)
async def get_alerts(
    user_id: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    priority: Optional[str] = Query(None)
):
    """Obtiene todas las alertas con filtros"""
    try:
        alerts = alert_service.get_all_alerts(user_id, alert_type, priority)
        return {
            "success": True,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener alertas: {str(e)}")


@router.put("/alerts/{alert_id}/read", response_model=dict)
async def mark_alert_read(alert_id: str):
    """Marca una alerta como leída"""
    try:
        success = alert_service.mark_alert_read(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        return {
            "success": True,
            "message": "Alerta marcada como leída"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking alert as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar alerta: {str(e)}")


@router.delete("/alerts/{alert_id}", response_model=dict)
async def delete_alert(alert_id: str):
    """Elimina una alerta"""
    try:
        success = alert_service.delete_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        return {
            "success": True,
            "message": "Alerta eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar alerta: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.3.0 ==========

@router.get("/temporal/structure", response_model=dict)
async def get_temporal_structure(track_id: str = Query(...)):
    """Analiza la estructura temporal de una canción"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        structure = temporal_analyzer.analyze_temporal_structure(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "temporal_structure": structure
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing temporal structure: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar estructura temporal: {str(e)}")


@router.get("/temporal/energy", response_model=dict)
async def get_energy_progression(track_id: str = Query(...)):
    """Analiza la progresión de energía a lo largo del tiempo"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        progression = temporal_analyzer.analyze_energy_progression(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "energy_progression": progression
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing energy progression: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar progresión de energía: {str(e)}")


@router.get("/temporal/tempo", response_model=dict)
async def get_tempo_changes(track_id: str = Query(...)):
    """Analiza cambios de tempo a lo largo del tiempo"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        changes = temporal_analyzer.analyze_tempo_changes(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "tempo_changes": changes
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing tempo changes: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar cambios de tempo: {str(e)}")


@router.get("/quality/analyze", response_model=dict)
async def analyze_quality(track_id: str = Query(...)):
    """Analiza la calidad de producción de un track"""
    try:
        quality = quality_analyzer.analyze_production_quality(track_id)
        return {
            "success": True,
            "quality_analysis": quality
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing quality: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar calidad: {str(e)}")


@router.post("/recommendations/contextual", response_model=dict)
async def recommend_contextual(
    track_id: str = Query(...),
    context: Optional[Dict[str, Any]] = None,
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en contexto personalizado"""
    try:
        if context is None:
            context = {}
        
        recommendations = contextual_recommender.recommend_by_context(track_id, context, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in contextual recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación contextual: {str(e)}")


@router.post("/recommendations/time-of-day", response_model=dict)
async def recommend_by_time(
    track_id: str = Query(...),
    time_of_day: str = Query(..., regex="^(morning|afternoon|evening|night)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en hora del día"""
    try:
        recommendations = contextual_recommender.recommend_by_time_of_day(track_id, time_of_day, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in time-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por hora: {str(e)}")


@router.post("/recommendations/activity", response_model=dict)
async def recommend_by_activity(
    track_id: str = Query(...),
    activity: str = Query(..., regex="^(workout|study|party|relax|drive)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en actividad"""
    try:
        recommendations = contextual_recommender.recommend_by_activity(track_id, activity, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in activity-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por actividad: {str(e)}")


@router.post("/recommendations/mood", response_model=dict)
async def recommend_by_mood(
    track_id: str = Query(...),
    target_mood: str = Query(..., regex="^(happy|sad|energetic|calm|romantic)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en mood objetivo"""
    try:
        recommendations = contextual_recommender.recommend_by_mood(track_id, target_mood, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in mood-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por mood: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.4.0 ==========

@router.post("/playlists/analyze", response_model=dict)
async def analyze_playlist(track_ids: List[str]):
    """Analiza una playlist completa"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        analysis = playlist_analyzer.analyze_playlist(track_ids)
        return {
            "success": True,
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar playlist: {str(e)}")


@router.post("/playlists/suggest-improvements", response_model=dict)
async def suggest_playlist_improvements(track_ids: List[str]):
    """Sugiere mejoras para una playlist"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        suggestions = playlist_analyzer.suggest_playlist_improvements(track_ids)
        return {
            "success": True,
            "suggestions": suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting improvements: {e}")
        raise HTTPException(status_code=500, detail=f"Error al sugerir mejoras: {str(e)}")


@router.post("/playlists/optimize-order", response_model=dict)
async def optimize_playlist_order(track_ids: List[str]):
    """Optimiza el orden de una playlist"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        optimization = playlist_analyzer.optimize_playlist_order(track_ids)
        return {
            "success": True,
            "optimization": optimization
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing playlist order: {e}")
        raise HTTPException(status_code=500, detail=f"Error al optimizar orden: {str(e)}")


@router.post("/artists/compare", response_model=dict)
async def compare_artists(
    artist_names: List[str],
    limit_per_artist: int = Query(10, ge=5, le=20)
):
    """Compara múltiples artistas"""
    try:
        if len(artist_names) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 artistas")
        
        if len(artist_names) > 5:
            raise HTTPException(status_code=400, detail="Máximo 5 artistas")
        
        comparison = artist_comparator.compare_artists(artist_names, limit_per_artist)
        return {
            "success": True,
            "comparison": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing artists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar artistas: {str(e)}")


@router.get("/artists/evolution", response_model=dict)
async def analyze_artist_evolution(
    artist_name: str = Query(...),
    limit: int = Query(20, ge=5, le=50)
):
    """Analiza la evolución musical de un artista"""
    try:
        evolution = artist_comparator.analyze_artist_evolution(artist_name, limit)
        return {
            "success": True,
            "evolution": evolution
        }
    except Exception as e:
        logger.error(f"Error analyzing artist evolution: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar evolución: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.5.0 ==========

@router.get("/discovery/similar-artists", response_model=dict)
async def discover_similar_artists(
    artist_name: str = Query(...),
    limit: int = Query(10, ge=1, le=20)
):
    """Descubre artistas similares a uno dado"""
    try:
        discovery = discovery_service.discover_similar_artists(artist_name, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering similar artists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir artistas: {str(e)}")


@router.get("/discovery/underground", response_model=dict)
async def discover_underground_tracks(
    genre: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50)
):
    """Descubre tracks underground (baja popularidad pero alta calidad)"""
    try:
        discovery = discovery_service.discover_underground_tracks(genre, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering underground tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir tracks underground: {str(e)}")


@router.get("/discovery/mood-transition", response_model=dict)
async def discover_mood_transition(
    start_mood: str = Query(..., regex="^(sad|happy|energetic|calm|romantic|angry)$"),
    target_mood: str = Query(..., regex="^(sad|happy|energetic|calm|romantic|angry)$"),
    limit: int = Query(10, ge=1, le=20)
):
    """Descubre tracks que transicionan entre moods"""
    try:
        discovery = discovery_service.discover_by_mood_transition(start_mood, target_mood, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering mood transitions: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir transiciones: {str(e)}")


@router.get("/discovery/fresh", response_model=dict)
async def discover_fresh_tracks(
    genre: Optional[str] = Query(None),
    days_old: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=50)
):
    """Descubre tracks frescos (recientes)"""
    try:
        discovery = discovery_service.discover_fresh_tracks(genre, days_old, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering fresh tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir tracks frescos: {str(e)}")


@router.get("/covers/analyze", response_model=dict)
async def analyze_cover(
    original_track_id: str = Query(...),
    cover_track_id: str = Query(...)
):
    """Analiza un cover comparándolo con el original"""
    try:
        analysis = cover_remix_analyzer.analyze_cover(original_track_id, cover_track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing cover: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar cover: {str(e)}")


@router.get("/remixes/analyze", response_model=dict)
async def analyze_remix(
    original_track_id: str = Query(...),
    remix_track_id: str = Query(...)
):
    """Analiza un remix comparándolo con el original"""
    try:
        analysis = cover_remix_analyzer.analyze_remix(original_track_id, remix_track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing remix: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar remix: {str(e)}")


@router.get("/covers/find", response_model=dict)
async def find_covers_and_remixes(
    track_id: str = Query(...),
    limit: int = Query(20, ge=1, le=50)
):
    """Encuentra covers y remixes de un track"""
    try:
        results = cover_remix_analyzer.find_covers_and_remixes(track_id, limit)
        return {
            "success": True,
            "results": results
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error finding covers and remixes: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar covers y remixes: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.6.0 ==========

@router.get("/trends/predict/genre", response_model=dict)
async def predict_genre_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de géneros musicales"""
    try:
        prediction = trend_predictor.predict_genre_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting genre trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/emotion", response_model=dict)
async def predict_emotion_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de emociones musicales"""
    try:
        prediction = trend_predictor.predict_emotion_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting emotion trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/features", response_model=dict)
async def predict_feature_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de características musicales"""
    try:
        prediction = trend_predictor.predict_feature_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting feature trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/instrumentation/analyze", response_model=dict)
async def analyze_instrumentation(track_id: str = Query(...)):
    """Analiza la instrumentación de un track"""
    try:
        analysis = instrumentation_analyzer.analyze_instrumentation(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing instrumentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar instrumentación: {str(e)}")


@router.post("/export/csv", response_model=dict)
async def export_to_csv(analyses: List[Dict[str, Any]]):
    """Exporta múltiples análisis a CSV"""
    try:
        csv_data = export_service.export_to_csv(analyses)
        return {
            "success": True,
            "format": "csv",
            "data": csv_data
        }
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar a CSV: {str(e)}")


@router.get("/export/comprehensive/{track_id}", response_model=dict)
async def export_comprehensive_report(track_id: str):
    """Exporta un reporte comprehensivo en Markdown"""
    try:
        # Obtener análisis completo
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        report = export_service.export_comprehensive_report(analysis)
        
        return {
            "success": True,
            "format": "markdown",
            "report": report
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar reporte: {str(e)}")


@router.post("/export/{track_id}", response_model=dict)
async def export_analysis(
    track_id: str,
    format: str = Query("json", regex="^(json|text|markdown)$"),
    include_coaching: bool = Query(True)
):
    """
    Exporta un análisis a diferentes formatos
    
    - **track_id**: ID de la canción
    - **format**: Formato de exportación (json, text, markdown)
    - **include_coaching**: Incluir análisis de coaching
    """
    try:
        start_time = time.time()
        
        # Obtener análisis
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        if include_coaching:
            coaching = music_coach.generate_coaching_analysis(analysis)
            analysis["coaching"] = coaching
        
        # Exportar según formato
        if format == "json":
            content = export_service.export_to_json(analysis, include_coaching)
            content_type = "application/json"
        elif format == "text":
            content = export_service.export_to_text(analysis, include_coaching)
            content_type = "text/plain"
        else:  # markdown
            content = export_service.export_to_markdown(analysis, include_coaching)
            content_type = "text/markdown"
        
        response_time = time.time() - start_time
        analytics_service.track_request(f"/export/{track_id}", "POST", response_time=response_time)
        
        return {
            "success": True,
            "format": format,
            "content": content,
            "content_type": content_type,
            "track_id": track_id
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar análisis: {str(e)}")


@router.get("/history", response_model=dict)
async def get_history(
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Obtiene el historial de análisis"""
    try:
        history = history_service.get_history(user_id, limit)
        return {
            "success": True,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.get("/history/stats", response_model=dict)
async def get_history_stats(user_id: Optional[str] = Query(None)):
    """Obtiene estadísticas del historial"""
    try:
        stats = history_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting history stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.delete("/history/{analysis_id}", response_model=dict)
async def delete_history_entry(
    analysis_id: str,
    user_id: Optional[str] = Query(None)
):
    """Elimina una entrada del historial"""
    try:
        success = history_service.delete_analysis(analysis_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Análisis no encontrado")
        return {
            "success": True,
            "message": "Análisis eliminado del historial"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history entry: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar entrada: {str(e)}")


@router.get("/analytics", response_model=dict)
async def get_analytics():
    """Obtiene estadísticas y métricas del sistema"""
    try:
        stats = analytics_service.get_stats()
        return {
            "success": True,
            "analytics": stats
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener analytics: {str(e)}")


@router.post("/analytics/reset", response_model=dict)
async def reset_analytics():
    """Resetea las estadísticas de analytics"""
    try:
        analytics_service.reset_stats()
        return {
            "success": True,
            "message": "Analytics reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al resetear analytics: {str(e)}")


@router.post("/favorites", response_model=dict)
async def add_favorite(
    track_id: str,
    track_name: str,
    artists: List[str],
    user_id: str = Query(..., description="ID del usuario"),
    notes: Optional[str] = Query(None, description="Notas opcionales")
):
    """Agrega una canción a favoritos"""
    try:
        success = favorites_service.add_favorite(user_id, track_id, track_name, artists, notes)
        if not success:
            raise HTTPException(status_code=400, detail="La canción ya está en favoritos")
        return {
            "success": True,
            "message": "Canción agregada a favoritos"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar favorito: {str(e)}")


@router.delete("/favorites/{track_id}", response_model=dict)
async def remove_favorite(track_id: str, user_id: str = Query(...)):
    """Elimina una canción de favoritos"""
    try:
        success = favorites_service.remove_favorite(user_id, track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Canción no encontrada en favoritos")
        return {
            "success": True,
            "message": "Canción eliminada de favoritos"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing favorite: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar favorito: {str(e)}")


@router.get("/favorites", response_model=dict)
async def get_favorites(user_id: str = Query(...)):
    """Obtiene los favoritos de un usuario"""
    try:
        favorites = favorites_service.get_favorites(user_id)
        return {
            "success": True,
            "favorites": favorites,
            "total": len(favorites)
        }
    except Exception as e:
        logger.error(f"Error getting favorites: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener favoritos: {str(e)}")


@router.get("/favorites/stats", response_model=dict)
async def get_favorites_stats(user_id: str = Query(...)):
    """Obtiene estadísticas de favoritos"""
    try:
        stats = favorites_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting favorites stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


@router.post("/tags", response_model=dict)
async def add_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
    tags: List[str] = Query(...),
    user_id: Optional[str] = Query(None)
):
    """Agrega tags a un recurso"""
    try:
        success = tagging_service.add_tags(resource_id, resource_type, tags, user_id)
        return {
            "success": True,
            "message": "Tags agregados",
            "tags": tagging_service.get_tags(resource_id, resource_type)
        }
    except Exception as e:
        logger.error(f"Error adding tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar tags: {str(e)}")


@router.delete("/tags", response_model=dict)
async def remove_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$"),
    tags: List[str] = Query(...)
):
    """Elimina tags de un recurso"""
    try:
        success = tagging_service.remove_tags(resource_id, resource_type, tags)
        if not success:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        return {
            "success": True,
            "message": "Tags eliminados",
            "tags": tagging_service.get_tags(resource_id, resource_type)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar tags: {str(e)}")


@router.get("/tags/{resource_id}", response_model=dict)
async def get_tags(
    resource_id: str,
    resource_type: str = Query("track", regex="^(track|analysis|playlist)$")
):
    """Obtiene los tags de un recurso"""
    try:
        tags = tagging_service.get_tags(resource_id, resource_type)
        return {
            "success": True,
            "tags": tags
        }
    except Exception as e:
        logger.error(f"Error getting tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener tags: {str(e)}")


@router.get("/tags/search", response_model=dict)
async def search_by_tags(
    tags: List[str] = Query(...),
    resource_type: Optional[str] = Query(None, regex="^(track|analysis|playlist)$")
):
    """Busca recursos por tags"""
    try:
        results = tagging_service.search_by_tags(tags, resource_type)
        return {
            "success": True,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Error searching by tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar por tags: {str(e)}")


@router.get("/tags/popular", response_model=dict)
async def get_popular_tags(
    limit: int = Query(20, ge=1, le=100),
    resource_type: Optional[str] = Query(None, regex="^(track|analysis|playlist)$")
):
    """Obtiene los tags más populares"""
    try:
        tags = tagging_service.get_popular_tags(limit, resource_type)
        return {
            "success": True,
            "tags": tags
        }
    except Exception as e:
        logger.error(f"Error getting popular tags: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener tags populares: {str(e)}")


@router.post("/webhooks", response_model=dict)
async def register_webhook(
    url: str,
    events: List[str],
    secret: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Registra un nuevo webhook"""
    try:
        # Validar eventos
        valid_events = [WebhookEvent(e) for e in events if e in [ev.value for ev in WebhookEvent]]
        
        if not valid_events:
            raise HTTPException(status_code=400, detail="No hay eventos válidos")
        
        webhook_id = webhook_service.register_webhook(url, valid_events, secret, user_id)
        
        return {
            "success": True,
            "webhook_id": webhook_id,
            "message": "Webhook registrado exitosamente"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Eventos inválidos")
    except Exception as e:
        logger.error(f"Error registering webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error al registrar webhook: {str(e)}")


@router.delete("/webhooks/{webhook_id}", response_model=dict)
async def unregister_webhook(webhook_id: str):
    """Elimina un webhook"""
    try:
        success = webhook_service.unregister_webhook(webhook_id)
        if not success:
            raise HTTPException(status_code=404, detail="Webhook no encontrado")
        return {
            "success": True,
            "message": "Webhook eliminado"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar webhook: {str(e)}")


@router.get("/webhooks", response_model=dict)
async def list_webhooks(user_id: Optional[str] = None):
    """Lista todos los webhooks"""
    try:
        webhooks = webhook_service.list_webhooks(user_id)
        return {
            "success": True,
            "webhooks": webhooks,
            "total": len(webhooks)
        }
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al listar webhooks: {str(e)}")


@router.post("/auth/register", response_model=dict)
async def register_user(
    username: str,
    email: str,
    password: str
):
    """Registra un nuevo usuario"""
    try:
        user = auth_service.register_user(username, email, password)
        token = auth_service.generate_token(user)
        
        return {
            "success": True,
            "user": user,
            "token": token,
            "message": "Usuario registrado exitosamente"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=f"Error al registrar usuario: {str(e)}")


@router.post("/auth/login", response_model=dict)
async def login(username: str, password: str):
    """Autentica un usuario y genera token"""
    try:
        user = auth_service.authenticate(username, password)
        
        if not user:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        token = auth_service.generate_token(user)
        
        return {
            "success": True,
            "user": user,
            "token": token,
            "message": "Login exitoso"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar sesión: {str(e)}")


@router.get("/auth/me", response_model=dict)
async def get_current_user(
    authorization: Optional[str] = Header(None)
):
    """Obtiene información del usuario actual"""
    try:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Token no proporcionado")
        
        token = authorization.replace("Bearer ", "")
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        
        user = auth_service.get_user(payload["user_id"])
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        return {
            "success": True,
            "user": user
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener usuario: {str(e)}")


@router.post("/playlists", response_model=dict)
async def create_playlist(
    name: str,
    user_id: str = Query(...),
    description: Optional[str] = None,
    is_public: bool = False
):
    """Crea una nueva playlist"""
    try:
        playlist_id = playlist_service.create_playlist(user_id, name, description, is_public)
        return {
            "success": True,
            "playlist_id": playlist_id,
            "message": "Playlist creada exitosamente"
        }
    except Exception as e:
        logger.error(f"Error creating playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al crear playlist: {str(e)}")


@router.get("/playlists", response_model=dict)
async def get_playlists(
    user_id: Optional[str] = None,
    public_only: bool = False
):
    """Obtiene playlists"""
    try:
        if public_only:
            playlists = playlist_service.get_public_playlists()
        elif user_id:
            playlists = playlist_service.get_user_playlists(user_id)
        else:
            raise HTTPException(status_code=400, detail="Debe proporcionar user_id o public_only=true")
        
        return {
            "success": True,
            "playlists": playlists,
            "total": len(playlists)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playlists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener playlists: {str(e)}")


@router.get("/playlists/{playlist_id}", response_model=dict)
async def get_playlist(playlist_id: str):
    """Obtiene una playlist específica"""
    try:
        playlist = playlist_service.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(status_code=404, detail="Playlist no encontrada")
        
        return {
            "success": True,
            "playlist": playlist
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener playlist: {str(e)}")


@router.post("/playlists/{playlist_id}/tracks", response_model=dict)
async def add_track_to_playlist(
    playlist_id: str,
    track_id: str,
    track_name: str,
    artists: List[str]
):
    """Agrega una canción a una playlist"""
    try:
        success = playlist_service.add_track_to_playlist(playlist_id, track_id, track_name, artists)
        if not success:
            raise HTTPException(status_code=400, detail="La canción ya está en la playlist o la playlist no existe")
        
        return {
            "success": True,
            "message": "Canción agregada a la playlist"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding track to playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agregar canción: {str(e)}")


@router.delete("/playlists/{playlist_id}/tracks/{track_id}", response_model=dict)
async def remove_track_from_playlist(playlist_id: str, track_id: str):
    """Elimina una canción de una playlist"""
    try:
        success = playlist_service.remove_track_from_playlist(playlist_id, track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Canción no encontrada en la playlist")
        
        return {
            "success": True,
            "message": "Canción eliminada de la playlist"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing track from playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar canción: {str(e)}")


@router.delete("/playlists/{playlist_id}", response_model=dict)
async def delete_playlist(playlist_id: str, user_id: str = Query(...)):
    """Elimina una playlist"""
    try:
        success = playlist_service.delete_playlist(playlist_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Playlist no encontrada o sin permisos")
        
        return {
            "success": True,
            "message": "Playlist eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar playlist: {str(e)}")


@router.post("/recommendations/intelligent", response_model=dict)
async def intelligent_recommendations(
    track_id: str,
    limit: int = Query(10, ge=1, le=50),
    method: str = Query("similarity", regex="^(similarity|mood|genre)$")
):
    """Recomendaciones inteligentes basadas en ML"""
    try:
        # Obtener track de referencia
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        target_features = spotify_data.get("audio_features", {})
        
        # Obtener recomendaciones de Spotify como base
        spotify_recommendations = spotify_service.get_recommendations(track_id, limit=50)
        
        # Preparar tracks para análisis
        tracks_with_features = []
        for track in spotify_recommendations[:30]:  # Limitar a 30 para análisis
            track_id_rec = track.get("id")
            try:
                features = spotify_service.get_track_audio_features(track_id_rec)
                tracks_with_features.append({
                    "track": track,
                    "audio_features": features
                })
            except:
                continue
        
        # Aplicar método de recomendación
        if method == "similarity":
            recommendations = intelligent_recommender.recommend_similar_tracks(
                target_features, tracks_with_features, limit
            )
        elif method == "mood":
            # Obtener mood del track objetivo
            from ..services.emotion_analyzer import EmotionAnalyzer
            emotion_analyzer = EmotionAnalyzer()
            emotions = emotion_analyzer.analyze_emotions(target_features)
            target_mood = emotions["primary_emotion"]
            
            recommendations = intelligent_recommender.recommend_by_mood(
                target_mood, tracks_with_features, limit
            )
        else:  # genre
            # Obtener género del track objetivo
            from ..services.genre_detector import GenreDetector
            genre_detector = GenreDetector()
            genre_analysis = genre_detector.detect_genre(target_features)
            target_genre = genre_analysis["primary_genre"]
            
            recommendations = intelligent_recommender.recommend_by_genre(
                target_genre, tracks_with_features, limit
            )
        
        return {
            "success": True,
            "seed_track_id": track_id,
            "method": method,
            "recommendations": recommendations,
            "total": len(recommendations)
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting intelligent recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener recomendaciones: {str(e)}")


@router.post("/recommendations/playlist", response_model=dict)
async def recommend_playlist(
    user_preferences: Dict[str, Any],
    playlist_length: int = Query(20, ge=5, le=100)
):
    """Genera una playlist recomendada basada en preferencias"""
    try:
        # Obtener tracks de ejemplo (podría venir de favoritos, historial, etc.)
        # Por ahora usamos recomendaciones de Spotify como base
        seed_track_id = user_preferences.get("seed_track_id")
        
        if seed_track_id:
            spotify_recommendations = spotify_service.get_recommendations(seed_track_id, limit=100)
        else:
            # Si no hay seed, usar búsqueda genérica
            spotify_recommendations = []
        
        # Preparar tracks con features
        available_tracks = []
        for track in spotify_recommendations[:50]:
            track_id = track.get("id")
            try:
                features = spotify_service.get_track_audio_features(track_id)
                available_tracks.append({
                    "track": track,
                    "audio_features": features
                })
            except:
                continue
        
        # Generar playlist recomendada
        recommended_playlist = intelligent_recommender.recommend_playlist(
            user_preferences, available_tracks, playlist_length
        )
        
        return {
            "success": True,
            "playlist": recommended_playlist,
            "total": len(recommended_playlist)
        }
    except Exception as e:
        logger.error(f"Error generating recommended playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar playlist: {str(e)}")


@router.get("/dashboard", response_model=dict)
async def get_dashboard(user_id: Optional[str] = None):
    """Obtiene dashboard completo de métricas"""
    try:
        dashboard = dashboard_service.get_comprehensive_dashboard(user_id)
        return {
            "success": True,
            "dashboard": dashboard
        }
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener dashboard: {str(e)}")


@router.get("/notifications", response_model=dict)
async def get_notifications(
    user_id: str = Query(...),
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=100)
):
    """Obtiene notificaciones de un usuario"""
    try:
        notifications = notification_service.get_notifications(user_id, unread_only, limit)
        unread_count = notification_service.get_unread_count(user_id)
        
        return {
            "success": True,
            "notifications": notifications,
            "unread_count": unread_count,
            "total": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener notificaciones: {str(e)}")


@router.put("/notifications/{notification_id}/read", response_model=dict)
async def mark_notification_read(
    notification_id: str,
    user_id: str = Query(...)
):
    """Marca una notificación como leída"""
    try:
        success = notification_service.mark_as_read(user_id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación marcada como leída"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar notificación: {str(e)}")


@router.put("/notifications/read-all", response_model=dict)
async def mark_all_notifications_read(user_id: str = Query(...)):
    """Marca todas las notificaciones como leídas"""
    try:
        count = notification_service.mark_all_as_read(user_id)
        return {
            "success": True,
            "message": f"{count} notificaciones marcadas como leídas",
            "count": count
        }
    except Exception as e:
        logger.error(f"Error marking all notifications as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar notificaciones: {str(e)}")


@router.delete("/notifications/{notification_id}", response_model=dict)
async def delete_notification(
    notification_id: str,
    user_id: str = Query(...)
):
    """Elimina una notificación"""
    try:
        success = notification_service.delete_notification(user_id, notification_id)
        if not success:
            raise HTTPException(status_code=404, detail="Notificación no encontrada")
        
        return {
            "success": True,
            "message": "Notificación eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar notificación: {str(e)}")


@router.get("/notifications/stats", response_model=dict)
async def get_notification_stats(user_id: str = Query(...)):
    """Obtiene estadísticas de notificaciones"""
    try:
        stats = notification_service.get_stats(user_id)
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.2.0 ==========

@router.get("/trends/popularity", response_model=dict)
async def get_popularity_trends(track_id: str = Query(...)):
    """Analiza tendencias de popularidad de un track"""
    try:
        trends = trends_analyzer.analyze_popularity_trends(track_id)
        return {
            "success": True,
            "track_id": track_id,
            "trends": trends
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing popularity trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar tendencias: {str(e)}")


@router.post("/trends/artists", response_model=dict)
async def analyze_artist_trends(artist_ids: List[str]):
    """Analiza tendencias de múltiples artistas"""
    try:
        if len(artist_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 artistas")
        
        trends = trends_analyzer.analyze_artist_trends(artist_ids)
        return {
            "success": True,
            "trends": trends
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing artist trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar tendencias de artistas: {str(e)}")


@router.post("/predict/success", response_model=dict)
async def predict_commercial_success(track_id: str = Query(...)):
    """Predice el éxito comercial de un track"""
    try:
        prediction = trends_analyzer.predict_commercial_success(track_id)
        return {
            "success": True,
            "prediction": prediction
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting success: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir éxito: {str(e)}")


@router.get("/rhythmic/patterns", response_model=dict)
async def get_rhythmic_patterns(track_id: str = Query(...)):
    """Analiza patrones rítmicos avanzados de un track"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        patterns = trends_analyzer.analyze_rhythmic_patterns(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "rhythmic_patterns": patterns
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing rhythmic patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar patrones rítmicos: {str(e)}")


@router.get("/collaborations/analyze", response_model=dict)
async def analyze_collaborations(track_id: str = Query(...)):
    """Analiza las colaboraciones en un track"""
    try:
        analysis = collaboration_analyzer.analyze_track_collaborations(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing collaborations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar colaboraciones: {str(e)}")


@router.post("/collaborations/network", response_model=dict)
async def analyze_artist_network(artist_ids: List[str]):
    """Analiza la red de colaboraciones de artistas"""
    try:
        if len(artist_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 artistas")
        
        network = collaboration_analyzer.analyze_artist_network(artist_ids)
        return {
            "success": True,
            "network": network
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing artist network: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar red de artistas: {str(e)}")


@router.post("/versions/compare", response_model=dict)
async def compare_versions(track_ids: List[str]):
    """Compara diferentes versiones de una canción"""
    try:
        if len(track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tracks")
        
        if len(track_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 tracks")
        
        comparison = collaboration_analyzer.compare_versions(track_ids)
        return {
            "success": True,
            "comparison": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar versiones: {str(e)}")


@router.get("/alerts/check", response_model=dict)
async def check_alerts(
    track_id: str = Query(...),
    previous_popularity: Optional[int] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """Verifica alertas para un track"""
    try:
        track_info = spotify_service.get_track(track_id)
        current_popularity = track_info.get("popularity", 0)
        
        # Alertas de popularidad
        popularity_alerts = alert_service.check_popularity_alerts(
            track_id, current_popularity, previous_popularity
        )
        
        # Obtener audio features para alertas de tendencias
        audio_features = spotify_service.get_track_audio_features(track_id)
        
        # Detectar género y emoción
        from ..services.genre_detector import GenreDetector
        from ..services.emotion_analyzer import EmotionAnalyzer
        genre_detector = GenreDetector()
        emotion_analyzer = EmotionAnalyzer()
        
        genre = genre_detector.detect_genre(audio_features)
        emotion = emotion_analyzer.analyze_emotions(audio_features)
        
        # Alertas de oportunidades
        trend_alerts = alert_service.check_trend_opportunities(
            track_id, audio_features,
            genre.get("primary_genre", "Unknown"),
            emotion.get("primary_emotion", "Unknown")
        )
        
        # Alertas de colaboraciones
        artists = track_info.get("artists", [])
        collaboration_alerts = alert_service.check_collaboration_alerts(track_id, artists)
        
        all_alerts = popularity_alerts + trend_alerts + collaboration_alerts
        
        return {
            "success": True,
            "track_id": track_id,
            "alerts_count": len(all_alerts),
            "alerts": all_alerts
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error al verificar alertas: {str(e)}")


@router.get("/alerts", response_model=dict)
async def get_alerts(
    user_id: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    priority: Optional[str] = Query(None)
):
    """Obtiene todas las alertas con filtros"""
    try:
        alerts = alert_service.get_all_alerts(user_id, alert_type, priority)
        return {
            "success": True,
            "alerts_count": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener alertas: {str(e)}")


@router.put("/alerts/{alert_id}/read", response_model=dict)
async def mark_alert_read(alert_id: str):
    """Marca una alerta como leída"""
    try:
        success = alert_service.mark_alert_read(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        return {
            "success": True,
            "message": "Alerta marcada como leída"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking alert as read: {e}")
        raise HTTPException(status_code=500, detail=f"Error al marcar alerta: {str(e)}")


@router.delete("/alerts/{alert_id}", response_model=dict)
async def delete_alert(alert_id: str):
    """Elimina una alerta"""
    try:
        success = alert_service.delete_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alerta no encontrada")
        return {
            "success": True,
            "message": "Alerta eliminada"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar alerta: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.3.0 ==========

@router.get("/temporal/structure", response_model=dict)
async def get_temporal_structure(track_id: str = Query(...)):
    """Analiza la estructura temporal de una canción"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        structure = temporal_analyzer.analyze_temporal_structure(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "temporal_structure": structure
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing temporal structure: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar estructura temporal: {str(e)}")


@router.get("/temporal/energy", response_model=dict)
async def get_energy_progression(track_id: str = Query(...)):
    """Analiza la progresión de energía a lo largo del tiempo"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        progression = temporal_analyzer.analyze_energy_progression(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "energy_progression": progression
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing energy progression: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar progresión de energía: {str(e)}")


@router.get("/temporal/tempo", response_model=dict)
async def get_tempo_changes(track_id: str = Query(...)):
    """Analiza cambios de tempo a lo largo del tiempo"""
    try:
        audio_analysis = spotify_service.get_track_audio_analysis(track_id)
        changes = temporal_analyzer.analyze_tempo_changes(audio_analysis)
        return {
            "success": True,
            "track_id": track_id,
            "tempo_changes": changes
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing tempo changes: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar cambios de tempo: {str(e)}")


@router.get("/quality/analyze", response_model=dict)
async def analyze_quality(track_id: str = Query(...)):
    """Analiza la calidad de producción de un track"""
    try:
        quality = quality_analyzer.analyze_production_quality(track_id)
        return {
            "success": True,
            "quality_analysis": quality
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing quality: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar calidad: {str(e)}")


@router.post("/recommendations/contextual", response_model=dict)
async def recommend_contextual(
    track_id: str = Query(...),
    context: Optional[Dict[str, Any]] = None,
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en contexto personalizado"""
    try:
        if context is None:
            context = {}
        
        recommendations = contextual_recommender.recommend_by_context(track_id, context, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in contextual recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación contextual: {str(e)}")


@router.post("/recommendations/time-of-day", response_model=dict)
async def recommend_by_time(
    track_id: str = Query(...),
    time_of_day: str = Query(..., regex="^(morning|afternoon|evening|night)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en hora del día"""
    try:
        recommendations = contextual_recommender.recommend_by_time_of_day(track_id, time_of_day, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in time-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por hora: {str(e)}")


@router.post("/recommendations/activity", response_model=dict)
async def recommend_by_activity(
    track_id: str = Query(...),
    activity: str = Query(..., regex="^(workout|study|party|relax|drive)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en actividad"""
    try:
        recommendations = contextual_recommender.recommend_by_activity(track_id, activity, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in activity-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por actividad: {str(e)}")


@router.post("/recommendations/mood", response_model=dict)
async def recommend_by_mood(
    track_id: str = Query(...),
    target_mood: str = Query(..., regex="^(happy|sad|energetic|calm|romantic)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Recomendaciones basadas en mood objetivo"""
    try:
        recommendations = contextual_recommender.recommend_by_mood(track_id, target_mood, limit)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in mood-based recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendación por mood: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.4.0 ==========

@router.post("/playlists/analyze", response_model=dict)
async def analyze_playlist(track_ids: List[str]):
    """Analiza una playlist completa"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        analysis = playlist_analyzer.analyze_playlist(track_ids)
        return {
            "success": True,
            "analysis": analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar playlist: {str(e)}")


@router.post("/playlists/suggest-improvements", response_model=dict)
async def suggest_playlist_improvements(track_ids: List[str]):
    """Sugiere mejoras para una playlist"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        suggestions = playlist_analyzer.suggest_playlist_improvements(track_ids)
        return {
            "success": True,
            "suggestions": suggestions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting improvements: {e}")
        raise HTTPException(status_code=500, detail=f"Error al sugerir mejoras: {str(e)}")


@router.post("/playlists/optimize-order", response_model=dict)
async def optimize_playlist_order(track_ids: List[str]):
    """Optimiza el orden de una playlist"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        optimization = playlist_analyzer.optimize_playlist_order(track_ids)
        return {
            "success": True,
            "optimization": optimization
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing playlist order: {e}")
        raise HTTPException(status_code=500, detail=f"Error al optimizar orden: {str(e)}")


@router.post("/artists/compare", response_model=dict)
async def compare_artists(
    artist_names: List[str],
    limit_per_artist: int = Query(10, ge=5, le=20)
):
    """Compara múltiples artistas"""
    try:
        if len(artist_names) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 artistas")
        
        if len(artist_names) > 5:
            raise HTTPException(status_code=400, detail="Máximo 5 artistas")
        
        comparison = artist_comparator.compare_artists(artist_names, limit_per_artist)
        return {
            "success": True,
            "comparison": comparison
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing artists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar artistas: {str(e)}")


@router.get("/artists/evolution", response_model=dict)
async def analyze_artist_evolution(
    artist_name: str = Query(...),
    limit: int = Query(20, ge=5, le=50)
):
    """Analiza la evolución musical de un artista"""
    try:
        evolution = artist_comparator.analyze_artist_evolution(artist_name, limit)
        return {
            "success": True,
            "evolution": evolution
        }
    except Exception as e:
        logger.error(f"Error analyzing artist evolution: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar evolución: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.5.0 ==========

@router.get("/discovery/similar-artists", response_model=dict)
async def discover_similar_artists(
    artist_name: str = Query(...),
    limit: int = Query(10, ge=1, le=20)
):
    """Descubre artistas similares a uno dado"""
    try:
        discovery = discovery_service.discover_similar_artists(artist_name, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering similar artists: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir artistas: {str(e)}")


@router.get("/discovery/underground", response_model=dict)
async def discover_underground_tracks(
    genre: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50)
):
    """Descubre tracks underground (baja popularidad pero alta calidad)"""
    try:
        discovery = discovery_service.discover_underground_tracks(genre, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering underground tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir tracks underground: {str(e)}")


@router.get("/discovery/mood-transition", response_model=dict)
async def discover_mood_transition(
    start_mood: str = Query(..., regex="^(sad|happy|energetic|calm|romantic|angry)$"),
    target_mood: str = Query(..., regex="^(sad|happy|energetic|calm|romantic|angry)$"),
    limit: int = Query(10, ge=1, le=20)
):
    """Descubre tracks que transicionan entre moods"""
    try:
        discovery = discovery_service.discover_by_mood_transition(start_mood, target_mood, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering mood transitions: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir transiciones: {str(e)}")


@router.get("/discovery/fresh", response_model=dict)
async def discover_fresh_tracks(
    genre: Optional[str] = Query(None),
    days_old: int = Query(30, ge=1, le=365),
    limit: int = Query(20, ge=1, le=50)
):
    """Descubre tracks frescos (recientes)"""
    try:
        discovery = discovery_service.discover_fresh_tracks(genre, days_old, limit)
        return {
            "success": True,
            "discovery": discovery
        }
    except Exception as e:
        logger.error(f"Error discovering fresh tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al descubrir tracks frescos: {str(e)}")


@router.get("/covers/analyze", response_model=dict)
async def analyze_cover(
    original_track_id: str = Query(...),
    cover_track_id: str = Query(...)
):
    """Analiza un cover comparándolo con el original"""
    try:
        analysis = cover_remix_analyzer.analyze_cover(original_track_id, cover_track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing cover: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar cover: {str(e)}")


@router.get("/remixes/analyze", response_model=dict)
async def analyze_remix(
    original_track_id: str = Query(...),
    remix_track_id: str = Query(...)
):
    """Analiza un remix comparándolo con el original"""
    try:
        analysis = cover_remix_analyzer.analyze_remix(original_track_id, remix_track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing remix: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar remix: {str(e)}")


@router.get("/covers/find", response_model=dict)
async def find_covers_and_remixes(
    track_id: str = Query(...),
    limit: int = Query(20, ge=1, le=50)
):
    """Encuentra covers y remixes de un track"""
    try:
        results = cover_remix_analyzer.find_covers_and_remixes(track_id, limit)
        return {
            "success": True,
            "results": results
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error finding covers and remixes: {e}")
        raise HTTPException(status_code=500, detail=f"Error al buscar covers y remixes: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.6.0 ==========

@router.get("/trends/predict/genre", response_model=dict)
async def predict_genre_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de géneros musicales"""
    try:
        prediction = trend_predictor.predict_genre_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting genre trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/emotion", response_model=dict)
async def predict_emotion_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de emociones musicales"""
    try:
        prediction = trend_predictor.predict_emotion_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting emotion trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/features", response_model=dict)
async def predict_feature_trends(
    time_horizon_days: int = Query(30, ge=7, le=365)
):
    """Predice tendencias de características musicales"""
    try:
        prediction = trend_predictor.predict_feature_trends(time_horizon_days)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting feature trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/instrumentation/analyze", response_model=dict)
async def analyze_instrumentation(track_id: str = Query(...)):
    """Analiza la instrumentación de un track"""
    try:
        analysis = instrumentation_analyzer.analyze_instrumentation(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing instrumentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar instrumentación: {str(e)}")


@router.post("/export/csv", response_model=dict)
async def export_to_csv(analyses: List[Dict[str, Any]]):
    """Exporta múltiples análisis a CSV"""
    try:
        csv_data = export_service.export_to_csv(analyses)
        return {
            "success": True,
            "format": "csv",
            "data": csv_data
        }
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar a CSV: {str(e)}")


@router.get("/export/comprehensive/{track_id}", response_model=dict)
async def export_comprehensive_report(track_id: str):
    """Exporta un reporte comprehensivo en Markdown"""
    try:
        # Obtener análisis completo
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        report = export_service.export_comprehensive_report(analysis)
        
        return {
            "success": True,
            "format": "markdown",
            "report": report
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar reporte: {str(e)}")

# ========== NUEVOS ENDPOINTS v2.6.0 ==========
# Agregar estos endpoints al final de music_api.py

@router.get("/instrumentation/analyze", response_model=dict)
async def analyze_instrumentation(track_id: str = Query(...)):
    """Analiza la instrumentación de un track"""
    try:
        analysis = instrumentation_analyzer.analyze_instrumentation(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing instrumentation: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar instrumentación: {str(e)}")


@router.get("/trends/predict/genres", response_model=dict)
async def predict_genre_trends(
    time_horizon: str = Query("6months", regex="^(3months|6months|1year)$")
):
    """Predice tendencias de géneros musicales"""
    try:
        prediction = trend_predictor.predict_genre_trends(time_horizon)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting genre trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/emotions", response_model=dict)
async def predict_emotion_trends(
    time_horizon: str = Query("6months", regex="^(3months|6months|1year)$")
):
    """Predice tendencias de emociones musicales"""
    try:
        prediction = trend_predictor.predict_emotion_trends(time_horizon)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting emotion trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/features", response_model=dict)
async def predict_feature_trends(
    time_horizon: str = Query("6months", regex="^(3months|6months|1year)$")
):
    """Predice tendencias de características musicales"""
    try:
        prediction = trend_predictor.predict_feature_trends(time_horizon)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting feature trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir tendencias: {str(e)}")


@router.get("/trends/predict/next-big-thing", response_model=dict)
async def predict_next_big_thing(genre: Optional[str] = Query(None)):
    """Predice el próximo gran éxito musical"""
    try:
        prediction = trend_predictor.predict_next_big_thing(genre)
        return {
            "success": True,
            "prediction": prediction
        }
    except Exception as e:
        logger.error(f"Error predicting next big thing: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir próximo éxito: {str(e)}")



# ========== NUEVOS ENDPOINTS v2.7.0 ==========

@router.post("/lyrics/analyze", response_model=dict)
async def analyze_lyrics(
    lyrics: str = Query(...),
    track_name: Optional[str] = Query(None)
):
    """Analiza letras de una canción"""
    try:
        analysis = lyrics_analyzer.analyze_lyrics(lyrics, track_name)
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing lyrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar letras: {str(e)}")


@router.get("/melodic/patterns/{track_id}", response_model=dict)
async def analyze_melodic_patterns(track_id: str):
    """Analiza patrones melódicos de un track"""
    try:
        analysis = melodic_pattern_analyzer.analyze_melodic_patterns(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing melodic patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar patrones melódicos: {str(e)}")


@router.get("/dynamics/analyze/{track_id}", response_model=dict)
async def analyze_dynamics(track_id: str):
    """Analiza la dinámica musical de un track"""
    try:
        analysis = dynamics_analyzer.analyze_dynamics(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing dynamics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar dinámica: {str(e)}")


@router.get("/market/position/{track_id}", response_model=dict)
async def analyze_market_position(track_id: str):
    """Analiza la posición de mercado de un track"""
    try:
        analysis = market_analyzer.analyze_market_position(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing market position: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar posición de mercado: {str(e)}")


@router.get("/market/competitors", response_model=dict)
async def analyze_competitor_landscape(
    genre: str = Query(...),
    limit: int = Query(20, ge=5, le=50)
):
    """Analiza el panorama competitivo de un género"""
    try:
        analysis = market_analyzer.analyze_competitor_landscape(genre, limit)
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing competitor landscape: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar competencia: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.8.0 ==========

@router.post("/audio/analyze-file", response_model=dict)
async def analyze_audio_file(file_path: str = Query(...)):
    """Analiza un archivo de audio local"""
    try:
        analysis = audio_file_analyzer.analyze_audio_file(file_path)
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar archivo: {str(e)}")


@router.get("/audio/validate", response_model=dict)
async def validate_audio_file(file_path: str = Query(...)):
    """Valida un archivo de audio"""
    try:
        validation = audio_file_analyzer.validate_audio_file(file_path)
        return {
            "success": True,
            "validation": validation
        }
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Error al validar archivo: {str(e)}")


@router.get("/audio/supported-formats", response_model=dict)
async def get_supported_formats():
    """Obtiene formatos de audio soportados"""
    try:
        formats = audio_file_analyzer.get_supported_formats()
        return {
            "success": True,
            "formats": formats
        }
    except Exception as e:
        logger.error(f"Error getting supported formats: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener formatos: {str(e)}")


@router.post("/benchmark/track", response_model=dict)
async def benchmark_track(
    track_id: str = Query(...),
    reference_tracks: List[str] = Query(...)
):
    """Compara un track con tracks de referencia"""
    try:
        benchmark = benchmark_service.benchmark_track(track_id, reference_tracks)
        return {
            "success": True,
            "benchmark": benchmark
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error benchmarking track: {e}")
        raise HTTPException(status_code=500, detail=f"Error al hacer benchmark: {str(e)}")


@router.get("/benchmark/create-set", response_model=dict)
async def create_benchmark_set(
    genre: str = Query(...),
    limit: int = Query(20, ge=5, le=50)
):
    """Crea un conjunto de referencia para benchmarking"""
    try:
        benchmark_set = benchmark_service.create_benchmark_set(genre, limit)
        return {
            "success": True,
            "benchmark_set": benchmark_set
        }
    except Exception as e:
        logger.error(f"Error creating benchmark set: {e}")
        raise HTTPException(status_code=500, detail=f"Error al crear benchmark set: {str(e)}")


@router.get("/harmony/advanced/{track_id}", response_model=dict)
async def analyze_advanced_harmony(track_id: str):
    """Análisis armónico avanzado"""
    try:
        analysis = advanced_harmonic_analyzer.analyze_advanced_harmony(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing advanced harmony: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar armonía: {str(e)}")


@router.get("/performance/metrics", response_model=dict)
async def get_performance_metrics():
    """Obtiene métricas de rendimiento del sistema"""
    try:
        metrics = performance_metrics.get_metrics()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener métricas: {str(e)}")


@router.get("/performance/endpoints", response_model=dict)
async def get_endpoint_metrics(limit: int = Query(10, ge=1, le=50)):
    """Obtiene métricas por endpoint"""
    try:
        metrics = performance_metrics.get_endpoint_metrics(limit)
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting endpoint metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener métricas: {str(e)}")


@router.get("/performance/summary", response_model=dict)
async def get_performance_summary():
    """Obtiene resumen de rendimiento"""
    try:
        summary = performance_metrics.get_performance_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener resumen: {str(e)}")


@router.post("/performance/reset", response_model=dict)
async def reset_performance_metrics():
    """Resetea las métricas de rendimiento"""
    try:
        performance_metrics.reset_metrics()
        return {
            "success": True,
            "message": "Métricas reseteadas correctamente"
        }
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al resetear métricas: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.9.0 ==========

@router.get("/structure/advanced/{track_id}", response_model=dict)
async def analyze_advanced_structure(track_id: str):
    """Análisis avanzado de estructura de canción"""
    try:
        analysis = advanced_structure_analyzer.analyze_advanced_structure(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing advanced structure: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar estructura: {str(e)}")


@router.get("/recommendations/enhanced/{track_id}", response_model=dict)
async def get_enhanced_recommendations(
    track_id: str,
    limit: int = Query(20, ge=1, le=50),
    include_factors: bool = Query(True)
):
    """Obtiene recomendaciones mejoradas con ML"""
    try:
        recommendations = enhanced_recommender.get_enhanced_recommendations(track_id, limit, include_factors)
        return {
            "success": True,
            "recommendations": recommendations
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting enhanced recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener recomendaciones: {str(e)}")


@router.post("/recommendations/contextual-playlist", response_model=dict)
async def generate_contextual_playlist(context: Dict[str, Any], length: int = Query(20, ge=5, le=50)):
    """Genera playlist contextual mejorada"""
    try:
        playlist = enhanced_recommender.get_contextual_playlist(context, length)
        return {
            "success": True,
            "playlist": playlist
        }
    except Exception as e:
        logger.error(f"Error generating contextual playlist: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar playlist: {str(e)}")


@router.get("/predict/success/{track_id}", response_model=dict)
async def predict_success(track_id: str):
    """Predice el éxito comercial de un track"""
    try:
        prediction = success_predictor.predict_success(track_id)
        return {
            "success": True,
            "prediction": prediction
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting success: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir éxito: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.10.0 ==========

@router.get("/collaborations/advanced/{track_id}", response_model=dict)
async def analyze_advanced_collaborations(track_id: str):
    """Análisis avanzado de colaboraciones"""
    try:
        analysis = advanced_collaboration_analyzer.analyze_advanced_collaborations(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing advanced collaborations: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar colaboraciones: {str(e)}")


@router.post("/collaborations/network", response_model=dict)
async def analyze_collaboration_network(
    artist_names: List[str] = Query(...),
    depth: int = Query(2, ge=1, le=3)
):
    """Analiza red de colaboraciones"""
    try:
        network = advanced_collaboration_analyzer.analyze_collaboration_network(artist_names, depth)
        return {
            "success": True,
            "network": network
        }
    except Exception as e:
        logger.error(f"Error analyzing collaboration network: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar red: {str(e)}")


@router.get("/rhythm/advanced/{track_id}", response_model=dict)
async def analyze_advanced_rhythm(track_id: str):
    """Análisis rítmico avanzado"""
    try:
        analysis = advanced_rhythmic_analyzer.analyze_advanced_rhythm(track_id)
        return {
            "success": True,
            "analysis": analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing advanced rhythm: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar ritmo: {str(e)}")


@router.get("/visualization/track/{track_id}", response_model=dict)
async def generate_track_visualization(track_id: str):
    """Genera datos para visualización de un track"""
    try:
        visualization = data_visualization.generate_track_visualization_data(track_id)
        return {
            "success": True,
            "visualization": visualization
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar visualización: {str(e)}")


@router.post("/visualization/compare", response_model=dict)
async def generate_comparison_visualization(track_ids: List[str] = Query(...)):
    """Genera datos para visualización comparativa"""
    try:
        visualization = data_visualization.generate_comparison_visualization(track_ids)
        return {
            "success": True,
            "visualization": visualization
        }
    except Exception as e:
        logger.error(f"Error generating comparison visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar visualización: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.11.0 ==========

@router.post("/lyrics/sentiment-detailed", response_model=dict)
async def analyze_sentiment_detailed(lyrics: str = Query(...)):
    """Análisis detallado de sentimiento en letras"""
    try:
        analysis = lyrics_analyzer.analyze_sentiment_detailed(lyrics)
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing detailed sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar sentimiento: {str(e)}")


@router.get("/reports/comprehensive/{track_id}", response_model=dict)
async def generate_comprehensive_report(
    track_id: str,
    include_all: bool = Query(True)
):
    """Genera reporte comprehensivo de un track"""
    try:
        report = advanced_report_generator.generate_comprehensive_report(track_id, include_all)
        return report
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar reporte: {str(e)}")


@router.post("/reports/compare", response_model=dict)
async def generate_comparison_report(track_ids: List[str] = Query(...)):
    """Genera reporte comparativo de múltiples tracks"""
    try:
        report = advanced_report_generator.generate_comparison_report(track_ids)
        return report
    except Exception as e:
        logger.error(f"Error generating comparison report: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar reporte: {str(e)}")


@router.post("/realtime/analyze", response_model=dict)
async def analyze_realtime_stream(audio_data: Dict[str, Any]):
    """Analiza stream de audio en tiempo real"""
    try:
        analysis = realtime_audio_analyzer.analyze_realtime_stream(audio_data)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing realtime stream: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar stream: {str(e)}")


@router.get("/realtime/history", response_model=dict)
async def get_realtime_history(limit: int = Query(50, ge=1, le=100)):
    """Obtiene historial de análisis en tiempo real"""
    try:
        history = realtime_audio_analyzer.get_analysis_history(limit)
        return {
            "success": True,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting realtime history: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener historial: {str(e)}")


@router.post("/realtime/clear", response_model=dict)
async def clear_realtime_buffer():
    """Limpia el buffer de análisis en tiempo real"""
    try:
        result = realtime_audio_analyzer.clear_buffer()
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error clearing buffer: {e}")
        raise HTTPException(status_code=500, detail=f"Error al limpiar buffer: {str(e)}")


# ========== NUEVOS ENDPOINTS v2.12.0 - DEEP LEARNING ==========

@router.post("/ml/deep-learning/initialize", response_model=dict)
async def initialize_deep_learning_model(
    feature_dim: int = Query(13, ge=1, le=50),
    d_model: int = Query(256, ge=64, le=1024),
    num_layers: int = Query(4, ge=1, le=12)
):
    """Inicializa modelo de deep learning"""
    try:
        result = deep_learning_service.initialize_model(feature_dim, d_model, num_layers)
        return result
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al inicializar modelo: {str(e)}")


@router.get("/ml/deep-learning/predict/{track_id}", response_model=dict)
async def predict_with_deep_learning(track_id: str):
    """Predice usando modelo de deep learning"""
    try:
        prediction = deep_learning_service.predict_with_model(track_id)
        return {
            "success": True,
            "prediction": prediction
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting with model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@router.post("/ml/deep-learning/batch-predict", response_model=dict)
async def batch_predict_deep_learning(track_ids: List[str] = Query(...)):
    """Predicción en batch usando deep learning"""
    try:
        if len(track_ids) > 50:
            raise HTTPException(status_code=400, detail="Máximo 50 tracks por batch")
        
        result = deep_learning_service.batch_predict(track_ids)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")


@router.get("/ml/deep-learning/model-info", response_model=dict)
async def get_deep_learning_model_info():
    """Obtiene información del modelo de deep learning"""
    try:
        info = deep_learning_service.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener info: {str(e)}")


@router.post("/ml/deep-learning/train", response_model=dict)
async def train_deep_learning_model(
    track_ids: List[str] = Query(...),
    genres: Optional[List[int]] = Query(None),
    emotions: Optional[List[int]] = Query(None),
    popularities: Optional[List[float]] = Query(None),
    epochs: int = Query(10, ge=1, le=100),
    batch_size: int = Query(32, ge=8, le=128),
    learning_rate: float = Query(1e-4, ge=1e-6, le=1e-2)
):
    """Entrena el modelo de deep learning con tracks"""
    try:
        if len(track_ids) < batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos {batch_size} tracks para entrenar"
            )
        
        result = deep_learning_service.train_model(
            track_ids=track_ids,
            genres=genres,
            emotions=emotions,
            popularities=popularities,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelo: {str(e)}")


@router.post("/ml/deep-learning/lyrics/analyze", response_model=dict)
async def analyze_lyrics_with_transformer(lyrics: str = Query(...)):
    """Analiza letras usando modelo Transformer"""
    try:
        if not lyrics or len(lyrics.strip()) == 0:
            raise HTTPException(status_code=400, detail="Las letras no pueden estar vacías")
        
        result = deep_learning_service.analyze_lyrics_with_transformer(lyrics)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "success": True,
            "analysis": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing lyrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar letras: {str(e)}")


@router.post("/ml/deep-learning/save", response_model=dict)
async def save_deep_learning_model(path: str = Query(...)):
    """Guarda el modelo entrenado"""
    try:
        result = deep_learning_service.save_model(path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar modelo: {str(e)}")


@router.post("/ml/deep-learning/load", response_model=dict)
async def load_deep_learning_model(path: str = Query(...)):
    """Carga un modelo pre-entrenado"""
    try:
        result = deep_learning_service.load_model(path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {str(e)}")


@router.post("/ml/deep-learning/evaluate-advanced", response_model=dict)
async def evaluate_model_advanced(
    track_ids: List[str] = Query(...),
    genres: Optional[List[int]] = Query(None),
    emotions: Optional[List[int]] = Query(None),
    popularities: Optional[List[float]] = Query(None)
):
    """Evaluación avanzada con métricas detalladas"""
    try:
        result = deep_learning_service.evaluate_model_advanced(
            track_ids=track_ids,
            genres=genres,
            emotions=emotions,
            popularities=popularities
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en evaluación: {str(e)}")


@router.post("/ml/deep-learning/train-with-validation", response_model=dict)
async def train_with_validation(
    train_track_ids: List[str] = Query(...),
    val_track_ids: List[str] = Query(...),
    train_genres: Optional[List[int]] = Query(None),
    train_emotions: Optional[List[int]] = Query(None),
    train_popularities: Optional[List[float]] = Query(None),
    val_genres: Optional[List[int]] = Query(None),
    val_emotions: Optional[List[int]] = Query(None),
    val_popularities: Optional[List[float]] = Query(None),
    epochs: int = Query(10, ge=1, le=100),
    batch_size: int = Query(32, ge=8, le=128),
    learning_rate: float = Query(1e-4, ge=1e-6, le=1e-2),
    early_stopping_patience: int = Query(5, ge=1, le=20),
    min_delta: float = Query(0.001, ge=0.0, le=0.1)
):
    """Entrenamiento con validación y early stopping"""
    try:
        if len(train_track_ids) < batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Se necesitan al menos {batch_size} tracks para entrenar"
            )
        
        result = deep_learning_service.train_with_validation(
            train_track_ids=train_track_ids,
            val_track_ids=val_track_ids,
            train_genres=train_genres,
            train_emotions=train_emotions,
            train_popularities=train_popularities,
            val_genres=val_genres,
            val_emotions=val_emotions,
            val_popularities=val_popularities,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training with validation: {e}")
        raise HTTPException(status_code=500, detail=f"Error al entrenar: {str(e)}")


@router.post("/ml/deep-learning/extract-embeddings", response_model=dict)
async def extract_embeddings(track_ids: List[str] = Query(...)):
    """Extrae embeddings musicales del modelo"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks por request")
        
        result = deep_learning_service.extract_embeddings(track_ids)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error al extraer embeddings: {str(e)}")


@router.post("/ml/deep-learning/save-training-history", response_model=dict)
async def save_training_history(path: str = Query(...)):
    """Guarda el historial de entrenamiento"""
    try:
        result = deep_learning_service.save_training_history(path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving training history: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar historial: {str(e)}")


@router.post("/ml/deep-learning/experiment/initialize", response_model=dict)
async def initialize_experiment_tracking(
    experiment_name: str = Query(...),
    use_wandb: bool = Query(False),
    use_tensorboard: bool = Query(False),
    project_name: str = Query("music-analyzer-ai")
):
    """Inicializa tracking de experimentos (wandb/tensorboard)"""
    try:
        result = deep_learning_service.initialize_experiment_tracking(
            experiment_name=experiment_name,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            project_name=project_name
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing experiment tracking: {e}")
        raise HTTPException(status_code=500, detail=f"Error al inicializar tracking: {str(e)}")


@router.post("/ml/deep-learning/find-similar", response_model=dict)
async def find_similar_tracks(
    reference_track_id: str = Query(...),
    candidate_track_ids: List[str] = Query(...),
    top_k: int = Query(10, ge=1, le=50),
    metric: str = Query("cosine", regex="^(cosine|euclidean)$")
):
    """Encuentra tracks similares usando embeddings"""
    try:
        if len(candidate_track_ids) > 200:
            raise HTTPException(status_code=400, detail="Máximo 200 candidatos")
        
        result = deep_learning_service.find_similar_tracks(
            reference_track_id=reference_track_id,
            candidate_track_ids=candidate_track_ids,
            top_k=top_k,
            metric=metric
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al encontrar similares: {str(e)}")


@router.post("/ml/deep-learning/recommend-embeddings", response_model=dict)
async def recommend_based_on_embeddings(
    seed_track_ids: List[str] = Query(...),
    candidate_track_ids: List[str] = Query(...),
    top_k: int = Query(20, ge=1, le=100),
    diversity_weight: float = Query(0.3, ge=0.0, le=1.0)
):
    """Recomendaciones basadas en embeddings con diversidad"""
    try:
        if len(seed_track_ids) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 seed tracks")
        if len(candidate_track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 candidatos")
        
        result = deep_learning_service.recommend_based_on_embeddings(
            seed_track_ids=seed_track_ids,
            candidate_track_ids=candidate_track_ids,
            top_k=top_k,
            diversity_weight=diversity_weight
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embedding recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error en recomendaciones: {str(e)}")


@router.post("/ml/deep-learning/optimize-hyperparameters", response_model=dict)
async def optimize_hyperparameters(
    train_track_ids: List[str] = Query(...),
    val_track_ids: List[str] = Query(...),
    train_genres: Optional[List[int]] = Query(None),
    train_emotions: Optional[List[int]] = Query(None),
    train_popularities: Optional[List[float]] = Query(None),
    val_genres: Optional[List[int]] = Query(None),
    val_emotions: Optional[List[int]] = Query(None),
    val_popularities: Optional[List[float]] = Query(None),
    max_trials: int = Query(10, ge=1, le=50)
):
    """Optimización de hiperparámetros"""
    try:
        if len(train_track_ids) < 32:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 32 tracks para entrenar")
        
        result = deep_learning_service.optimize_hyperparameters(
            train_track_ids=train_track_ids,
            val_track_ids=val_track_ids,
            train_genres=train_genres,
            train_emotions=train_emotions,
            train_popularities=train_popularities,
            val_genres=val_genres,
            val_emotions=val_emotions,
            val_popularities=val_popularities,
            max_trials=max_trials
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing hyperparameters: {e}")
        raise HTTPException(status_code=500, detail=f"Error en optimización: {str(e)}")


@router.post("/ml/deep-learning/cluster-tracks", response_model=dict)
async def cluster_tracks(
    track_ids: List[str] = Query(...),
    n_clusters: int = Query(5, ge=2, le=20),
    method: str = Query("kmeans", regex="^(kmeans|dbscan)$"),
    use_pca: bool = Query(False),
    pca_components: int = Query(2, ge=2, le=10)
):
    """Agrupa tracks usando embeddings con clustering"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.cluster_tracks(
            track_ids=track_ids,
            n_clusters=n_clusters,
            method=method,
            use_pca=use_pca,
            pca_components=pca_components
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clustering tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error en clustering: {str(e)}")


@router.post("/ml/deep-learning/analyze-feature-importance", response_model=dict)
async def analyze_feature_importance(
    track_ids: List[str] = Query(...),
    target_values: List[float] = Query(...),
    feature_names: Optional[List[str]] = Query(None)
):
    """Analiza importancia de características usando correlación"""
    try:
        if len(track_ids) != len(target_values):
            raise HTTPException(status_code=400, detail="track_ids y target_values deben tener la misma longitud")
        
        result = deep_learning_service.analyze_feature_importance(
            track_ids=track_ids,
            target_values=target_values,
            feature_names=feature_names
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/ml/deep-learning/compare-models", response_model=dict)
async def compare_models(
    model_paths: List[str] = Query(...),
    test_track_ids: List[str] = Query(...),
    test_genres: Optional[List[int]] = Query(None),
    test_emotions: Optional[List[int]] = Query(None),
    test_popularities: Optional[List[float]] = Query(None)
):
    """Compara múltiples modelos en el mismo conjunto de prueba"""
    try:
        if len(model_paths) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 modelos")
        if len(model_paths) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 modelos")
        
        result = deep_learning_service.compare_models(
            model_paths=model_paths,
            test_track_ids=test_track_ids,
            test_genres=test_genres,
            test_emotions=test_emotions,
            test_popularities=test_popularities
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error en comparación: {str(e)}")


@router.post("/ml/deep-learning/export-results", response_model=dict)
async def export_training_results(
    output_path: str = Query(...),
    include_embeddings: bool = Query(False),
    track_ids: Optional[List[str]] = Query(None)
):
    """Exporta resultados de entrenamiento en formato JSON"""
    try:
        if include_embeddings and not track_ids:
            raise HTTPException(status_code=400, detail="track_ids requerido cuando include_embeddings=True")
        
        result = deep_learning_service.export_training_results(
            output_path=output_path,
            include_embeddings=include_embeddings,
            track_ids=track_ids
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise HTTPException(status_code=500, detail=f"Error al exportar: {str(e)}")


@router.post("/ml/deep-learning/analyze-embedding-trends", response_model=dict)
async def analyze_embedding_trends(
    track_ids_by_time: Dict[str, List[str]] = Body(...)
):
    """Analiza tendencias temporales en embeddings"""
    try:
        result = deep_learning_service.analyze_embedding_trends(track_ids_by_time)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing embedding trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis de tendencias: {str(e)}")


@router.post("/ml/deep-learning/analyze-bias-fairness", response_model=dict)
async def analyze_bias_fairness(
    track_ids: List[str] = Query(...),
    genres: List[int] = Query(...)
):
    """Analiza bias y fairness en predicciones del modelo"""
    try:
        if len(track_ids) != len(genres):
            raise HTTPException(status_code=400, detail="track_ids y genres deben tener la misma longitud")
        
        result = deep_learning_service.analyze_bias_fairness(
            track_ids=track_ids,
            genres=genres
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing bias and fairness: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.get("/ml/deep-learning/training-report", response_model=dict)
async def generate_training_report(
    include_visualizations: bool = Query(False)
):
    """Genera un reporte completo del entrenamiento"""
    try:
        result = deep_learning_service.generate_training_report(
            include_visualizations=include_visualizations
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating training report: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar reporte: {str(e)}")


@router.post("/ml/deep-learning/fine-tune", response_model=dict)
async def fine_tune_model(
    base_model_path: str = Query(...),
    fine_tune_track_ids: List[str] = Query(...),
    fine_tune_genres: Optional[List[int]] = Query(None),
    fine_tune_emotions: Optional[List[int]] = Query(None),
    fine_tune_popularities: Optional[List[float]] = Query(None),
    epochs: int = Query(5, ge=1, le=50),
    learning_rate: float = Query(1e-5, ge=1e-7, le=1e-3),
    freeze_encoder: bool = Query(False)
):
    """Fine-tuning de un modelo pre-entrenado"""
    try:
        if len(fine_tune_track_ids) < 10:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 10 tracks para fine-tuning")
        
        result = deep_learning_service.fine_tune_model(
            base_model_path=base_model_path,
            fine_tune_track_ids=fine_tune_track_ids,
            fine_tune_genres=fine_tune_genres,
            fine_tune_emotions=fine_tune_emotions,
            fine_tune_popularities=fine_tune_popularities,
            epochs=epochs,
            learning_rate=learning_rate,
            freeze_encoder=freeze_encoder
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fine-tuning model: {e}")
        raise HTTPException(status_code=500, detail=f"Error en fine-tuning: {str(e)}")


@router.get("/ml/deep-learning/explain/{track_id}", response_model=dict)
async def explain_prediction(
    track_id: str,
    method: str = Query("gradient", regex="^(gradient)$")
):
    """Explica una predicción usando interpretabilidad"""
    try:
        result = deep_learning_service.explain_prediction(
            track_id=track_id,
            method=method
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error en explicación: {str(e)}")


@router.post("/ml/deep-learning/ab-test", response_model=dict)
async def ab_test_models(
    model_a_path: str = Query(...),
    model_b_path: str = Query(...),
    test_track_ids: List[str] = Query(...),
    test_genres: Optional[List[int]] = Query(None),
    test_emotions: Optional[List[int]] = Query(None),
    test_popularities: Optional[List[float]] = Query(None),
    metric: str = Query("accuracy", regex="^(accuracy|f1|rmse)$")
):
    """A/B testing entre dos modelos"""
    try:
        result = deep_learning_service.ab_test_models(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            test_track_ids=test_track_ids,
            test_genres=test_genres,
            test_emotions=test_emotions,
            test_popularities=test_popularities,
            metric=metric
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in A/B testing: {e}")
        raise HTTPException(status_code=500, detail=f"Error en A/B testing: {str(e)}")


@router.post("/ml/deep-learning/analyze-robustness", response_model=dict)
async def analyze_robustness(
    track_ids: List[str] = Query(...),
    noise_level: float = Query(0.1, ge=0.0, le=1.0),
    num_perturbations: int = Query(10, ge=1, le=50)
):
    """Analiza robustez del modelo ante perturbaciones"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        result = deep_learning_service.analyze_robustness(
            track_ids=track_ids,
            noise_level=noise_level,
            num_perturbations=num_perturbations
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing robustness: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/ml/deep-learning/version", response_model=dict)
async def version_model(
    version_name: str = Query(...),
    description: Optional[str] = Query(None),
    metadata: Optional[Dict[str, Any]] = Body(None)
):
    """Versiona el modelo actual"""
    try:
        result = deep_learning_service.version_model(
            version_name=version_name,
            description=description,
            metadata=metadata
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error versioning model: {e}")
        raise HTTPException(status_code=500, detail=f"Error al versionar: {str(e)}")


@router.get("/ml/deep-learning/versions", response_model=dict)
async def list_model_versions():
    """Lista todas las versiones del modelo"""
    try:
        result = deep_learning_service.list_model_versions()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error al listar versiones: {str(e)}")


@router.get("/ml/deep-learning/production-metrics", response_model=dict)
async def get_production_metrics():
    """Obtiene métricas de producción del modelo"""
    try:
        result = deep_learning_service.get_production_metrics()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting production metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener métricas: {str(e)}")


@router.post("/ml/deep-learning/detect-drift", response_model=dict)
async def detect_data_drift(
    reference_track_ids: List[str] = Query(...),
    current_track_ids: List[str] = Query(...),
    threshold: float = Query(0.1, ge=0.0, le=1.0)
):
    """Detecta drift en los datos"""
    try:
        if len(reference_track_ids) > 500 or len(current_track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks por conjunto")
        
        result = deep_learning_service.detect_data_drift(
            reference_track_ids=reference_track_ids,
            current_track_ids=current_track_ids,
            threshold=threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")


@router.post("/ml/deep-learning/check-degradation", response_model=dict)
async def check_model_degradation(
    baseline_track_ids: List[str] = Query(...),
    baseline_genres: Optional[List[int]] = Query(None),
    baseline_emotions: Optional[List[int]] = Query(None),
    baseline_popularities: Optional[List[float]] = Query(None),
    current_track_ids: Optional[List[str]] = Query(None),
    degradation_threshold: float = Query(0.05, ge=0.0, le=1.0)
):
    """Verifica degradación del modelo comparando con baseline"""
    try:
        result = deep_learning_service.check_model_degradation(
            baseline_track_ids=baseline_track_ids,
            baseline_genres=baseline_genres,
            baseline_emotions=baseline_emotions,
            baseline_popularities=baseline_popularities,
            current_track_ids=current_track_ids,
            degradation_threshold=degradation_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking degradation: {e}")
        raise HTTPException(status_code=500, detail=f"Error en verificación: {str(e)}")


@router.post("/ml/deep-learning/auto-retrain", response_model=dict)
async def auto_retrain(
    trigger: str = Query("degradation", regex="^(degradation|drift|scheduled)$"),
    train_track_ids: List[str] = Query(...),
    val_track_ids: List[str] = Query(...),
    train_genres: Optional[List[int]] = Query(None),
    train_emotions: Optional[List[int]] = Query(None),
    train_popularities: Optional[List[float]] = Query(None),
    val_genres: Optional[List[int]] = Query(None),
    val_emotions: Optional[List[int]] = Query(None),
    val_popularities: Optional[List[float]] = Query(None),
    epochs: int = Query(5, ge=1, le=20),
    improvement_threshold: float = Query(0.01, ge=0.0, le=1.0)
):
    """Auto-retraining del modelo basado en triggers"""
    try:
        if len(train_track_ids) < 32:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 32 tracks para entrenar")
        
        result = deep_learning_service.auto_retrain(
            trigger=trigger,
            train_track_ids=train_track_ids,
            val_track_ids=val_track_ids,
            train_genres=train_genres,
            train_emotions=train_emotions,
            train_popularities=train_popularities,
            val_genres=val_genres,
            val_emotions=val_emotions,
            val_popularities=val_popularities,
            epochs=epochs,
            improvement_threshold=improvement_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in auto-retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Error en auto-retraining: {str(e)}")


@router.post("/ml/deep-learning/analyze-confidence", response_model=dict)
async def analyze_prediction_confidence(
    track_ids: List[str] = Query(...),
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0)
):
    """Analiza la confianza de las predicciones"""
    try:
        if len(track_ids) > 200:
            raise HTTPException(status_code=400, detail="Máximo 200 tracks")
        
        result = deep_learning_service.analyze_prediction_confidence(
            track_ids=track_ids,
            confidence_threshold=confidence_threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing confidence: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/ml/deep-learning/detect-outliers", response_model=dict)
async def detect_outliers(
    track_ids: List[str] = Query(...),
    method: str = Query("zscore", regex="^(zscore|isolation)$"),
    threshold: float = Query(3.0, ge=0.0)
):
    """Detecta outliers en los embeddings"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.detect_outliers(
            track_ids=track_ids,
            method=method,
            threshold=threshold
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")


@router.post("/ml/deep-learning/create-ensemble", response_model=dict)
async def create_ensemble(
    model_paths: List[str] = Query(...),
    weights: Optional[List[float]] = Query(None)
):
    """Crea un ensemble de modelos"""
    try:
        if len(model_paths) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 modelos")
        if len(model_paths) > 5:
            raise HTTPException(status_code=400, detail="Máximo 5 modelos")
        
        result = deep_learning_service.create_ensemble(
            model_paths=model_paths,
            weights=weights
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}")
        raise HTTPException(status_code=500, detail=f"Error en ensemble: {str(e)}")


@router.get("/ml/deep-learning/predict-ensemble/{track_id}", response_model=dict)
async def predict_with_ensemble(track_id: str):
    """Predice usando ensemble de modelos"""
    try:
        result = deep_learning_service.predict_with_ensemble(track_id=track_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting with ensemble: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@router.post("/ml/deep-learning/batch-advanced", response_model=dict)
async def batch_process_advanced(
    track_ids: List[str] = Query(...),
    batch_size: int = Query(32, ge=1, le=128),
    use_cache: bool = Query(True)
):
    """Procesamiento batch avanzado con caching"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.batch_process_advanced(
            track_ids=track_ids,
            batch_size=batch_size,
            use_cache=use_cache
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")


@router.post("/ml/deep-learning/clear-cache", response_model=dict)
async def clear_cache():
    """Limpia el cache de modelos y embeddings"""
    try:
        result = deep_learning_service.clear_cache()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error limpiando cache: {str(e)}")


@router.post("/ml/deep-learning/calibrate", response_model=dict)
async def calibrate_model(
    track_ids: List[str] = Query(...),
    true_genres: Optional[List[int]] = Query(None),
    true_emotions: Optional[List[int]] = Query(None),
    method: str = Query("isotonic", regex="^(isotonic|platt)$")
):
    """Calibra las probabilidades del modelo"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.calibrate_model(
            track_ids=track_ids,
            true_genres=true_genres,
            true_emotions=true_emotions,
            method=method
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calibrating model: {e}")
        raise HTTPException(status_code=500, detail=f"Error en calibración: {str(e)}")


@router.post("/ml/deep-learning/analyze-uncertainty", response_model=dict)
async def analyze_uncertainty(
    track_ids: List[str] = Query(...),
    num_samples: int = Query(10, ge=5, le=50)
):
    """Analiza la incertidumbre usando Monte Carlo Dropout"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        result = deep_learning_service.analyze_uncertainty(
            track_ids=track_ids,
            num_samples=num_samples
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing uncertainty: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/ml/deep-learning/active-learning", response_model=dict)
async def active_learning_query(
    track_ids: List[str] = Query(...),
    query_strategy: str = Query("uncertainty", regex="^(uncertainty|diversity)$"),
    num_samples: int = Query(10, ge=1, le=50)
):
    """Selecciona muestras para etiquetar usando active learning"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.active_learning_query(
            track_ids=track_ids,
            query_strategy=query_strategy,
            num_samples=num_samples
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in active learning: {e}")
        raise HTTPException(status_code=500, detail=f"Error en active learning: {str(e)}")


@router.post("/ml/deep-learning/transfer-learning", response_model=dict)
async def transfer_learning_analysis(
    source_track_ids: List[str] = Query(...),
    target_track_ids: List[str] = Query(...)
):
    """Analiza transfer learning entre dominios"""
    try:
        if len(source_track_ids) > 500 or len(target_track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks por dominio")
        
        result = deep_learning_service.transfer_learning_analysis(
            source_track_ids=source_track_ids,
            target_track_ids=target_track_ids
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in transfer learning analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.post("/ml/deep-learning/detect-adversarial/{track_id}", response_model=dict)
async def detect_adversarial_examples(
    track_id: str,
    epsilon: float = Query(0.01, ge=0.001, le=0.1),
    num_perturbations: int = Query(10, ge=5, le=50)
):
    """Detecta si un track es vulnerable a adversarial examples"""
    try:
        result = deep_learning_service.detect_adversarial_examples(
            track_id=track_id,
            epsilon=epsilon,
            num_perturbations=num_perturbations
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting adversarial examples: {e}")
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")


@router.post("/ml/deep-learning/meta-learning", response_model=dict)
async def meta_learning_adapt(
    support_track_ids: List[str] = Query(...),
    support_genres: Optional[List[int]] = Query(None),
    support_emotions: Optional[List[int]] = Query(None),
    query_track_ids: List[str] = Query(...),
    adaptation_steps: int = Query(5, ge=1, le=20)
):
    """Adapta el modelo usando meta-learning"""
    try:
        if len(support_track_ids) < 5:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 5 tracks de soporte")
        
        support_labels = {}
        if support_genres:
            support_labels["genres"] = support_genres
        if support_emotions:
            support_labels["emotions"] = support_emotions
        
        result = deep_learning_service.meta_learning_adapt(
            support_track_ids=support_track_ids,
            support_labels=support_labels,
            query_track_ids=query_track_ids,
            adaptation_steps=adaptation_steps
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in meta-learning: {e}")
        raise HTTPException(status_code=500, detail=f"Error en meta-learning: {str(e)}")


@router.post("/ml/deep-learning/few-shot", response_model=dict)
async def few_shot_learning(
    task_name: str = Query(...),
    example_track_ids: List[str] = Query(...),
    example_genres: Optional[List[int]] = Query(None),
    example_emotions: Optional[List[int]] = Query(None),
    query_track_ids: List[str] = Query(...)
):
    """Aprende de pocos ejemplos (few-shot learning)"""
    try:
        if len(example_track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 ejemplos")
        
        example_labels = {}
        if example_genres:
            example_labels["genres"] = example_genres
        if example_emotions:
            example_labels["emotions"] = example_emotions
        
        result = deep_learning_service.few_shot_learning(
            task_name=task_name,
            example_track_ids=example_track_ids,
            example_labels=example_labels,
            query_track_ids=query_track_ids
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in few-shot learning: {e}")
        raise HTTPException(status_code=500, detail=f"Error en few-shot learning: {str(e)}")


@router.post("/ml/deep-learning/analyze-causality", response_model=dict)
async def analyze_causality(
    track_ids: List[str] = Query(...),
    target_variable: str = Query("popularity", regex="^(popularity|danceability|energy|valence)$"),
    feature_names: Optional[List[str]] = Query(None)
):
    """Analiza relaciones causales entre características"""
    try:
        if len(track_ids) > 500:
            raise HTTPException(status_code=400, detail="Máximo 500 tracks")
        
        result = deep_learning_service.analyze_causality(
            track_ids=track_ids,
            target_variable=target_variable,
            feature_names=feature_names
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing causality: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@router.get("/ml/deep-learning/explain-advanced/{track_id}", response_model=dict)
async def explain_prediction_advanced(
    track_id: str,
    method: str = Query("gradient", regex="^(gradient|shap|lime)$"),
    top_k: int = Query(5, ge=1, le=10)
):
    """Explicación avanzada de predicciones"""
    try:
        result = deep_learning_service.explain_prediction_advanced(
            track_id=track_id,
            method=method,
            top_k=top_k
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error en explicación: {str(e)}")


@router.post("/ml/deep-learning/analyze-concepts", response_model=dict)
async def analyze_concepts(
    concept_track_ids: Dict[str, List[str]] = Query(...),
    query_track_ids: List[str] = Query(...)
):
    """Analiza conceptos musicales y su presencia en tracks"""
    try:
        if len(query_track_ids) > 200:
            raise HTTPException(status_code=400, detail="Máximo 200 tracks de query")
        
        result = deep_learning_service.analyze_concepts(
            concept_track_ids=concept_track_ids,
            query_track_ids=query_track_ids
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing concepts: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")
