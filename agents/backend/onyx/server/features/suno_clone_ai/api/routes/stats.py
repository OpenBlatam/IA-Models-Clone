"""
Endpoints para estadísticas avanzadas y análisis

Este módulo proporciona endpoints para obtener estadísticas detalladas
sobre el uso del sistema, canciones, usuarios y métricas de rendimiento.

Características:
- Estadísticas generales del sistema
- Rankings y top canciones
- Análisis por período de tiempo
- Métricas de usuarios
- Estadísticas de géneros y tags
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends
from datetime import datetime, timedelta

from ..dependencies import SongServiceDep, MetricsServiceDep

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/stats",
    tags=["stats"],
    responses={
        400: {"description": "Bad request - Parámetros inválidos"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "/overview",
    summary="Estadísticas generales",
    description="Obtiene estadísticas generales del sistema para un período específico"
)
async def get_overview_stats(
    days: int = Query(7, ge=1, le=365, description="Número de días para el análisis"),
    include_trends: bool = Query(False, description="Incluir tendencias comparativas"),
    metrics_service: MetricsServiceDep = Depends(),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene estadísticas generales del sistema.
    
    Incluye:
    - Total de canciones generadas
    - Canciones por estado (processing, completed, failed)
    - Canciones por género
    - Usuarios más activos
    - Métricas de rendimiento
    - Tendencias comparativas (opcional)
    
    Args:
        days: Número de días para el período de análisis (1-365)
        include_trends: Incluir comparación con período anterior
        metrics_service: Servicio de métricas (inyectado)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Diccionario con estadísticas completas
    
    Example:
        ```
        GET /suno/stats/overview?days=30&include_trends=true
        ```
    """
    try:
        start_time = time.time()
        
        # Obtener todas las canciones
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        # Filtrar por período si es necesario
        if days < 365:
            cutoff_date = datetime.now() - timedelta(days=days)
            all_songs = [
                song for song in all_songs
                if song.get("metadata", {}).get("created_at")
                and datetime.fromisoformat(song.get("metadata", {}).get("created_at", "2000-01-01")) >= cutoff_date
            ]
        
        # Estadísticas básicas
        total_songs = len(all_songs)
        by_status = {}
        by_genre = {}
        user_activity = {}
        ratings_stats = {
            "total_rated": 0,
            "average_rating": 0.0,
            "total_ratings": 0
        }
        
        for song in all_songs:
            # Por estado
            status_val = song.get("status", "unknown")
            by_status[status_val] = by_status.get(status_val, 0) + 1
            
            # Por género
            genre = song.get("metadata", {}).get("genre", "unknown")
            by_genre[genre] = by_genre.get(genre, 0) + 1
            
            # Actividad de usuarios
            user_id = song.get("user_id")
            if user_id:
                user_activity[user_id] = user_activity.get(user_id, 0) + 1
            
            # Estadísticas de ratings
            metadata = song.get("metadata", {})
            if metadata.get("ratings"):
                ratings_stats["total_rated"] += 1
                ratings_stats["total_ratings"] += len(metadata.get("ratings", {}))
                avg_rating = metadata.get("average_rating", 0.0)
                if avg_rating > 0:
                    ratings_stats["average_rating"] = (
                        (ratings_stats["average_rating"] * (ratings_stats["total_rated"] - 1) + avg_rating) /
                        ratings_stats["total_rated"]
                    )
        
        # Obtener métricas del servicio
        metrics = metrics_service.get_stats(days=days) if metrics_service else {}
        
        result = {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "total_songs": total_songs,
            "by_status": by_status,
            "by_genre": by_genre,
            "top_users": dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "ratings": {
                **ratings_stats,
                "average_rating": round(ratings_stats["average_rating"], 2)
            },
            "metrics": metrics
        }
        
        # Agregar tendencias si se solicita
        if include_trends:
            previous_period_songs = song_service.list_songs(limit=10000, offset=0)
            previous_cutoff = datetime.now() - timedelta(days=days * 2)
            previous_songs = [
                song for song in previous_period_songs
                if song.get("metadata", {}).get("created_at")
                and previous_cutoff <= datetime.fromisoformat(song.get("metadata", {}).get("created_at", "2000-01-01")) < cutoff_date
            ]
            
            result["trends"] = {
                "previous_period_songs": len(previous_songs),
                "current_period_songs": total_songs,
                "growth": total_songs - len(previous_songs),
                "growth_percentage": round(((total_songs - len(previous_songs)) / len(previous_songs) * 100) if previous_songs else 0, 2)
            }
        
        processing_time = time.time() - start_time
        result["processing_time_ms"] = round(processing_time * 1000, 2)
        
        return result
    except Exception as e:
        logger.error(f"Error getting overview stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting overview stats: {str(e)}"
        )


@router.get(
    "/top-rated",
    summary="Top canciones calificadas",
    description="Obtiene las canciones mejor calificadas del sistema"
)
async def get_top_rated_songs(
    limit: int = Query(10, ge=1, le=100, description="Número de canciones a retornar"),
    min_ratings: int = Query(1, ge=1, description="Mínimo de ratings requeridos"),
    min_rating_value: Optional[float] = Query(None, ge=0.0, le=5.0, description="Rating mínimo promedio"),
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene las canciones mejor calificadas del sistema.
    
    Las canciones se ordenan por rating promedio y se filtran por
    número mínimo de ratings para evitar sesgos.
    
    Args:
        limit: Número de canciones a retornar (1-100)
        min_ratings: Mínimo de ratings requeridos para aparecer
        min_rating_value: Rating promedio mínimo (opcional)
        song_service: Servicio de canciones (inyectado)
    
    Returns:
        Lista de canciones ordenadas por rating promedio
    
    Example:
        ```
        GET /suno/stats/top-rated?limit=20&min_ratings=5&min_rating_value=4.0
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        rated_songs = []
        for song in all_songs:
            metadata = song.get("metadata", {})
            ratings = metadata.get("ratings", {})
            total_ratings = len(ratings)
            average_rating = metadata.get("average_rating", 0.0)
            
            # Aplicar filtros
            if total_ratings < min_ratings:
                continue
            
            if min_rating_value is not None and average_rating < min_rating_value:
                continue
            
            rated_songs.append({
                "song": song,
                "average_rating": round(average_rating, 2),
                "total_ratings": total_ratings,
                "song_id": song.get("song_id"),
                "prompt": song.get("prompt", "")
            })
        
        # Ordenar por rating promedio (descendente), luego por total de ratings
        rated_songs.sort(
            key=lambda x: (x["average_rating"], x["total_ratings"]),
            reverse=True
        )
        
        return {
            "top_rated": rated_songs[:limit],
            "total_found": len(rated_songs),
            "min_ratings": min_ratings,
            "min_rating_value": min_rating_value,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting top rated songs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting top rated songs: {str(e)}"
        )


@router.get("/genres")
async def get_genre_stats(
    song_service: SongServiceDep = Depends()
) -> Dict[str, Any]:
    """
    Obtiene estadísticas por género musical.
    
    Returns:
        Estadísticas detalladas de cada género
    
    Example:
        ```
        GET /suno/stats/genres
        ```
    """
    try:
        all_songs = song_service.list_songs(limit=10000, offset=0)
        
        genre_stats = {}
        
        for song in all_songs:
            genre = song.get("metadata", {}).get("genre", "unknown")
            if genre not in genre_stats:
                genre_stats[genre] = {
                    "count": 0,
                    "total_ratings": 0,
                    "average_rating": 0.0,
                    "total_favorites": 0
                }
            
            genre_stats[genre]["count"] += 1
            
            metadata = song.get("metadata", {})
            ratings = metadata.get("ratings", {})
            if ratings:
                genre_stats[genre]["total_ratings"] += len(ratings)
                avg_rating = metadata.get("average_rating", 0.0)
                if avg_rating > 0:
                    current_avg = genre_stats[genre]["average_rating"]
                    current_count = genre_stats[genre]["count"]
                    genre_stats[genre]["average_rating"] = (
                        (current_avg * (current_count - 1) + avg_rating) / current_count
                    )
            
            favorites = metadata.get("favorites", [])
            genre_stats[genre]["total_favorites"] += len(favorites)
        
        # Calcular promedios finales
        for genre, stats in genre_stats.items():
            stats["average_rating"] = round(stats["average_rating"], 2)
        
        return {
            "genres": genre_stats,
            "total_genres": len(genre_stats),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting genre stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting genre stats: {str(e)}"
        )

