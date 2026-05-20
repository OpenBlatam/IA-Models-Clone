"""
Request validators for API endpoints
"""

from typing import List, Optional
from fastapi import HTTPException


def validate_track_id(track_id: str) -> None:
    """Validate Spotify track ID format"""
    if not track_id or not isinstance(track_id, str):
        raise HTTPException(status_code=400, detail="track_id es requerido")
    if len(track_id) < 10:
        raise HTTPException(status_code=400, detail="track_id inválido")


def validate_track_ids(track_ids: List[str], min_count: int = 1, max_count: int = 100) -> None:
    """Validate list of track IDs"""
    if not track_ids:
        raise HTTPException(status_code=400, detail="Lista de track IDs no puede estar vacía")
    if len(track_ids) < min_count:
        raise HTTPException(
            status_code=400,
            detail=f"Se necesitan al menos {min_count} track(s)"
        )
    if len(track_ids) > max_count:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {max_count} tracks"
        )


def validate_limit(limit: int, min_val: int = 1, max_val: int = 100) -> None:
    """Validate limit parameter"""
    if limit < min_val or limit > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"Limit debe estar entre {min_val} y {max_val}"
        )


def validate_user_id(user_id: Optional[str], required: bool = False) -> None:
    """Validate user ID"""
    if required and not user_id:
        raise HTTPException(status_code=400, detail="user_id es requerido")


def validate_search_query(query: str) -> None:
    """Validate search query"""
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Query de búsqueda es requerido")
    if len(query.strip()) < 1:
        raise HTTPException(status_code=400, detail="Query de búsqueda no puede estar vacío")
    if len(query) > 200:
        raise HTTPException(status_code=400, detail="Query de búsqueda demasiado largo")

