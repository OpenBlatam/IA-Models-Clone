"""
Track-related helper functions for common track operations.

This module provides utilities for track ID resolution, validation,
and other common track-related operations.
"""

from typing import Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def resolve_track_id(
    track_id: Optional[str],
    track_name: Optional[str],
    spotify_service,
    raise_on_not_found: bool = True
) -> Optional[str]:
    """
    Resolve track_id from either track_id or track_name.
    
    If track_id is provided, returns it directly.
    If only track_name is provided, searches for the track and returns its ID.
    
    Args:
        track_id: Optional Spotify track ID
        track_name: Optional track name to search
        spotify_service: Spotify service instance with search_track method
        raise_on_not_found: Whether to raise exception if track not found
    
    Returns:
        Resolved track ID, or None if not found and raise_on_not_found=False
    
    Raises:
        HTTPException: If track_id is None and track_name search fails
            (when raise_on_not_found=True)
        ValueError: If both track_id and track_name are None
    """
    # If track_id is provided, use it
    if track_id:
        return track_id
    
    # If track_name is provided, search for it
    if track_name:
        try:
            tracks = spotify_service.search_track(track_name, limit=1)
            if not tracks:
                if raise_on_not_found:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No se encontró la canción: {track_name}"
                    )
                return None
            
            found_track_id = tracks[0].get("id")
            if not found_track_id:
                if raise_on_not_found:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Canción encontrada pero sin ID: {track_name}"
                    )
                return None
            
            return found_track_id
            
        except Exception as e:
            logger.error(f"Error searching for track '{track_name}': {e}")
            if raise_on_not_found:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error al buscar la canción: {str(e)}"
                )
            return None
    
    # Neither provided
    raise ValueError("Either track_id or track_name must be provided")


def resolve_track_id_from_request(
    track_id: Optional[str],
    track_name: Optional[str],
    spotify_service: Any
) -> str:
    """
    Resolve track ID from request parameters.
    
    Handles:
    - Direct track_id provided -> return as-is (after validation)
    - track_name provided -> search and return first result
    - Neither provided -> raise HTTPException
    
    Args:
        track_id: Optional track ID
        track_name: Optional track name to search
        spotify_service: Spotify service instance with search_track method
    
    Returns:
        Resolved and validated track ID string
    
    Raises:
        HTTPException: If neither provided or track not found
    
    Example:
        track_id = resolve_track_id_from_request(
            request.track_id,
            request.track_name,
            spotify_service
        )
    """
    from fastapi import HTTPException
    
    # If track_id provided, validate and return
    if track_id:
        validate_track_id(track_id)
        return track_id
    
    # If track_name provided, search for it
    if track_name:
        tracks = spotify_service.search_track(track_name, limit=1)
        if not tracks:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró la canción: {track_name}"
            )
        found_track_id = tracks[0].get("id")
        if found_track_id:
            return found_track_id
        raise HTTPException(
            status_code=404,
            detail=f"Canción encontrada pero sin ID válido: {track_name}"
        )
    
    # Neither provided
    raise HTTPException(
        status_code=400,
        detail="Debe proporcionar track_id o track_name"
    )


def validate_track_id(track_id: str) -> None:
    """
    Validate that a track ID is not empty.
    
    Args:
        track_id: Track ID to validate
    
    Raises:
        HTTPException: If track_id is empty or None
    """
    if not track_id or not track_id.strip():
        raise HTTPException(
            status_code=400,
            detail="track_id no puede estar vacío"
        )

