"""
Helper functions for common service operations
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_track_or_search(
    spotify_service,
    track_id: Optional[str] = None,
    track_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get track by ID or search by name
    
    Args:
        spotify_service: Spotify service instance
        track_id: Optional track ID
        track_name: Optional track name to search
    
    Returns:
        Track ID string
    
    Raises:
        ValueError: If neither track_id nor track_name provided
        HTTPException: If track not found
    """
    if track_id:
        return track_id
    
    if track_name:
        tracks = spotify_service.search_track(track_name, limit=1)
        if not tracks:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró la canción: {track_name}"
            )
        return tracks[0]["id"]
    
    from fastapi import HTTPException
    raise HTTPException(
        status_code=400,
        detail="Debe proporcionar track_id o track_name"
    )


def get_track_full_data(
    spotify_service,
    track_id: str
) -> Dict[str, Any]:
    """
    Get complete track data including analysis
    
    Args:
        spotify_service: Spotify service instance
        track_id: Track ID
    
    Returns:
        Complete track data dictionary
    """
    return spotify_service.get_track_full_analysis(track_id)


def format_artists(artists: list) -> list:
    """
    Format artists list from Spotify response.
    
    Uses object_helpers for safe attribute access.
    
    Args:
        artists: List of artist objects from Spotify
    
    Returns:
        List of artist names
    """
    from .object_helpers import safe_get_attribute
    
    if not artists:
        return []
    
    return [
        safe_get_attribute(artist, "name", default=str(artist))
        for artist in artists
    ]


def safe_get_nested(data: Dict[str, Any], *keys, default=None) -> Any:
    """
    Safely get nested dictionary value.
    
    This is a convenience wrapper around safe_get_attribute for multiple keys.
    Consider using safe_get_attribute with dot notation for better flexibility.
    
    Args:
        data: Dictionary to search
        *keys: Keys to traverse
        default: Default value if not found
    
    Returns:
        Value or default
    
    Example:
        # Old way
        value = safe_get_nested(data, "user", "profile", "name", default="Unknown")
        
        # New way (preferred)
        from .object_helpers import safe_get_attribute
        value = safe_get_attribute(data, "user.profile.name", default="Unknown")
    """
    from .object_helpers import safe_get_attribute
    
    # Convert keys to dot notation path
    path = ".".join(str(k) for k in keys)
    return safe_get_attribute(data, path, default=default)

