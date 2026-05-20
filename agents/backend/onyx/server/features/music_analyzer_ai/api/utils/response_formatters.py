"""
Response formatters for consistent API responses.

This module provides utilities for formatting API responses consistently.
Uses object_helpers for safe attribute access.
"""

from typing import Any, Optional, List, Dict
from .object_helpers import safe_get_attribute


def format_track_response(track: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a track response with consistent structure.
    
    Uses safe_get_attribute for nested access to prevent KeyError.
    
    Args:
        track: Track dictionary from Spotify API or similar
    
    Returns:
        Formatted track dictionary
    """
    # Extract artists safely
    artists_raw = track.get("artists", [])
    artists = [
        safe_get_attribute(artist, "name", default=str(artist))
        for artist in artists_raw
    ]
    
    # Extract album name safely
    album = safe_get_attribute(track, "album.name", default=None)
    if not album and isinstance(track.get("album"), str):
        album = track.get("album")
    
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": artists,
        "album": album,
        "duration_ms": track.get("duration_ms"),
        "preview_url": track.get("preview_url"),
        "external_urls": track.get("external_urls", {}),
        "popularity": track.get("popularity", 0)
    }


def format_tracks_response(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format a list of tracks"""
    return [format_track_response(track) for track in tracks]


def format_paginated_response(
    items: List[Any],
    page: int = 1,
    limit: int = 20,
    total: Optional[int] = None
) -> Dict[str, Any]:
    """Format a paginated response"""
    if total is None:
        total = len(items)
    
    return {
        "items": items,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit if limit > 0 else 0
        }
    }

