"""
Formatting helper functions for common data formatting patterns.

This module provides utilities for formatting common data structures
like tracks, artists, albums, etc. with consistent patterns.
"""

from typing import Any, List, Dict, Optional, Callable, Union
from .object_helpers import safe_get_attribute
from .data_transformation_helpers import map_list


def format_artist_name(artist: Any) -> str:
    """
    Format artist name from various formats.
    
    Handles:
    - Dictionary with "name" key
    - String
    - Objects with name attribute
    
    Args:
        artist: Artist data (dict, string, or object)
    
    Returns:
        Artist name as string
    
    Example:
        name = format_artist_name(artist_dict)  # "Artist Name"
        name = format_artist_name("Artist Name")  # "Artist Name"
    """
    return safe_get_attribute(artist, "name", default=str(artist))


def format_artist_list(artists: List[Any]) -> List[str]:
    """
    Format list of artists to list of artist names.
    
    Args:
        artists: List of artist objects/dictionaries
    
    Returns:
        List of artist names
    
    Example:
        names = format_artist_list(track.get("artists", []))
        # ["Artist 1", "Artist 2"]
    """
    if not artists:
        return []
    
    return map_list(artists, format_artist_name, filter_none=True)


def format_album_name(album: Any) -> Optional[str]:
    """
    Format album name from various formats.
    
    Args:
        album: Album data (dict, string, or None)
    
    Returns:
        Album name or None
    
    Example:
        name = format_album_name(track.get("album"))  # "Album Name"
    """
    if album is None:
        return None
    
    if isinstance(album, str):
        return album
    
    return safe_get_attribute(album, "name", default=None)


def format_track_basic_info(track: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format basic track information consistently.
    
    Extracts and formats common track fields:
    - id, name, artists, album, duration_ms, preview_url, etc.
    
    Args:
        track: Track dictionary from API
    
    Returns:
        Formatted track dictionary
    
    Example:
        formatted = format_track_basic_info(spotify_track)
    """
    return {
        "id": track.get("id"),
        "name": track.get("name"),
        "artists": format_artist_list(track.get("artists", [])),
        "album": format_album_name(track.get("album")),
        "duration_ms": track.get("duration_ms"),
        "preview_url": track.get("preview_url"),
        "external_urls": track.get("external_urls", {}),
        "popularity": track.get("popularity", 0)
    }


def format_track_list_basic(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format a list of tracks with basic information.
    
    Args:
        tracks: List of track dictionaries
    
    Returns:
        List of formatted track dictionaries
    
    Example:
        formatted_tracks = format_track_list_basic(spotify_tracks)
    """
    return [format_track_basic_info(track) for track in tracks]


def extract_track_ids(tracks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract track IDs from a list of tracks.
    
    Args:
        tracks: List of track dictionaries
    
    Returns:
        List of track IDs
    
    Example:
        ids = extract_track_ids(tracks)  # ["id1", "id2", "id3"]
    """
    return [
        track.get("id")
        for track in tracks
        if track.get("id")
    ]


def extract_track_names(tracks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract track names from a list of tracks.
    
    Args:
        tracks: List of track dictionaries
    
    Returns:
        List of track names
    
    Example:
        names = extract_track_names(tracks)  # ["Track 1", "Track 2"]
    """
    return [
        track.get("name")
        for track in tracks
        if track.get("name")
    ]


def format_duration(
    duration_ms: Optional[int],
    format: str = "seconds"
) -> Optional[Union[int, str, Dict[str, int]]]:
    """
    Format duration from milliseconds to various formats.
    
    Args:
        duration_ms: Duration in milliseconds
        format: Output format - "seconds", "minutes", "mm:ss", "dict"
    
    Returns:
        Formatted duration
    
    Example:
        seconds = format_duration(180000, format="seconds")  # 180
        minutes = format_duration(180000, format="minutes")  # 3
        time_str = format_duration(180000, format="mm:ss")  # "3:00"
        time_dict = format_duration(180000, format="dict")  # {"minutes": 3, "seconds": 0}
    """
    if duration_ms is None:
        return None
    
    seconds = duration_ms // 1000
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    
    if format == "seconds":
        return seconds
    elif format == "minutes":
        return minutes
    elif format == "mm:ss":
        return f"{minutes}:{remaining_seconds:02d}"
    elif format == "dict":
        return {
            "total_ms": duration_ms,
            "total_seconds": seconds,
            "minutes": minutes,
            "seconds": remaining_seconds
        }
    
    return seconds


def format_popularity_score(
    popularity: Optional[int],
    scale: int = 100
) -> Optional[float]:
    """
    Format popularity score as percentage or normalized value.
    
    Args:
        popularity: Popularity score (0-100 typically)
        scale: Scale factor (default: 100 for percentage)
    
    Returns:
        Formatted popularity (0.0-1.0 or 0-100)
    
    Example:
        percentage = format_popularity_score(75)  # 0.75
        percentage = format_popularity_score(75, scale=100)  # 0.75
    """
    if popularity is None:
        return None
    
    return popularity / scale if scale > 0 else popularity

