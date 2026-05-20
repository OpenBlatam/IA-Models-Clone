"""
Data extraction helper functions for use cases.

Provides reusable functions to safely extract data from dictionaries,
handling different key formats and nested structures.
"""

from typing import Dict, Any, List, Optional, Union


def extract_track_id(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract track ID from dictionary, handling different key formats.
    
    Tries "id" first, then "track_id" as fallback.
    
    Args:
        data: Dictionary with track data
    
    Returns:
        Track ID or None
    
    Example:
        >>> track_id = extract_track_id(track_data)
        >>> # Returns track_data.get("id") or track_data.get("track_id")
    """
    return data.get("id") or data.get("track_id")


def extract_track_name(data: Dict[str, Any], default: str = "Unknown") -> str:
    """
    Extract track name from dictionary, handling different key formats.
    
    Tries "name" first, then "track_name" as fallback.
    
    Args:
        data: Dictionary with track data
        default: Default value if not found
    
    Returns:
        Track name or default
    
    Example:
        >>> name = extract_track_name(track_data)
        >>> # Returns track_data.get("name") or track_data.get("track_name", "Unknown")
    """
    return data.get("name") or data.get("track_name", default)


def extract_artists(data: Dict[str, Any]) -> List[str]:
    """
    Extract artists list from dictionary, handling different formats.
    
    Handles multiple formats:
    - List of artist objects: [{"name": "Artist1"}, {"name": "Artist2"}]
    - List of strings: ["Artist1", "Artist2"]
    - Single artist object: {"name": "Artist1"}
    - Single string: "Artist1"
    
    Args:
        data: Dictionary with track data
    
    Returns:
        List of artist names (always returns a list, never empty)
    
    Example:
        >>> artists = extract_artists(track_data)
        >>> # Returns ["Artist1", "Artist2"] or ["Unknown"] if not found
    """
    artists_data = data.get("artists", [])
    
    if not artists_data:
        return ["Unknown"]
    
    # If it's a list
    if isinstance(artists_data, list):
        # If list is empty, return default
        if not artists_data:
            return ["Unknown"]
        
        # Check first item to determine format
        first_item = artists_data[0]
        
        # List of artist objects: [{"name": "Artist1"}, ...]
        if isinstance(first_item, dict):
            return [
                artist.get("name", "Unknown")
                for artist in artists_data
            ]
        
        # List of strings: ["Artist1", "Artist2"]
        elif isinstance(first_item, str):
            return artists_data
        
        # Unknown format, return default
        return ["Unknown"]
    
    # Single artist object: {"name": "Artist1"}
    elif isinstance(artists_data, dict):
        return [artists_data.get("name", "Unknown")]
    
    # Single string: "Artist1"
    elif isinstance(artists_data, str):
        return [artists_data]
    
    # Unknown format
    return ["Unknown"]


def extract_album_name(data: Dict[str, Any], default: Optional[str] = None) -> Optional[str]:
    """
    Extract album name from dictionary, handling nested structures.
    
    Handles:
    - Nested: {"album": {"name": "Album Name"}}
    - Direct: {"album": "Album Name"}
    - Missing: {"album": None} or no "album" key
    
    Args:
        data: Dictionary with track data
        default: Default value if not found
    
    Returns:
        Album name or default
    
    Example:
        >>> album = extract_album_name(track_data)
        >>> # Returns album.name if album is dict, or album if string, or None
    """
    album_data = data.get("album")
    
    # If album is a dictionary, get name
    if isinstance(album_data, dict):
        return album_data.get("name", default)
    
    # If album is a string, return it
    if isinstance(album_data, str):
        return album_data
    
    # If album is not present or None
    return default


def safe_get_nested(
    data: Dict[str, Any],
    keys: List[str],
    default: Any = None
) -> Any:
    """
    Safely get value from dictionary using multiple possible keys.
    
    Tries each key in order until one is found. Useful for handling
    different API response formats.
    
    Args:
        data: Dictionary to search
        keys: List of keys to try (in order of preference)
        default: Default value if none found
    
    Returns:
        Value from first matching key, or default
    
    Example:
        >>> score = safe_get_nested(rec_data, ["similarity_score", "similarity"], 0.0)
        >>> # Tries "similarity_score" first, then "similarity", then returns 0.0
    """
    for key in keys:
        if key in data:
            return data[key]
    return default








