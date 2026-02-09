"""
Response building helper functions for consistent API responses.

This module provides utilities to build standardized response formats
across all API endpoints.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime


def build_analysis_response(
    result: Any,
    include_coaching: bool = True
) -> Dict[str, Any]:
    """
    Build a standardized analysis response from a use case result.
    
    Handles both DTO objects and dictionaries, making it flexible
    for different use case return types.
    
    Args:
        result: Analysis result (DTO or dict) from use case
        include_coaching: Whether to include coaching data
    
    Returns:
        Standardized response dictionary with structure:
        {
            "success": True,
            "track_id": str,
            "track_name": str,
            "artists": list,
            "album": str,
            "duration_seconds": int,
            "analysis": dict,
            "coaching": dict (optional)
        }
    """
    # Handle DTO objects (with attributes)
    if hasattr(result, 'track_id'):
        response = {
            "success": True,
            "track_id": result.track_id,
            "track_name": getattr(result, 'track_name', None),
            "artists": getattr(result, 'artists', []),
            "album": getattr(result, 'album', None),
            "duration_seconds": getattr(result, 'duration_seconds', None),
            "analysis": getattr(result, 'analysis', {})
        }
        
        if include_coaching and hasattr(result, 'coaching') and result.coaching:
            response["coaching"] = result.coaching
    
    # Handle dictionaries
    elif isinstance(result, dict):
        response = {
            "success": True,
            "track_id": result.get("track_id"),
            "track_name": result.get("track_name"),
            "artists": result.get("artists", []),
            "album": result.get("album"),
            "duration_seconds": result.get("duration_seconds"),
            "analysis": result.get("analysis", {})
        }
        
        if include_coaching and result.get("coaching"):
            response["coaching"] = result["coaching"]
    
    else:
        # Fallback: wrap the result
        response = {
            "success": True,
            "data": result
        }
    
    return response


def build_search_response(
    tracks: List[Dict[str, Any]],
    query: str,
    total: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a standardized search response.
    
    Args:
        tracks: List of track dictionaries
        query: Search query string
        total: Total count (if different from len(tracks), e.g., paginated)
        metadata: Optional additional metadata
    
    Returns:
        Standardized search response with structure:
        {
            "success": True,
            "query": str,
            "results": list,
            "total": int,
            "metadata": dict (optional)
        }
    """
    response = {
        "success": True,
        "query": query,
        "results": tracks,
        "total": total or len(tracks)
    }
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def build_search_response_from_objects(
    tracks: List[Any],
    query: str,
    total: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a standardized search response from a list of track objects.
    
    Automatically converts objects to dictionaries before building response.
    This ensures consistency even if use case returns objects instead of dicts.
    
    Args:
        tracks: List of track objects or dictionaries
        query: Search query string
        total: Total count (if different from len(tracks), e.g., paginated)
        metadata: Optional additional metadata
    
    Returns:
        Standardized search response with structure:
        {
            "success": True,
            "query": str,
            "results": [dict, dict, ...],
            "total": int,
            "metadata": dict (optional)
        }
    
    Example:
        tracks = await use_case.execute(query, limit=limit)
        return build_search_response_from_objects(
            tracks,
            query=query,
            total=total_count
        )
    """
    from .object_helpers import to_dict_list
    
    # Convert objects to dictionaries
    tracks_dict = to_dict_list(tracks)
    
    # Build response using existing helper
    return build_search_response(
        tracks=tracks_dict,
        query=query,
        total=total,
        metadata=metadata
    )


def build_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a generic success response.
    
    Args:
        data: Response data
        message: Optional success message
        metadata: Optional metadata
    
    Returns:
        Standardized success response
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def build_list_response(
    items: List[Any],
    key: str = "items",
    include_total: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized list response.
    
    Args:
        items: List of items
        key: Key name for items in response (default: "items")
        include_total: Whether to include total count
        **kwargs: Additional fields to include in response
    
    Returns:
        Standardized list response
    """
    response = {
        key: items
    }
    
    if include_total:
        response["total"] = len(items)
    
    response.update(kwargs)
    
    return {
        "success": True,
        **response
    }


def build_list_response_from_objects(
    items: List[Any],
    key: str = "items",
    include_total: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized list response from a list of objects.
    
    Automatically converts objects to dictionaries before building response.
    This eliminates the need for separate to_dict_list() calls.
    
    Args:
        items: List of objects to convert and include in response
        key: Key name for items in response (default: "items")
        include_total: Whether to include total count
        **kwargs: Additional fields to include in response
    
    Returns:
        Standardized list response with structure:
        {
            "success": True,
            key: [dict, dict, ...],
            "total": int,
            ...kwargs
        }
    
    Example:
        recommendations = await use_case.execute(...)
        return build_list_response_from_objects(
            recommendations,
            key="recommendations",
            track_id=track_id,
            method=method
        )
    """
    from .object_helpers import to_dict_list
    
    # Convert objects to dictionaries
    items_dict = to_dict_list(items)
    
    # Build response using existing helper
    return build_list_response(
        items=items_dict,
        key=key,
        include_total=include_total,
        **kwargs
    )


def build_success_response_from_object(
    obj: Any,
    data_key: str = "data",
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build a standardized success response from an object.
    
    Automatically converts object to dictionary before building response.
    This eliminates the need for separate to_dict() calls.
    
    Args:
        obj: Object to convert and include in response
        data_key: Key name for object in response data (default: "data")
        message: Optional success message
        metadata: Optional metadata
        **kwargs: Additional fields to include in response data
    
    Returns:
        Standardized success response with structure:
        {
            "success": True,
            "data": {
                data_key: dict,
                ...kwargs
            },
            "message": str (optional),
            "metadata": dict (optional),
            "timestamp": str
        }
    
    Example:
        playlist = await use_case.execute(...)
        return build_success_response_from_object(
            playlist,
            data_key="playlist",
            criteria=criteria
        )
    """
    from .object_helpers import to_dict
    
    # Convert object to dictionary
    obj_dict = to_dict(obj)
    
    # Build data dictionary
    data = {
        data_key: obj_dict,
        **kwargs
    }
    
    # Build response using existing helper
    return build_success_response(
        data=data,
        message=message,
        metadata=metadata
    )
