"""
DTO conversion helper functions for use cases.

Provides reusable functions to convert dictionaries to DTOs,
eliminating repetitive conversion code across use cases.
"""

from typing import List, Dict, Any
from ...dto.recommendations import RecommendationDTO
from ...dto.analysis import TrackAnalysisDTO
from .data_extractors import (
    extract_track_id,
    extract_track_name,
    extract_artists,
    extract_album_name,
    safe_get_nested,
)


def convert_dict_to_recommendation_dto(rec_data: Dict[str, Any]) -> RecommendationDTO:
    """
    Convert a dictionary to RecommendationDTO.
    
    Handles different dictionary key formats and nested structures.
    This eliminates the repetitive conversion code that appears in
    get_recommendations.py and generate_playlist.py.
    
    Args:
        rec_data: Dictionary with track/recommendation data
    
    Returns:
        RecommendationDTO instance
    
    Example:
        >>> dto = convert_dict_to_recommendation_dto(track_dict)
        >>> # Handles id/track_id, name/track_name, nested album, etc.
    """
    return RecommendationDTO(
        track_id=extract_track_id(rec_data),
        track_name=extract_track_name(rec_data),
        artists=extract_artists(rec_data),
        similarity_score=safe_get_nested(rec_data, ["similarity_score", "similarity"]),
        reason=rec_data.get("reason"),
        album=extract_album_name(rec_data),
        preview_url=rec_data.get("preview_url"),
        popularity=rec_data.get("popularity")
    )


def convert_dict_to_track_analysis_dto(track_data: Dict[str, Any]) -> TrackAnalysisDTO:
    """
    Convert a dictionary to TrackAnalysisDTO.
    
    Handles different dictionary key formats and nested structures.
    This eliminates the repetitive conversion code that appears in
    search_tracks.py and analyze_track.py.
    
    Args:
        track_data: Dictionary with track data
    
    Returns:
        TrackAnalysisDTO instance
    
    Example:
        >>> dto = convert_dict_to_track_analysis_dto(track_dict)
        >>> # Handles id, name, nested artists, nested album, etc.
    """
    return TrackAnalysisDTO(
        track_id=extract_track_id(track_data),
        track_name=extract_track_name(track_data),
        artists=extract_artists(track_data),
        album=extract_album_name(track_data),
        duration_ms=track_data.get("duration_ms"),
        preview_url=track_data.get("preview_url"),
        popularity=track_data.get("popularity")
    )


def convert_dict_list_to_recommendation_dtos(
    data_list: List[Dict[str, Any]]
) -> List[RecommendationDTO]:
    """
    Convert a list of dictionaries to a list of RecommendationDTOs.
    
    Filters out non-dictionary items and converts each dictionary
    using convert_dict_to_recommendation_dto().
    
    Args:
        data_list: List of track/recommendation dictionaries
    
    Returns:
        List of RecommendationDTO instances
    
    Example:
        >>> dtos = convert_dict_list_to_recommendation_dtos(tracks_data)
        >>> # Converts all dict items, skips non-dict items
    """
    recommendations = []
    for rec_data in data_list:
        if isinstance(rec_data, dict):
            recommendations.append(convert_dict_to_recommendation_dto(rec_data))
    return recommendations


def convert_dict_list_to_track_analysis_dtos(
    data_list: List[Dict[str, Any]]
) -> List[TrackAnalysisDTO]:
    """
    Convert a list of dictionaries to a list of TrackAnalysisDTOs.
    
    Filters out non-dictionary items and converts each dictionary
    using convert_dict_to_track_analysis_dto().
    
    Args:
        data_list: List of track dictionaries
    
    Returns:
        List of TrackAnalysisDTO instances
    
    Example:
        >>> dtos = convert_dict_list_to_track_analysis_dtos(tracks_data)
        >>> # Converts all dict items, skips non-dict items
    """
    return [
        convert_dict_to_track_analysis_dto(track_data)
        for track_data in data_list
        if isinstance(track_data, dict)
    ]








