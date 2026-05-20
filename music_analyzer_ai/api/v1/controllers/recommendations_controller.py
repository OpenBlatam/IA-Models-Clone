"""
Recommendations Controller

Handles recommendation-related HTTP requests using use cases.
Refactored to use helper functions for cleaner, more maintainable code.
"""

from fastapi import APIRouter, Depends, Query, Body
from typing import Optional, List
import logging

from ...dependencies import (
    get_recommendations_use_case,
    get_generate_playlist_use_case
)
from ..schemas.requests import GeneratePlaylistRequest
from ..schemas.responses import RecommendationResponse, PlaylistResponse, ErrorResponse
from ....application.use_cases.recommendations import (
    GetRecommendationsUseCase,
    GeneratePlaylistUseCase
)
from ...utils.controller_helpers import handle_use_case_exceptions
from ...utils.response_helpers import (
    build_list_response_from_objects,
    build_success_response_from_object
)
from ...utils.request_helpers import build_criteria_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get(
    "/track/{track_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
@handle_use_case_exceptions
async def get_track_recommendations(
    track_id: str,
    limit: int = Query(20, ge=1, le=50, description="Maximum number of recommendations"),
    method: str = Query("similarity", description="Recommendation method: similarity, mood, genre"),
    mood: Optional[str] = Query(None, description="Target mood (for mood-based recommendations)"),
    genre: Optional[str] = Query(None, description="Target genre (for genre-based recommendations)"),
    use_case: GetRecommendationsUseCase = Depends(get_recommendations_use_case)
):
    """
    Get recommendations for a track.
    
    - **track_id**: Spotify track ID
    - **limit**: Maximum number of recommendations
    - **method**: Recommendation method (similarity, mood, genre)
    - **mood**: Target mood (optional, for mood-based recommendations)
    - **genre**: Target genre (optional, for genre-based recommendations)
    """
    recommendations = await use_case.execute(
        track_id=track_id,
        limit=limit,
        method=method,
        mood=mood,
        genre=genre
    )
    
    # Build response - conversion handled automatically
    return build_list_response_from_objects(
        recommendations,
        key="recommendations",
        track_id=track_id,
        method=method
    )


@router.post(
    "/playlist",
    response_model=PlaylistResponse,
    responses={500: {"model": ErrorResponse}}
)
@handle_use_case_exceptions
async def generate_playlist(
    request: GeneratePlaylistRequest,
    use_case: GeneratePlaylistUseCase = Depends(get_generate_playlist_use_case)
):
    """
    Generate a playlist based on criteria.
    
    - **genres**: Target genres
    - **moods**: Target moods
    - **energy_range**: Energy range [min, max]
    - **tempo_range**: Tempo range [min, max]
    - **length**: Desired playlist length
    """
    # Build criteria dict from request using helper
    criteria = build_criteria_dict(
        genres=request.genres,
        moods=request.moods,
        energy_range=request.energy_range,
        tempo_range=request.tempo_range,
        seed_track_id=request.seed_track_id
    )
    
    playlist = await use_case.execute(criteria, length=request.length)
    
    # Build response - conversion handled automatically
    return build_success_response_from_object(
        playlist,
        data_key="playlist",
        criteria=criteria
    )

