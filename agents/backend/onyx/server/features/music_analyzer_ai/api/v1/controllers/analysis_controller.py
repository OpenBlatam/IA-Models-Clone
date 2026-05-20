"""
Analysis Controller

Handles analysis-related HTTP requests using use cases.
Refactored to use helper functions for cleaner, more maintainable code.
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional
import logging

from ...dependencies import get_analyze_track_use_case
from ..schemas.requests import AnalyzeTrackRequest
from ..schemas.responses import AnalysisResponse, ErrorResponse
from ....application.use_cases.analysis import AnalyzeTrackUseCase
from ...utils.controller_helpers import handle_use_case_exceptions
from ...utils.response_helpers import build_analysis_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    """
    Analyze a music track.
    
    - **track_id**: Spotify track ID (optional if track_name provided)
    - **track_name**: Track name to search (optional if track_id provided)
    - **include_coaching**: Whether to include coaching recommendations
    """
    # Execute use case (now supports both track_id and track_name)
    result = await use_case.execute(
        track_id=request.track_id,
        track_name=request.track_name,
        include_coaching=request.include_coaching
    )
    
    # Build standardized response using helper
    return build_analysis_response(result, include_coaching=request.include_coaching)


@router.get("/{track_id}", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track_by_id(
    track_id: str,
    include_coaching: bool = Query(False, description="Include coaching analysis"),
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    """
    Analyze a track by its Spotify ID.
    
    - **track_id**: Spotify track ID
    - **include_coaching**: Whether to include coaching recommendations
    """
    result = await use_case.execute(track_id, include_coaching=include_coaching)
    
    # Build standardized response using helper
    return build_analysis_response(result, include_coaching=include_coaching)

