"""
Search Controller

Handles search-related HTTP requests using use cases.
Refactored to use helper functions for cleaner, more maintainable code.
"""

from fastapi import APIRouter, Depends, Query
from typing import Optional
import logging

from ...dependencies import get_search_tracks_use_case
from ..schemas.requests import SearchTracksRequest
from ..schemas.responses import SearchResponse, ErrorResponse
from ....application.use_cases.analysis import SearchTracksUseCase
from ...utils.controller_helpers import handle_use_case_exceptions
from ...utils.response_helpers import build_search_response_from_objects

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse, responses={400: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def search_tracks(
    request: SearchTracksRequest,
    use_case: SearchTracksUseCase = Depends(get_search_tracks_use_case)
):
    """
    Search for music tracks.
    
    - **query**: Search query string
    - **limit**: Maximum number of results (1-50)
    - **offset**: Pagination offset
    """
    tracks = await use_case.execute(
        request.query,
        limit=request.limit,
        offset=request.offset
    )
    
    # Build standardized response - handles object conversion automatically
    return build_search_response_from_objects(
        tracks,
        query=request.query,
        metadata={"limit": request.limit, "offset": request.offset}
    )


@router.get("", response_model=SearchResponse, responses={400: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def search_tracks_get(
    q: str = Query(..., description="Search query", alias="query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    use_case: SearchTracksUseCase = Depends(get_search_tracks_use_case)
):
    """
    Search for music tracks (GET method).
    
    - **q**: Search query string
    - **limit**: Maximum number of results (1-50)
    - **offset**: Pagination offset
    """
    tracks = await use_case.execute(q, limit=limit, offset=offset)
    
    # Build standardized response - handles object conversion automatically
    return build_search_response_from_objects(
        tracks,
        query=q,
        metadata={"limit": limit, "offset": offset}
    )

