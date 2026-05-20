# Refactoring Example: Using Helper Functions

This document shows before/after examples of refactoring controllers to use the new helper functions.

## Example 1: Analysis Controller

### Before (Original Code)

```python
# api/v1/controllers/analysis_controller.py (BEFORE)

@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
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
    try:
        # Execute use case (now supports both track_id and track_name)
        result = await use_case.execute(
            track_id=request.track_id,
            track_name=request.track_name,
            include_coaching=request.include_coaching
        )
        
        # Convert DTO to response format
        response = {
            "success": True,
            "track_id": result.track_id,
            "track_name": result.track_name,
            "artists": result.artists,
            "album": result.album,
            "duration_seconds": result.duration_seconds,
            "analysis": result.analysis,
            "coaching": result.coaching
        }
        
        return response
        
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in analyze_track: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{track_id}", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
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
    try:
        result = await use_case.execute(track_id, include_coaching=include_coaching)
        
        response = {
            "success": True,
            "track_id": result.track_id,
            "track_name": result.track_name,
            "artists": result.artists,
            "album": result.album,
            "duration_seconds": result.duration_seconds,
            "analysis": result.analysis,
            "coaching": result.coaching
        }
        
        return response
        
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in analyze_track_by_id: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Lines of code**: ~60 lines
**Duplication**: High (exception handling and response building duplicated)

---

### After (Refactored with Helpers)

```python
# api/v1/controllers/analysis_controller.py (AFTER)

from fastapi import APIRouter, Depends, HTTPException, Query
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
    result = await use_case.execute(
        track_id=request.track_id,
        track_name=request.track_name,
        include_coaching=request.include_coaching
    )
    
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
    return build_analysis_response(result, include_coaching=include_coaching)
```

**Lines of code**: ~35 lines
**Duplication**: None
**Improvement**: 42% reduction in code, 100% elimination of duplication

---

## Example 2: Search Controller

### Before (Original Code)

```python
# api/v1/controllers/search_controller.py (BEFORE)

@router.post("", response_model=SearchResponse, responses={400: {"model": ErrorResponse}})
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
    try:
        result = await use_case.execute(
            request.query,
            limit=request.limit,
            offset=request.offset
        )
        return result
        
    except UseCaseException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in search_tracks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("", response_model=SearchResponse, responses={400: {"model": ErrorResponse}})
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
    try:
        result = await use_case.execute(q, limit=limit, offset=offset)
        return result
        
    except UseCaseException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in search_tracks_get: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Lines of code**: ~40 lines
**Duplication**: High (exception handling duplicated)

---

### After (Refactored with Helpers)

```python
# api/v1/controllers/search_controller.py (AFTER)

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import logging

from ...dependencies import get_search_tracks_use_case
from ..schemas.requests import SearchTracksRequest
from ..schemas.responses import SearchResponse, ErrorResponse
from ....application.use_cases.analysis import SearchTracksUseCase
from ...utils.controller_helpers import handle_use_case_exceptions

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
    return await use_case.execute(
        request.query,
        limit=request.limit,
        offset=request.offset
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
    return await use_case.execute(q, limit=limit, offset=offset)
```

**Lines of code**: ~30 lines
**Duplication**: None
**Improvement**: 25% reduction in code, 100% elimination of duplication

---

## Example 3: Using Track Resolution Helper

### Before (Original Code)

```python
# api/music_api.py (BEFORE)

@router.post("/analyze", response_model=dict)
async def analyze_track(request: TrackAnalysisRequest):
    try:
        # Obtener track_id
        track_id = request.track_id
        
        if not track_id and request.track_name:
            # Buscar la canción
            tracks = spotify_service.search_track(request.track_name, limit=1)
            if not tracks:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró la canción: {request.track_name}"
                )
            track_id = tracks[0]["id"]
        
        # Continue with analysis...
```

---

### After (Refactored with Helper)

```python
# api/music_api.py (AFTER)

from ..utils.track_helpers import resolve_track_id

@router.post("/analyze", response_model=dict)
async def analyze_track(request: TrackAnalysisRequest):
    try:
        spotify_service = get_spotify_service()
        
        # Resolve track_id using helper
        track_id = resolve_track_id(
            request.track_id,
            request.track_name,
            spotify_service
        )
        
        # Continue with analysis...
```

**Improvement**: 
- Eliminates 8 lines of duplicated search logic
- Consistent error handling
- Reusable across all endpoints

---

## Summary of Benefits

### Code Reduction
- **Analysis Controller**: 60 → 35 lines (42% reduction)
- **Search Controller**: 40 → 30 lines (25% reduction)
- **Overall**: Estimated 200-300 lines reduced across codebase

### Consistency
- All endpoints handle exceptions the same way
- All responses follow the same format
- Track resolution logic is consistent

### Maintainability
- Update error handling in one place (`controller_helpers.py`)
- Update response format in one place (`response_helpers.py`)
- Update track resolution in one place (`track_helpers.py`)

### Testability
- Helper functions can be unit tested independently
- Controllers become simpler and easier to test
- Mock helpers for integration tests

### Developer Experience
- Less boilerplate code to write
- Clear patterns to follow
- Better IDE support with type hints








