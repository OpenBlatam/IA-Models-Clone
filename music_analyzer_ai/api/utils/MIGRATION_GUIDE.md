# Migration Guide: Using Helper Functions

This guide helps you migrate existing endpoints to use the new helper functions.

## Step-by-Step Migration Process

### Step 1: Identify Patterns

Look for these patterns in your code:

1. **Try-except blocks** with TrackNotFoundException, AnalysisException, etc.
2. **Manual response building** with dictionaries
3. **Track ID resolution** from track_id or track_name
4. **Criteria building** with None value filtering
5. **Manual pagination** calculations

### Step 2: Add Imports

Add the necessary imports at the top of your file:

```python
# Exception handling
from ..utils.controller_helpers import handle_use_case_exceptions

# Response building (choose what you need)
from ..utils.response_helpers import (
    build_analysis_response,
    build_search_response,
    build_success_response,
    build_list_response
)

# Request processing
from ..utils.request_helpers import build_criteria_dict

# Validation
from ..utils.validation_helpers import validate_limit, validate_track_id_format

# Pagination
from ..utils.pagination_helpers import paginate_items, build_paginated_response

# Track utilities
from ..utils.track_helpers import resolve_track_id

# Service validation
from ..utils.service_result_helpers import validate_service_result
```

### Step 3: Add Exception Handling Decorator

**Before**:
```python
@router.post("/endpoint")
async def my_endpoint(...):
    try:
        result = await use_case.execute(...)
        return result
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AnalysisException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**After**:
```python
@router.post("/endpoint")
@handle_use_case_exceptions  # Add this decorator
async def my_endpoint(...):
    result = await use_case.execute(...)
    return result  # Exceptions handled automatically
```

### Step 4: Replace Response Building

**Before**:
```python
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
```

**After**:
```python
return build_analysis_response(result, include_coaching=True)
```

### Step 5: Replace Criteria Building

**Before**:
```python
criteria = {
    "genres": request.genres,
    "moods": request.moods,
    "energy_range": request.energy_range
}
criteria = {k: v for k, v in criteria.items() if v is not None}
```

**After**:
```python
criteria = build_criteria_dict(
    genres=request.genres,
    moods=request.moods,
    energy_range=request.energy_range
)
```

### Step 6: Replace Track ID Resolution

**Before**:
```python
track_id = request.track_id
if not track_id and request.track_name:
    tracks = spotify_service.search_track(request.track_name, limit=1)
    if not tracks:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontró la canción: {request.track_name}"
        )
    track_id = tracks[0]["id"]
```

**After**:
```python
track_id = resolve_track_id(
    request.track_id,
    request.track_name,
    spotify_service
)
```

### Step 7: Replace Manual Validation

**Before**:
```python
if limit < 1 or limit > 50:
    raise HTTPException(
        status_code=400,
        detail="Limit must be between 1 and 50"
    )
```

**After**:
```python
limit = validate_limit(request.limit, min_val=1, max_val=50)
```

### Step 8: Replace Manual Pagination

**Before**:
```python
total = len(items)
page_size = 20
total_pages = (total + page_size - 1) // page_size
start = (page - 1) * page_size
end = start + page_size
paginated_items = items[start:end]

response = {
    "success": True,
    "items": paginated_items,
    "pagination": {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages
    }
}
```

**After**:
```python
items, pagination = paginate_items(all_items, page=1, page_size=20)
return build_paginated_response(items, page=1, page_size=20, total=len(all_items))
```

---

## Complete Example: Full Migration

### Before (Original Code)

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import logging

from ...dependencies import get_analyze_track_use_case
from ..schemas.requests import AnalyzeTrackRequest
from ..schemas.responses import AnalysisResponse, ErrorResponse
from ....application.use_cases.analysis import AnalyzeTrackUseCase
from ....application.exceptions import TrackNotFoundException, AnalysisException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Analysis"])

@router.post("", response_model=AnalysisResponse)
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    try:
        result = await use_case.execute(
            track_id=request.track_id,
            track_name=request.track_name,
            include_coaching=request.include_coaching
        )
        
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
```

### After (Refactored with Helpers)

```python
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

@router.post("", response_model=AnalysisResponse)
@handle_use_case_exceptions
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(
        track_id=request.track_id,
        track_name=request.track_name,
        include_coaching=request.include_coaching
    )
    
    return build_analysis_response(result, include_coaching=request.include_coaching)
```

**Result**: 60 lines → 25 lines (58% reduction)

---

## Common Migration Patterns

### Pattern 1: Simple Endpoint

**Before**:
```python
@router.get("/endpoint")
async def endpoint():
    try:
        result = await use_case.execute()
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**After**:
```python
@router.get("/endpoint")
@handle_use_case_exceptions
async def endpoint():
    result = await use_case.execute()
    return build_success_response(result)
```

### Pattern 2: Endpoint with Parameters

**Before**:
```python
@router.get("/endpoint")
async def endpoint(limit: int = Query(10)):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Invalid limit")
    try:
        result = await use_case.execute(limit=limit)
        return {"success": True, "items": result, "total": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**After**:
```python
@router.get("/endpoint")
@handle_use_case_exceptions
async def endpoint(limit: int = Query(10)):
    limit = validate_limit(limit, min_val=1, max_val=100)
    result = await use_case.execute(limit=limit)
    return build_list_response(result, key="items")
```

### Pattern 3: Endpoint with Pagination

**Before**:
```python
@router.get("/endpoint")
async def endpoint(page: int = Query(1), page_size: int = Query(20)):
    total = len(all_items)
    start = (page - 1) * page_size
    end = start + page_size
    items = all_items[start:end]
    return {
        "success": True,
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    }
```

**After**:
```python
@router.get("/endpoint")
@handle_use_case_exceptions
async def endpoint(page: int = Query(1), page_size: int = Query(20)):
    page, page_size = validate_pagination_params(page, page_size)
    items, pagination = paginate_items(all_items, page, page_size)
    return build_paginated_response(items, page, page_size, total=len(all_items))
```

---

## Testing After Migration

1. **Test Normal Flow**: Verify endpoint works with valid inputs
2. **Test Error Cases**: Verify exceptions are handled correctly
3. **Test Edge Cases**: Test with None values, empty lists, etc.
4. **Test Response Format**: Verify response structure matches expected format
5. **Integration Tests**: Run full integration tests

---

## Troubleshooting

### Issue: Exception not being caught

**Solution**: Make sure `@handle_use_case_exceptions` decorator is applied **before** the route decorator:

```python
@router.post("/endpoint")  # Route decorator first
@handle_use_case_exceptions  # Then exception handler
async def endpoint(...):
    ...
```

### Issue: Response format doesn't match

**Solution**: Check that you're using the correct helper function. Use `build_analysis_response()` for analysis endpoints, `build_search_response()` for search endpoints, etc.

### Issue: Validation errors

**Solution**: Make sure you're using validation helpers before using the values:

```python
limit = validate_limit(request.limit, min_val=1, max_val=50)  # Validate first
result = await use_case.execute(limit=limit)  # Then use
```

### Issue: Track not found

**Solution**: `resolve_track_id()` raises HTTPException by default. If you want to handle it differently:

```python
track_id = resolve_track_id(
    request.track_id,
    request.track_name,
    spotify_service,
    raise_on_not_found=False  # Returns None instead of raising
)
if not track_id:
    # Handle not found case
```

---

## Benefits Checklist

After migration, you should see:

- ✅ Less code (30-50% reduction)
- ✅ No duplicated exception handling
- ✅ Consistent response formats
- ✅ Easier to maintain
- ✅ Better testability
- ✅ Clearer code intent

---

## Need Help?

- Check `HELPERS_QUICK_REFERENCE.md` for quick examples
- Check `REFACTORING_EXAMPLE.md` for before/after comparisons
- Check `HELPER_FUNCTIONS_ANALYSIS.md` for detailed explanations

---

**Happy Migrating!** 🚀








