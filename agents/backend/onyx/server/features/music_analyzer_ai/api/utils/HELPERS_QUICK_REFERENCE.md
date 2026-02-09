# Helper Functions Quick Reference

Quick reference guide for using the helper functions in controllers and routes.

## Import Statements

```python
# Exception handling
from ..utils.controller_helpers import handle_use_case_exceptions

# Response building
from ..utils.response_helpers import (
    build_analysis_response,
    build_search_response,
    build_success_response,
    build_list_response
)

# Track utilities
from ..utils.track_helpers import resolve_track_id, validate_track_id

# Service validation
from ..utils.service_result_helpers import (
    validate_service_result,
    require_success,
    require_not_none
)
```

## Common Patterns

### Pattern 1: Controller with Exception Handling

```python
from fastapi import APIRouter, Depends
from ..utils.controller_helpers import handle_use_case_exceptions
from ..utils.response_helpers import build_analysis_response

router = APIRouter()

@router.post("/analyze")
@handle_use_case_exceptions  # Handles all exceptions automatically
async def analyze_track(request: Request, use_case: UseCase = Depends(...)):
    result = await use_case.execute(...)
    return build_analysis_response(result)
```

### Pattern 2: Track ID Resolution

```python
from ..utils.track_helpers import resolve_track_id

# In your endpoint
spotify_service = get_spotify_service()
track_id = resolve_track_id(
    request.track_id,
    request.track_name,
    spotify_service
)
# track_id is now guaranteed to be valid or exception raised
```

### Pattern 3: Service Result Validation

```python
from ..utils.service_result_helpers import validate_service_result

# Validate before using
result = some_service.do_something()
validate_service_result(result, error_message="Operation failed")
# Continue with valid result...
```

### Pattern 4: Search Response

```python
from ..utils.response_helpers import build_search_response

tracks = spotify_service.search_track(query, limit)
return build_search_response(tracks, query=query)
```

### Pattern 5: List Response

```python
from ..utils.response_helpers import build_list_response

items = get_items()
return build_list_response(
    items,
    key="results",
    include_total=True,
    query=query  # Additional metadata
)
```

## Exception Handling

### Automatic (Recommended)

```python
@handle_use_case_exceptions
async def my_endpoint(...):
    # All exceptions handled automatically
    result = await use_case.execute(...)
    return result
```

**Handles**:
- `TrackNotFoundException` → 404
- `AnalysisException` → 500
- `UseCaseException` → 400
- Generic `Exception` → 500 (with logging)

### Manual (If Needed)

```python
try:
    result = await use_case.execute(...)
except TrackNotFoundException as e:
    raise HTTPException(status_code=404, detail=str(e))
# ... etc
```

## Response Building

### Analysis Response

```python
result = await use_case.execute(...)
return build_analysis_response(
    result,
    include_coaching=True  # Optional
)
```

### Search Response

```python
tracks = spotify_service.search_track(query, limit)
return build_search_response(
    tracks,
    query=query,
    total=total_count,  # Optional (for pagination)
    metadata={"source": "spotify"}  # Optional
)
```

### Generic Success Response

```python
return build_success_response(
    data={"key": "value"},
    message="Operation successful",  # Optional
    metadata={"timestamp": "..."}  # Optional
)
```

### List Response

```python
return build_list_response(
    items,
    key="items",  # Default: "items"
    include_total=True,  # Default: True
    page=1,  # Optional additional fields
    limit=20
)
```

## Track Utilities

### Resolve Track ID

```python
# From track_id or track_name
track_id = resolve_track_id(
    request.track_id,      # Optional
    request.track_name,    # Optional (one required)
    spotify_service,
    raise_on_not_found=True  # Default: True
)
```

### Validate Track ID

```python
validate_track_id(track_id)  # Raises HTTPException if invalid
```

## Service Validation

### Validate Service Result

```python
# Raises exception if invalid
validate_service_result(
    result,
    error_message="Operation failed",
    status_code=500,
    raise_on_error=True  # Default: True
)

# Or return boolean
is_valid = validate_service_result(
    result,
    raise_on_error=False
)
```

### Require Success

```python
require_success(
    result,
    error_message="Operation failed",
    status_code=400
)
```

### Require Not None

```python
require_not_none(
    value,
    error_message="Value not found",
    status_code=404
)
```

## Complete Example

```python
from fastapi import APIRouter, Depends, Query
from ..utils.controller_helpers import handle_use_case_exceptions
from ..utils.response_helpers import build_analysis_response
from ..utils.track_helpers import resolve_track_id
from ..utils.service_result_helpers import validate_service_result

router = APIRouter(prefix="/music", tags=["Music"])

@router.post("/analyze")
@handle_use_case_exceptions
async def analyze_track(
    track_id: str = None,
    track_name: str = None,
    include_coaching: bool = Query(False),
    spotify_service = Depends(get_spotify_service),
    use_case = Depends(get_analyze_use_case)
):
    # Resolve track_id
    resolved_id = resolve_track_id(track_id, track_name, spotify_service)
    
    # Execute use case
    result = await use_case.execute(
        track_id=resolved_id,
        include_coaching=include_coaching
    )
    
    # Validate result
    validate_service_result(result, error_message="Analysis failed")
    
    # Build and return response
    return build_analysis_response(result, include_coaching=include_coaching)
```

## Benefits Checklist

✅ **Less Code**: ~30-40% reduction in boilerplate
✅ **Consistent**: Same patterns everywhere
✅ **Maintainable**: Update logic in one place
✅ **Testable**: Helpers can be tested independently
✅ **Type Safe**: Full type hints support
✅ **Documented**: Clear docstrings and examples

## Troubleshooting

### Import Errors
- Make sure you're importing from the correct path
- Check that `__init__.py` files exist in utils directory

### Exception Not Caught
- Make sure decorator is applied: `@handle_use_case_exceptions`
- Check that exception type is in the handler's exception list

### Response Format Wrong
- Check helper function signature
- Verify DTO structure matches expected format

### Track Not Found
- `resolve_track_id()` raises HTTPException by default
- Set `raise_on_not_found=False` to return None instead

---

**See Also**:
- `HELPER_FUNCTIONS_ANALYSIS.md` - Full analysis
- `REFACTORING_EXAMPLE.md` - Before/after examples
- `HELPER_FUNCTIONS_SUMMARY.md` - Implementation summary








