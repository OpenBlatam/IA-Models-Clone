# Route Refactoring Guide - Decorator Patterns

## Overview

This guide shows how to refactor route decorators and query parameters using the new helper functions to reduce boilerplate and ensure consistency.

---

## Pattern 1: Repetitive Route Decorators

### Problem Identified

**Location**: All controller files

**Current Code Pattern**:
```python
@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(...):
    ...
```

The same `response_model` and `responses` patterns are repeated across multiple endpoints.

### Reasoning

1. **Boilerplate**: Every endpoint repeats the same decorator configuration
2. **Inconsistency**: Easy to forget error responses or use wrong models
3. **Maintenance**: Changing error response format requires updating many files

### Proposed Helper Functions

**File**: `api/utils/route_helpers.py`

```python
def standard_error_responses(
    *status_codes: int,
    error_model: Type = None
) -> Dict[int, Dict[str, Type]]:
    """
    Create standard error responses dictionary.
    
    Example:
        responses = standard_error_responses(404, 500, error_model=ErrorResponse)
    """
    if error_model is None:
        from ..v1.schemas.responses import ErrorResponse
        error_model = ErrorResponse
    
    return {code: {"model": error_model} for code in status_codes}
```

### Integration Example

**Before**:
```python
@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(...):
    ...
```

**After**:
```python
from ..utils.route_helpers import standard_error_responses

@router.post(
    "",
    response_model=AnalysisResponse,
    responses=standard_error_responses(404, 500)
)
@handle_use_case_exceptions
async def analyze_track(...):
    ...
```

**Benefits**:
- ✅ Less boilerplate
- ✅ Consistent error responses
- ✅ Easy to update error model in one place

---

## Pattern 2: Repetitive Query Parameters

### Problem Identified

**Location**: Multiple controllers with query parameters

**Current Code Pattern**:
```python
limit: int = Query(20, ge=1, le=50, description="Maximum number of results")
offset: int = Query(0, ge=0, description="Pagination offset")
page: int = Query(1, ge=1, description="Page number")
page_size: int = Query(20, ge=1, le=100, description="Items per page")
```

Same validation patterns repeated for limit, offset, page, page_size.

### Reasoning

1. **Repetition**: Same Query() configuration repeated everywhere
2. **Inconsistency**: Different endpoints might use different limits
3. **Maintenance**: Changing default values requires updating many places

### Proposed Helper Functions

**File**: `api/utils/route_helpers.py`

```python
def create_limit_param(
    default: int = 20,
    min_val: int = 1,
    max_val: int = 100,
    description: Optional[str] = None
) -> int:
    """Create standardized limit query parameter"""

def create_offset_param(
    default: int = 0,
    min_val: int = 0,
    description: Optional[str] = None
) -> int:
    """Create standardized offset query parameter"""

def create_page_param(
    default: int = 1,
    min_val: int = 1,
    description: Optional[str] = None
) -> int:
    """Create standardized page query parameter"""

def create_page_size_param(
    default: int = 20,
    min_val: int = 1,
    max_val: int = 100,
    description: Optional[str] = None
) -> int:
    """Create standardized page_size query parameter"""
```

### Integration Example

**Before**:
```python
limit: int = Query(20, ge=1, le=50, description="Maximum number of results")
offset: int = Query(0, ge=0, description="Pagination offset")
```

**After**:
```python
from ..utils.route_helpers import create_limit_param, create_offset_param

limit: int = create_limit_param(default=20, min_val=1, max_val=50)
offset: int = create_offset_param(default=0)
```

**Benefits**:
- ✅ Consistent validation
- ✅ Less boilerplate
- ✅ Easy to change defaults globally

---

## Pattern 3: Complete Endpoint Building

### Problem Identified

**Location**: All controllers

**Current Code Pattern**:
```python
@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(...):
    ...
```

Multiple decorators and configurations repeated.

### Reasoning

1. **Multiple Decorators**: Route decorator + exception handler + potentially logging
2. **Configuration**: response_model, responses, tags, etc. repeated
3. **Order Dependency**: Decorator order matters and is easy to get wrong

### Proposed Helper Function

**File**: `api/utils/endpoint_builder_helpers.py`

```python
def build_endpoint(
    router: APIRouter,
    method: str,
    path: str,
    handler: Callable,
    response_model: Optional[Type] = None,
    error_responses: Optional[Dict] = None,
    use_exception_handler: bool = True,
    tags: Optional[list] = None
) -> Callable:
    """
    Build complete endpoint with all patterns applied.
    """
    # Applies route decorator, exception handler, etc.
```

### Integration Example

**Before**:
```python
@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(...):
    ...
```

**After**:
```python
from ..utils.endpoint_builder_helpers import build_endpoint
from ..utils.route_helpers import standard_error_responses

async def analyze_track_impl(...):
    # Implementation
    ...

analyze_track = build_endpoint(
    router,
    "post",
    "",
    analyze_track_impl,
    response_model=AnalysisResponse,
    error_responses=standard_error_responses(404, 500),
    use_exception_handler=True
)
```

**Or using factory pattern**:
```python
from ..utils.endpoint_builder_helpers import endpoint_factory
from ..utils.route_helpers import standard_error_responses

create_endpoint = endpoint_factory(
    router,
    response_model=AnalysisResponse,
    error_responses=standard_error_responses(404, 500),
    tags=["Analysis"]
)

# Create endpoints easily
create_endpoint("post", "", analyze_track_handler)
create_endpoint("get", "/{track_id}", analyze_track_by_id_handler)
```

**Benefits**:
- ✅ Single function call for complete endpoint setup
- ✅ Consistent configuration
- ✅ No decorator order issues
- ✅ Easy to add new patterns (logging, caching, etc.)

---

## Complete Refactoring Example

### Before: Manual Route Definition

```python
from fastapi import APIRouter, Depends, Query
from ..schemas.responses import AnalysisResponse, ErrorResponse
from ..utils.controller_helpers import handle_use_case_exceptions

router = APIRouter(prefix="/analyze", tags=["Analysis"])

@router.post("", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
    return build_analysis_response(result)

@router.get("/{track_id}", response_model=AnalysisResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
@handle_use_case_exceptions
async def analyze_track_by_id(
    track_id: str,
    include_coaching: bool = Query(False, description="Include coaching analysis"),
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
    return build_analysis_response(result)
```

### After: Using Route Helpers

```python
from fastapi import APIRouter, Depends
from ..schemas.responses import AnalysisResponse, ErrorResponse
from ..utils.route_helpers import (
    standard_error_responses,
    create_query_param
)
from ..utils.endpoint_builder_helpers import endpoint_factory

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Create endpoint factory
create_endpoint = endpoint_factory(
    router,
    response_model=AnalysisResponse,
    error_responses=standard_error_responses(404, 500),
    use_exception_handler=True,
    tags=["Analysis"]
)

# Define handlers (clean, no decorators)
async def analyze_track_impl(
    request: AnalyzeTrackRequest,
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
    return build_analysis_response(result)

async def analyze_track_by_id_impl(
    track_id: str,
    include_coaching: bool = create_query_param(
        default=False,
        description="Include coaching analysis"
    ),
    use_case: AnalyzeTrackUseCase = Depends(get_analyze_track_use_case)
):
    result = await use_case.execute(...)
    return build_analysis_response(result)

# Register endpoints
create_endpoint("post", "", analyze_track_impl)
create_endpoint("get", "/{track_id}", analyze_track_by_id_impl)
```

**Benefits**:
- ✅ **Less boilerplate**: No repetitive decorators
- ✅ **Consistent**: All endpoints use same configuration
- ✅ **Maintainable**: Change configuration in one place
- ✅ **Clean handlers**: Handler functions are pure logic

---

## Usage Patterns

### Pattern 1: Standard Error Responses

```python
from ..utils.route_helpers import standard_error_responses

# Single error model
responses = standard_error_responses(404, 500)

# Custom error model
responses = standard_error_responses(400, 404, 500, error_model=CustomErrorResponse)
```

### Pattern 2: Query Parameters

```python
from ..utils.route_helpers import (
    create_limit_param,
    create_offset_param,
    create_page_param,
    create_page_size_param
)

# Standard parameters
limit: int = create_limit_param(default=20, min_val=1, max_val=50)
offset: int = create_offset_param(default=0)
page: int = create_page_param(default=1)
page_size: int = create_page_size_param(default=20, max_val=100)
```

### Pattern 3: Endpoint Factory

```python
from ..utils.endpoint_builder_helpers import endpoint_factory
from ..utils.route_helpers import standard_error_responses

# Create factory once
create_endpoint = endpoint_factory(
    router,
    response_model=AnalysisResponse,
    error_responses=standard_error_responses(404, 500),
    tags=["Analysis"]
)

# Use factory for all endpoints
create_endpoint("post", "", handler1)
create_endpoint("get", "/{id}", handler2)
create_endpoint("put", "/{id}", handler3)
```

---

## Summary

### Helpers Created
- `standard_error_responses()` - Create error response dicts
- `create_query_param()` - Create Query parameters
- `create_limit_param()` - Standardized limit parameter
- `create_offset_param()` - Standardized offset parameter
- `create_page_param()` - Standardized page parameter
- `create_page_size_param()` - Standardized page_size parameter
- `build_endpoint()` - Build complete endpoint
- `endpoint_factory()` - Create endpoint factory

### Benefits
- ✅ **Reduced boilerplate**: Less repetitive decorator code
- ✅ **Consistency**: All endpoints follow same patterns
- ✅ **Maintainability**: Change configuration in one place
- ✅ **Type safety**: Full type hints throughout

---

**Status**: ✅ Complete
**New Helpers**: 8 functions
**Impact**: Significant reduction in route definition boilerplate








