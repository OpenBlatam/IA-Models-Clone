# Music Analyzer AI - Complete Refactoring Guide

## 📚 Quick Start Guide

### Using the Refactored API

#### Option 1: Use Refactored Version (Recommended)

Update `main.py`:

```python
# Change from:
from api.music_api import router

# To:
from api.music_api_refactored import router
```

#### Option 2: Keep Original (Backward Compatible)

The original `music_api.py` remains functional. You can use both versions side-by-side.

## 🏗️ Architecture Overview

### Structure

```
api/
├── base_router.py              # Base class for all routers
├── music_api_refactored.py     # Main refactored entry point
├── config/                     # Router configuration
├── decorators/                 # Router decorators
├── docs/                       # Documentation utilities
├── middleware/                 # Request middleware
├── schemas/                    # Response schemas
├── utils/                      # Utilities (formatters, error handlers)
├── validators/                 # Request validators
└── routes/                     # 28 domain routers
```

## 📖 Creating a New Router

### Step 1: Create Router File

Create `api/routes/my_router.py`:

```python
from ..base_router import BaseRouter
from ..validators import validate_track_id
from ..utils import format_track_response
from ..decorators import log_request

class MyRouter(BaseRouter):
    def __init__(self):
        super().__init__(prefix="/my", tags=["My Domain"])
        self._register_routes()
    
    def _register_routes(self):
        @self.router.get("/endpoint")
        @self.handle_exceptions
        @log_request
        async def my_endpoint(track_id: str):
            validate_track_id(track_id)
            service = self.get_service("my_service")
            result = service.get_data(track_id)
            return self.success_response(result)
```

### Step 2: Add to Main Router

Update `api/routes/main_router.py`:

```python
from .my_router import get_my_router

# In create_main_router():
routers = [
    # ... existing routers
    get_my_router(),
]
```

## 🔧 Using Utilities

### Validators

```python
from api.validators import (
    validate_track_id,
    validate_track_ids,
    validate_limit,
    validate_user_id,
    validate_search_query
)

validate_track_id(track_id)
validate_track_ids(track_ids, min_count=2, max_count=10)
validate_limit(limit, min_val=1, max_val=50)
```

### Response Formatters

```python
from api.utils import (
    format_track_response,
    format_tracks_response,
    format_paginated_response
)

# Format single track
track = format_track_response(raw_track)

# Format list
tracks = format_tracks_response(raw_tracks)

# Format paginated
response = format_paginated_response(items, page=1, limit=20, total=100)
```

### Error Handlers

```python
from api.utils import create_error_response

# In endpoint
if not found:
    return create_error_response(
        message="Resource not found",
        status_code=404,
        error_code="NOT_FOUND"
    )
```

### Base Router Methods

```python
# In router class
self.validate_track_ids(track_ids, min_count=2, max_count=10)
self.validate_limit(limit, min_val=1, max_val=50)
return self.paginated_response(items, page=1, limit=20)
return self.track_not_found(track_id)
return self.invalid_request("Invalid parameter")
```

### Decorators

```python
from api.decorators import log_request, cache_response

@self.router.get("/endpoint")
@log_request
@cache_response(ttl=300)
async def my_endpoint():
    # ...
```

### Schemas

```python
from api.schemas import SuccessResponse, TrackResponse

@router.get("/track", response_model=SuccessResponse)
async def get_track():
    track = TrackResponse(**track_data)
    return SuccessResponse(data=track, message="Success")
```

## 📊 Router List

1. **SearchRouter** - `/search`
2. **AnalysisRouter** - `/analyze`
3. **TracksRouter** - `/track`
4. **CoachingRouter** - `/coaching`
5. **ComparisonRouter** - `/compare`
6. **CacheRouter** - `/cache`
7. **ExportRouter** - `/export`
8. **HistoryRouter** - `/history`
9. **AnalyticsRouter** - `/analytics`
10. **FavoritesRouter** - `/favorites`
11. **TagsRouter** - `/tags`
12. **WebhooksRouter** - `/webhooks`
13. **AuthRouter** - `/auth`
14. **PlaylistsRouter** - `/playlists`
15. **RecommendationsRouter** - `/recommendations`
16. **DashboardRouter** - `/dashboard`
17. **NotificationsRouter** - `/notifications`
18. **TrendsRouter** - `/trends`
19. **CollaborationsRouter** - `/collaborations`
20. **AlertsRouter** - `/alerts`
21. **TemporalRouter** - `/temporal`
22. **QualityRouter** - `/quality`
23. **ArtistsRouter** - `/artists`
24. **DiscoveryRouter** - `/discovery`
25. **CoversRemixesRouter** - `/covers`
26. **RemixesRouter** - `/remixes`
27. **InstrumentationRouter** - `/instrumentation`
28. **PlaylistAnalysisRouter** - `/playlists` (analysis endpoints)
29. **PredictionsRouter** - `/predict`

## ✅ Best Practices

1. **Always use BaseRouter** - Inherit from BaseRouter for consistency
2. **Use validators** - Validate inputs before processing
3. **Use formatters** - Format responses consistently
4. **Handle errors** - Use handle_exceptions decorator
5. **Log requests** - Use log_request decorator
6. **Use schemas** - Define response schemas for type safety
7. **Get services** - Use get_service() method
8. **Return standardized responses** - Use success_response() method

## 🎯 Migration Checklist

- [ ] Review existing endpoints
- [ ] Identify domain boundaries
- [ ] Create router file
- [ ] Implement endpoints
- [ ] Add to main_router.py
- [ ] Test endpoints
- [ ] Update documentation
- [ ] Deploy and monitor

## 📝 Documentation

See:
- `REFACTORING_SUMMARY.md` - Detailed refactoring info
- `REFACTORING_COMPLETE.md` - Quick reference
- `REFACTORING_ULTIMATE.md` - Complete feature list
- `api/examples/router_example.py` - Example implementation

## 🚀 Status

✅ **Complete and Production Ready**

All infrastructure is in place and ready for use!

