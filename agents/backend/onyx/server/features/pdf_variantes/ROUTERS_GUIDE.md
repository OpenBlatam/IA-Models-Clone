# Routers Guide - PDF Variantes

## ✅ Recommended Routers

### `api/routes.py` - **USE THIS**

The canonical router file containing all API endpoints:

```python
from api.routes import (
    pdf_router,
    variant_router,
    topic_router,
    brainstorm_router,
    collaboration_router,
    export_router,
    analytics_router,
    search_router,
    batch_router,
    health_router
)
```

### `api/routers.py` - **Router Registration**

The centralized router registration:

```python
from api.routers import register_routers
from api.main import app

# Register all routers
register_routers(app)
```

**Features:**
- Centralized router registration
- Organized by feature/domain
- Proper prefix and tags
- All endpoints in one place

## 📦 Available Routers

### Core Routers (from `api/routes.py`)

#### `pdf_router`
- PDF upload and processing
- PDF metadata extraction
- PDF document management

#### `variant_router`
- Variant generation
- Variant management
- Variant status tracking

#### `topic_router`
- Topic extraction
- Topic analysis
- Topic management

#### `brainstorm_router`
- Brainstorming generation
- Idea management
- Brainstorm sessions

#### `collaboration_router`
- Document sharing
- Real-time collaboration
- Comments and annotations

#### `export_router`
- PDF export
- Format conversion
- Download management

#### `analytics_router`
- Analytics and metrics
- Reporting
- Dashboard data

#### `search_router`
- Document search
- Content search
- Advanced search

#### `batch_router`
- Batch processing
- Bulk operations
- Job management

#### `health_router`
- Health checks
- System status
- Monitoring endpoints

### Specialized Routers

#### `routers/analytics_router.py`
- **Status**: ✅ Active (Specialized)
- **Purpose**: Analytics-specific endpoints
- **Usage**: Registered via `api/routers.py`

#### `routers/collaboration_router.py`
- **Status**: ✅ Active (Specialized)
- **Purpose**: Collaboration-specific endpoints
- **Usage**: Registered via `api/routers.py`

## ⚠️ Deprecated Router Files

The following router files in `routers/` are **deprecated** and should not be used for new code:

### `routers/pdf_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate of `api/routes.py` pdf_router
- **Migration**: Use `api.routes.pdf_router`

### `routers/enhanced_pdf_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/routes.py` instead
- **Migration**: Use `api.routes.pdf_router`

### `routers/enhanced_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/routes.py` instead
- **Migration**: Use `api.routes` routers

### `routers/optimized_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/routes.py` instead
- **Migration**: Use `api.routes` routers

### `routers/ultra_efficient_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/routes.py` instead
- **Migration**: Use `api.routes` routers

### `routers/ultra_optimized_router.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/routes.py` instead
- **Migration**: Use `api.routes` routers

## 🏗️ Router Structure

```
pdf_variantes/
├── api/
│   ├── routes.py              # ✅ Canonical routers file
│   ├── routers.py             # ✅ Router registration
│   └── main.py                # ✅ App initialization
├── routers/
│   ├── analytics_router.py    # ✅ Active (specialized)
│   ├── collaboration_router.py # ✅ Active (specialized)
│   ├── pdf_router.py          # ⚠️ Deprecated
│   ├── enhanced_pdf_router.py # ⚠️ Deprecated
│   ├── enhanced_router.py     # ⚠️ Deprecated
│   ├── optimized_router.py   # ⚠️ Deprecated
│   ├── ultra_efficient_router.py # ⚠️ Deprecated
│   └── ultra_optimized_router.py # ⚠️ Deprecated
```

## 📝 Usage Examples

### Registering Routers

```python
from api.main import create_application
from api.routers import register_routers

app = create_application()
register_routers(app)
```

### Using Individual Routers

```python
from api.routes import pdf_router, variant_router
from fastapi import FastAPI

app = FastAPI()

# Include specific routers
app.include_router(
    pdf_router,
    prefix="/api/v1/pdf",
    tags=["PDF Processing"]
)

app.include_router(
    variant_router,
    prefix="/api/v1/variants",
    tags=["Variant Generation"]
)
```

### Creating New Router

```python
from fastapi import APIRouter, Depends
from api.dependencies import get_pdf_service

router = APIRouter(prefix="/api/v1/my-feature", tags=["My Feature"])

@router.get("/endpoint")
async def my_endpoint(
    service: PDFVariantesService = Depends(get_pdf_service)
):
    # Your endpoint logic
    pass
```

Then register it in `api/routers.py`:

```python
from .routes import my_router

def register_routers(app: FastAPI) -> None:
    # ... existing routers ...
    app.include_router(
        my_router,
        prefix="/api/v1/my-feature",
        tags=["My Feature"]
    )
```

## 🔄 Migration Guide

### From `routers/pdf_router.py`
```python
# Old
from routers.pdf_router import router
app.include_router(router)

# New
from api.routes import pdf_router
from api.routers import register_routers
register_routers(app)
```

### From `routers/enhanced_pdf_router.py`
```python
# Old
from routers.enhanced_pdf_router import router
app.include_router(router)

# New
from api.routes import pdf_router
from api.routers import register_routers
register_routers(app)
```

### From `routers/optimized_router.py`, `ultra_efficient_router.py`, etc.
```python
# Old
from routers.optimized_router import router
# or
from routers.ultra_efficient_router import router

# New
from api.routes import pdf_router
from api.routers import register_routers
register_routers(app)
```

## 📚 Additional Resources

- See `api/routes.py` for all available routers
- See `api/routers.py` for router registration
- See `api/main.py` for app initialization
- See `REFACTORING_STATUS.md` for refactoring progress






