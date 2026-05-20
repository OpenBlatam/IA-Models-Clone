# Dependencies Guide - Addiction Recovery AI

## ✅ Recommended Dependencies

### `api/dependencies/` - **USE THIS**

The canonical dependencies directory for API routes:

```python
from api.dependencies import (
    get_pagination_params,
    get_optional_auth,
    get_required_auth,
    PaginationParams,
    OptionalAuth,
    RequiredAuth
)
```

**Features:**
- Organized in `api/dependencies/` directory
- Common dependencies for API routes
- Pagination support
- Authentication support
- Type-safe with Pydantic

**Structure:**
```
api/dependencies/
├── __init__.py          # ✅ Exports common dependencies
└── common.py            # ✅ Common dependency functions
```

## 📋 Alternative Dependencies

### `dependencies.py` (Root) - Legacy Dependencies
- **Status**: ⚠️ Potentially Deprecated
- **Purpose**: Legacy dependency injection for services
- **Note**: Appears to be older version, consider using `api/dependencies/` for new code

**Current Usage:**
```python
from dependencies import (
    get_addiction_analyzer,
    get_recovery_planner,
    get_progress_tracker
)
```

**Migration:**
- For API route dependencies: Use `api/dependencies/`
- For service dependencies: May still use `dependencies.py` or migrate to service factory

## 🏗️ Dependencies Structure

```
addiction_recovery_ai/
├── api/
│   └── dependencies/          # ✅ Canonical (API dependencies)
│       ├── __init__.py
│       └── common.py
└── dependencies.py            # ⚠️ Legacy (service dependencies)
```

## 📝 Usage Examples

### API Route Dependencies (Recommended)
```python
from fastapi import APIRouter, Depends
from api.dependencies import (
    get_pagination_params,
    get_required_auth,
    PaginationParams,
    RequiredAuth
)

router = APIRouter()

@router.get("/items")
async def get_items(
    pagination: PaginationParams = Depends(get_pagination_params),
    auth: RequiredAuth = Depends(get_required_auth)
):
    # Use pagination and auth
    pass
```

### Service Dependencies (Legacy)
```python
from fastapi import Depends
from dependencies import get_addiction_analyzer

@router.post("/analyze")
async def analyze(
    analyzer = Depends(get_addiction_analyzer)
):
    # Use analyzer service
    pass
```

## 🔄 Migration Guide

### From Root `dependencies.py` to `api/dependencies/`

**For API Route Dependencies:**
```python
# Old (if used in routes)
from dependencies import get_something

# New
from api.dependencies import get_something
```

**For Service Dependencies:**
- Consider using service factory pattern
- Or continue using `dependencies.py` if it's for services, not routes

## 🎯 Quick Reference

| Location | Purpose | Status | When to Use |
|----------|---------|--------|-------------|
| `api/dependencies/` | API route dependencies | ✅ Canonical | API routes, pagination, auth |
| `dependencies.py` (root) | Service dependencies | ⚠️ Legacy | Service injection (consider migration) |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `API_GUIDE.md` for API structure
- See `services/service_factory.py` for service factory pattern






