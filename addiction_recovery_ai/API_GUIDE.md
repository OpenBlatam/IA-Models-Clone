# API Guide - Addiction Recovery AI

## ✅ Recommended API Structure

### `api/recovery_api_refactored.py` - **USE THIS**

The canonical API router with modular route structure:

```python
from api.recovery_api_refactored import router
# or
from api import router  # This imports from recovery_api_refactored
```

**Features:**
- Modular route structure
- Routes organized in `api/routes/` directory
- Better maintainability
- Cleaner organization

**Structure:**
```
api/
├── recovery_api_refactored.py  # ✅ Canonical router
├── routes/                     # ✅ Route modules
│   ├── assessment/
│   ├── progress/
│   ├── relapse/
│   └── support/
└── __init__.py                 # Exports router from recovery_api_refactored
```

## ⚠️ Deprecated API File

### `api/recovery_api.py` - **DEPRECATED**

- **Status**: Deprecated (kept for backward compatibility)
- **Reason**: Monolithic file with 4932+ lines
- **Migration**: Use `api/recovery_api_refactored.py` and route modules
- **Note**: All new endpoints should be added to appropriate router in `api/routes/`

**Warning:**
```python
# ⚠️ DON'T USE THIS
from api.recovery_api import router  # Deprecated

# ✅ USE THIS INSTEAD
from api.recovery_api_refactored import router
# or
from api import router
```

## 🏗️ API Structure

```
api/
├── recovery_api_refactored.py  # ✅ Canonical router
├── recovery_api.py             # ⚠️ Deprecated (4932+ lines)
├── routes/                     # ✅ Route modules
│   ├── assessment/             # Assessment endpoints
│   ├── progress/               # Progress tracking
│   ├── relapse/                # Relapse prevention
│   └── support/                # Support and coaching
├── health.py                   # Health checks
├── websocket_api.py            # WebSocket support
├── graphql_api.py              # GraphQL API (optional)
└── __init__.py                 # Exports canonical router
```

## 📝 Usage Examples

### Adding New Endpoints

**Old Way (Deprecated):**
```python
# ⚠️ DON'T add to recovery_api.py
# It's deprecated and has 4932+ lines
```

**New Way (Recommended):**
```python
# ✅ Add to appropriate route module in api/routes/
# Example: api/routes/assessment/assessment_routes.py

from fastapi import APIRouter

router = APIRouter(prefix="/api/assessment", tags=["Assessment"])

@router.post("/create")
async def create_assessment(...):
    # Your endpoint logic
    pass
```

### Importing the Router

```python
# ✅ Recommended
from api import router
# or
from api.recovery_api_refactored import router

# Include in your app
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
```

## 🔄 Migration Guide

### From `recovery_api.py` to Modular Routes

1. **Identify the endpoint category** (assessment, progress, relapse, support)
2. **Find or create the appropriate route module** in `api/routes/`
3. **Move endpoint logic** to the route module
4. **Update imports** to use the new route module
5. **Test thoroughly** before removing from `recovery_api.py`

### Example Migration

**Before (in recovery_api.py):**
```python
@router.post("/api/assessment/create")
async def create_assessment(...):
    # Logic here
    pass
```

**After (in api/routes/assessment/assessment_routes.py):**
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/assessment", tags=["Assessment"])

@router.post("/create")
async def create_assessment(...):
    # Same logic, better organized
    pass
```

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `startup_docs/QUICK_REFERENCE.md` for API endpoints reference
- See route modules in `api/routes/` for examples






