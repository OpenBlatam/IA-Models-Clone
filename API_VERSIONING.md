# 🔄 Guía de Versionado de API - Blatam Academy Features

## 📋 Estrategias de Versionado

### Estrategia 1: URL Path Versioning

```
GET /v1/api/query
GET /v2/api/query
```

**Ventajas**: Simple, claro
**Desventajas**: URLs más largas

### Estrategia 2: Header Versioning

```
GET /api/query
Headers:
  API-Version: v1
  API-Version: v2
```

**Ventajas**: URLs limpias
**Desventajas**: Menos visible

### Estrategia 3: Query Parameter

```
GET /api/query?version=v1
GET /api/query?version=v2
```

**Ventajas**: Flexible
**Desventajas**: Puede ser confuso

## 🎯 Implementación Recomendada

### FastAPI Implementation

```python
from fastapi import APIRouter, Header
from typing import Optional

# Router para v1
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/api/query")
async def query_v1(request: dict):
    """V1 API - Legacy implementation."""
    # Implementación v1
    pass

# Router para v2
v2_router = APIRouter(prefix="/v2")

@v2_router.post("/api/query")
async def query_v2(request: dict):
    """V2 API - Enhanced with caching."""
    # Implementación v2 con mejoras
    pass

# Router principal
main_router = APIRouter()
main_router.include_router(v1_router)
main_router.include_router(v2_router)
```

### Header-based Versioning

```python
from fastapi import Header, HTTPException

async def get_api_version(
    api_version: Optional[str] = Header(None, alias="API-Version")
) -> str:
    """Obtener versión de API desde header."""
    if api_version is None:
        api_version = "v1"  # Default
    
    if api_version not in ["v1", "v2"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {api_version}"
        )
    
    return api_version

@app.post("/api/query")
async def query(request: dict, version: str = Depends(get_api_version)):
    """Query endpoint con versionado."""
    if version == "v1":
        return await query_v1_implementation(request)
    elif version == "v2":
        return await query_v2_implementation(request)
```

## 📊 Semántica de Versionado

### Semantic Versioning para APIs

```
v1.0.0 - Initial release
v1.1.0 - New features (backward compatible)
v1.1.1 - Bug fixes
v2.0.0 - Breaking changes
```

### Cambios por Tipo

#### Patch (v1.0.0 → v1.0.1)
- Bug fixes
- Mejoras de performance
- Sin cambios de API

#### Minor (v1.0.0 → v1.1.0)
- Nuevos endpoints
- Nuevos campos opcionales
- Backward compatible

#### Major (v1.0.0 → v2.0.0)
- Breaking changes
- Endpoints removidos
- Cambios en formato de respuesta

## 🔄 Migración entre Versiones

### Deprecation Policy

```python
from fastapi import Depends
from datetime import datetime, timedelta

DEPRECATION_WARNING_DAYS = 180  # 6 meses

@v1_router.post("/api/query")
async def query_v1(request: dict):
    """V1 API - Deprecated.
    
    This endpoint is deprecated and will be removed on 2024-07-01.
    Please migrate to v2 API.
    """
    # Agregar header de deprecation
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2024-07-01T00:00:00Z"
    response.headers["Link"] = '</v2/api/query>; rel="successor-version"'
    
    # Implementación v1
    return await query_v1_implementation(request)
```

### Migration Guide Endpoint

```python
@app.get("/api/migration-guide")
async def migration_guide(from_version: str, to_version: str):
    """Guía de migración entre versiones."""
    guides = {
        ("v1", "v2"): {
            "breaking_changes": [
                "Response format changed",
                "New required field 'priority'"
            ],
            "new_features": [
                "Improved caching",
                "Batch processing support"
            ],
            "migration_steps": [
                "1. Update API version header",
                "2. Add 'priority' field to requests",
                "3. Update response parsing"
            ]
        }
    }
    
    key = (from_version, to_version)
    if key in guides:
        return guides[key]
    else:
        raise HTTPException(404, "Migration guide not found")
```

## 📝 Changelog de API

### Estructura de Changelog

```markdown
# API Changelog

## [2.0.0] - 2024-01-15

### Breaking Changes
- Response format changed from object to array
- Removed `/api/legacy` endpoint

### Added
- New `/api/batch` endpoint
- Caching support in responses

### Deprecated
- `/api/query` (v1) - use v2 instead

## [1.1.0] - 2024-01-01

### Added
- New `priority` field in request
- Batch processing support

### Fixed
- Memory leak in cache
```

## ✅ Versioning Checklist

### Pre-Release
- [ ] Version number decidido
- [ ] Breaking changes documentados
- [ ] Migration guide creado
- [ ] Deprecation warnings agregados
- [ ] Tests para nueva versión
- [ ] Changelog actualizado

### Post-Release
- [ ] Documentación actualizada
- [ ] Clients notificados
- [ ] Deprecation timeline comunicado
- [ ] Monitoring de uso de versión antigua

---

**Más información:**
- [API Reference](API_REFERENCE.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Changelog](CHANGELOG.md)

