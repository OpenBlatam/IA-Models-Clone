# Services Guide - PDF Variantes

## ✅ Recommended Services

### `services/pdf_service.py` - **USE THIS**

The canonical PDF processing service:

```python
from services.pdf_service import PDFVariantesService
from utils.config import get_settings

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()

# Use the service
result = await service.upload_pdf(file, request, user_id)
```

**Features:**
- Complete PDF processing service
- AI capabilities integration
- Caching support
- File management
- Variant generation
- Topic extraction
- Brainstorming
- Async/await support

### `services/` Directory Structure

The `services/` directory contains all service implementations:

```
services/
├── __init__.py              # Service exports and factory functions
├── pdf_service.py           # ✅ Canonical PDF processing service
├── collaboration_service.py  # Collaboration features
├── cache_service.py         # Caching service
├── security_service.py      # Security service
├── performance_service.py   # Performance optimization
└── monitoring_service.py    # Monitoring and analytics
```

## 📦 Available Services

### PDF Service
**File**: `services/pdf_service.py`
**Class**: `PDFVariantesService`

Main service for PDF processing:
- PDF upload and processing
- Variant generation
- Topic extraction
- Brainstorming
- AI integration

### Collaboration Service
**File**: `services/collaboration_service.py`
**Class**: `CollaborationService`

Features:
- Document sharing
- Real-time collaboration
- Comments and annotations
- Version control

### Cache Service
**File**: `services/cache_service.py`
**Class**: `CacheService`

Features:
- Response caching
- Cache invalidation
- Performance optimization

### Security Service
**File**: `services/security_service.py`
**Class**: `SecurityService`

Features:
- Authentication
- Authorization
- Security policies
- Audit logging

### Performance Service
**File**: `services/performance_service.py`
**Class**: `PerformanceService`

Features:
- Performance monitoring
- Optimization recommendations
- Resource management

### Monitoring Service
**File**: `services/monitoring_service.py`
**Classes**: `MonitoringSystem`, `AnalyticsService`, `HealthService`, `NotificationService`

Features:
- System monitoring
- Analytics and metrics
- Health checks
- Notifications

## ⚠️ Deprecated Service Files

The following files are **deprecated** and should not be used for new code:

### `services.py` (Root)
- **Status**: Deprecated
- **Reason**: Duplicate of `services/pdf_service.py`
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `core_services.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `async_services.py`
- **Status**: Deprecated
- **Reason**: Async utilities are now part of `services/pdf_service.py`
- **Migration**: Use `services.pdf_service.PDFVariantesService` for async operations

## 🏗️ Service Structure

```
pdf_variantes/
├── services/                    # ✅ Services directory
│   ├── __init__.py             # ✅ Service exports
│   ├── pdf_service.py          # ✅ Canonical PDF service
│   ├── collaboration_service.py # ✅ Collaboration service
│   ├── cache_service.py        # ✅ Cache service
│   ├── security_service.py     # ✅ Security service
│   ├── performance_service.py  # ✅ Performance service
│   └── monitoring_service.py    # ✅ Monitoring service
├── services.py                 # ⚠️ Deprecated
├── core_services.py            # ⚠️ Deprecated
└── async_services.py           # ⚠️ Deprecated
```

## 📝 Usage Examples

### Basic Service Usage

```python
from services.pdf_service import PDFVariantesService
from utils.config import get_settings
from fastapi import UploadFile

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()

# Upload PDF
upload_request = PDFUploadRequest(
    filename="document.pdf",
    extract_text=True,
    extract_images=True
)

result = await service.upload_pdf(
    file=upload_file,
    request=upload_request,
    user_id="user123"
)
```

### Using Multiple Services

```python
from services import (
    PDFVariantesService,
    CollaborationService,
    CacheService,
    SecurityService
)
from utils.config import get_settings

settings = get_settings()

# Initialize services
pdf_service = PDFVariantesService(settings)
await pdf_service.initialize()

collab_service = CollaborationService(settings)
await collab_service.initialize()

cache_service = CacheService(settings)
await cache_service.initialize()

security_service = SecurityService(settings)
await security_service.initialize()
```

### Using Service Factory Functions

```python
from services import (
    create_pdf_service,
    create_collaboration_service,
    initialize_all_services
)
from utils.config import get_settings

settings = get_settings()

# Create individual service
pdf_service = await create_pdf_service(settings)

# Initialize all services
all_services = await initialize_all_services(settings)
pdf_service = all_services["pdf_service"]
collab_service = all_services["collaboration_service"]
```

### Dependency Injection

```python
from api.dependencies import get_pdf_service, get_services
from fastapi import Depends

# Get PDF service
@app.get("/pdf/upload")
async def upload_pdf(
    service: PDFVariantesService = Depends(get_pdf_service)
):
    # Use service
    pass

# Get all services
@app.get("/admin/stats")
async def get_stats(
    services: dict = Depends(get_services)
):
    pdf_service = services["pdf_service"]
    analytics = services["analytics_service"]
    # Use services
    pass
```

## 🔄 Migration Guide

### From `services.py`
```python
# Old
from services import PDFVariantesService
service = PDFVariantesService(upload_dir=Path("./uploads"))

# New
from services.pdf_service import PDFVariantesService
from utils.config import get_settings

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()
```

### From `core_services.py`
```python
# Old
from core_services import process_pdf_completely
result = await process_pdf_completely(file_content, filename, ...)

# New
from services.pdf_service import PDFVariantesService
from utils.config import get_settings

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()
result = await service.upload_pdf(file, request, user_id)
```

### From `async_services.py`
```python
# Old
from async_services import with_timeout, parallel_extract_features
result = await with_timeout(operation, timeout=10)

# New
from services.pdf_service import PDFVariantesService
# All async operations are handled by the service
service = PDFVariantesService(settings)
await service.initialize()
# Service methods handle async operations internally
```

## 🚀 Service Lifecycle

### Initialization

```python
from services.pdf_service import PDFVariantesService
from utils.config import get_settings

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()
```

### Cleanup

```python
# Services with cleanup methods
if hasattr(service, 'cleanup'):
    await service.cleanup()
```

### Using Service Registry

```python
from services import initialize_all_services, cleanup_all_services
from utils.config import get_settings

settings = get_settings()

# Initialize all services
services = await initialize_all_services(settings)

# Use services
pdf_service = services["pdf_service"]
collab_service = services["collaboration_service"]

# Cleanup
await cleanup_all_services(services)
```

## 📚 Additional Resources

- See `services/__init__.py` for service exports
- See `api/dependencies.py` for dependency injection setup
- See `api/lifecycle.py` for service lifecycle management
- See `REFACTORING_STATUS.md` for refactoring progress






