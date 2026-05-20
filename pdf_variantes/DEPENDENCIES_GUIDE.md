# Dependencies Guide - PDF Variantes

## ✅ Recommended Dependencies

### `api/dependencies.py` - **USE THIS**

The canonical dependency injection file for FastAPI:

```python
from api.dependencies import get_services, get_pdf_service, get_current_user
from fastapi import Depends

@app.get("/pdf/upload")
async def upload_pdf(
    service: PDFVariantesService = Depends(get_pdf_service),
    user: dict = Depends(get_current_user)
):
    # Use service and user
    pass
```

**Features:**
- Service dependency injection
- User authentication
- Configuration management
- File validation
- Security helpers

## 📦 Available Dependencies

### Service Dependencies

#### `get_services()`
Get all services as a dictionary:
```python
from api.dependencies import get_services
from fastapi import Depends

@app.get("/admin/stats")
async def get_stats(services: dict = Depends(get_services)):
    pdf_service = services["pdf_service"]
    analytics = services["analytics_service"]
    # Use services
    pass
```

#### `get_pdf_service()`
Get PDF processing service:
```python
from api.dependencies import get_pdf_service
from fastapi import Depends

@app.post("/pdf/upload")
async def upload_pdf(service: PDFVariantesService = Depends(get_pdf_service)):
    # Use service
    pass
```

### Authentication Dependencies

#### `get_current_user()`
Get current authenticated user:
```python
from api.dependencies import get_current_user
from fastapi import Depends

@app.get("/profile")
async def get_profile(user: dict = Depends(get_current_user)):
    # Use user information
    pass
```

#### `get_admin_user()`
Get admin user (requires admin role):
```python
from api.dependencies import get_admin_user
from fastapi import Depends

@app.delete("/admin/users/{user_id}")
async def delete_user(admin: dict = Depends(get_admin_user)):
    # Admin-only operation
    pass
```

### Validation Dependencies

#### `validate_file_size()`
Validate uploaded file size:
```python
from api.dependencies import validate_file_size
from fastapi import Depends, UploadFile

@app.post("/upload")
async def upload(
    file: UploadFile = Depends(validate_file_size(max_size_mb=10))
):
    # File size validated
    pass
```

## ⚠️ Deprecated Dependency Files

The following files are **deprecated** and should not be used for new code:

### `dependencies.py` (Root)
- **Status**: Deprecated
- **Reason**: Duplicate of `api/dependencies.py`
- **Migration**: Use `api.dependencies` instead

### `enhanced_dependencies.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `api/dependencies.py` instead
- **Migration**: Use `api.dependencies` instead

## 🏗️ Dependencies Structure

```
pdf_variantes/
├── api/
│   └── dependencies.py          # ✅ Canonical dependencies file
├── dependencies.py              # ⚠️ Deprecated
└── enhanced_dependencies.py     # ⚠️ Deprecated
```

## 📝 Usage Examples

### Basic Service Injection

```python
from api.dependencies import get_pdf_service
from services.pdf_service import PDFVariantesService
from fastapi import Depends, APIRouter

router = APIRouter()

@router.post("/pdf/upload")
async def upload_pdf(
    service: PDFVariantesService = Depends(get_pdf_service)
):
    result = await service.upload_pdf(file, request, user_id)
    return result
```

### Multiple Dependencies

```python
from api.dependencies import get_pdf_service, get_current_user, validate_file_size
from fastapi import Depends, UploadFile, APIRouter

router = APIRouter()

@router.post("/pdf/upload")
async def upload_pdf(
    file: UploadFile = Depends(validate_file_size(max_size_mb=10)),
    service: PDFVariantesService = Depends(get_pdf_service),
    user: dict = Depends(get_current_user)
):
    result = await service.upload_pdf(file, request, user["id"])
    return result
```

### All Services

```python
from api.dependencies import get_services
from fastapi import Depends, APIRouter

router = APIRouter()

@router.get("/admin/dashboard")
async def get_dashboard(services: dict = Depends(get_services)):
    pdf_service = services["pdf_service"]
    analytics = services["analytics_service"]
    monitoring = services["monitoring_system"]
    
    # Use all services
    pass
```

### Admin Only Endpoint

```python
from api.dependencies import get_admin_user
from fastapi import Depends, APIRouter

router = APIRouter()

@router.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: dict = Depends(get_admin_user)
):
    # Admin-only operation
    pass
```

## 🔄 Migration Guide

### From `dependencies.py`
```python
# Old
from dependencies import get_pdf_service, get_current_user
service = Depends(get_pdf_service)

# New
from api.dependencies import get_pdf_service, get_current_user
service = Depends(get_pdf_service)
```

### From `enhanced_dependencies.py`
```python
# Old
from enhanced_dependencies import get_config, get_current_user
config = Depends(get_config)

# New
from api.dependencies import get_services, get_current_user
# Configuration is available through services
services = Depends(get_services)
```

## 📚 Additional Resources

- See `api/dependencies.py` for all available dependencies
- See `api/README.md` for API documentation
- See `SERVICES_GUIDE.md` for service usage
- See `REFACTORING_STATUS.md` for refactoring progress






