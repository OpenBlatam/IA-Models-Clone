# Exceptions Guide - PDF Variantes

## ✅ Recommended Exceptions

### `api/exceptions.py` - **USE THIS**

The canonical exceptions file containing all custom exceptions:

```python
from api.exceptions import (
    BaseAPIException,
    ValidationError,
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
    InternalServerError,
)
```

### `api/handlers/exceptions.py` - **Exception Handlers**

The centralized exception handlers:

```python
from api.handlers.exceptions import setup_exception_handlers
from api.main import app

# Setup exception handlers
setup_exception_handlers(app)
```

**Features:**
- Centralized exception handling
- Consistent error responses
- Proper error codes
- Frontend-friendly format
- Request ID tracking

## 📦 Available Exceptions

### Core Exceptions (from `api/exceptions.py`)

#### `BaseAPIException`
- Base exception for all API exceptions
- Includes status code, detail, error code, and metadata
- Extends FastAPI's `HTTPException`

#### `ValidationError`
- Status: 422
- Error Code: `VALIDATION_ERROR`
- For validation failures

#### `NotFoundError`
- Status: 404
- Error Code: `NOT_FOUND`
- For resource not found

#### `ConflictError`
- Status: 409
- Error Code: `CONFLICT`
- For resource conflicts

#### `UnauthorizedError`
- Status: 401
- Error Code: `UNAUTHORIZED`
- For authentication failures

#### `ForbiddenError`
- Status: 403
- Error Code: `FORBIDDEN`
- For authorization failures

#### `RateLimitError`
- Status: 429
- Error Code: `RATE_LIMIT_EXCEEDED`
- For rate limit violations

#### `InternalServerError`
- Status: 500
- Error Code: `INTERNAL_SERVER_ERROR`
- For unexpected errors

## ⚠️ Deprecated Exception Files

The following exception files are **deprecated** and should not be used for new code:

### `exceptions.py` (root)
- **Status**: Deprecated
- **Reason**: Duplicate of `api/exceptions.py`
- **Migration**: Use `api.exceptions` instead

**Old exceptions in `exceptions.py`:**
- `PDFVariantesError` → Use `BaseAPIException`
- `PDFNotFoundError` → Use `NotFoundError`
- `InvalidFileError` → Use `ValidationError`
- `RateLimitError` → Use `api.exceptions.RateLimitError`

## 🏗️ Exception Structure

```
pdf_variantes/
├── api/
│   ├── exceptions.py              # ✅ Canonical exceptions file
│   └── handlers/
│       └── exceptions.py         # ✅ Exception handlers
└── exceptions.py                  # ⚠️ Deprecated
```

## 📝 Usage Examples

### Raising Exceptions

```python
from api.exceptions import NotFoundError, ValidationError

# Raise not found error
if not pdf_document:
    raise NotFoundError(resource_type="PDF", resource_id=pdf_id)

# Raise validation error
if not is_valid_file(file):
    raise ValidationError(
        detail="Invalid file format",
        field="file",
        metadata={"allowed_formats": ["pdf"]}
    )
```

### Setting Up Exception Handlers

```python
from api.main import create_application
from api.handlers.exceptions import setup_exception_handlers

app = create_application()
setup_exception_handlers(app)
```

### Custom Exception Handler

```python
from api.exceptions import BaseAPIException
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(BaseAPIException)
async def custom_exception_handler(request: Request, exc: BaseAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "metadata": exc.metadata
            }
        }
    )
```

## 🔄 Migration Guide

### From `exceptions.py` (root)

```python
# Old
from exceptions import PDFNotFoundError, InvalidFileError, RateLimitError

try:
    pdf = get_pdf(id)
except PDFNotFoundError:
    # handle error
    pass

# New
from api.exceptions import NotFoundError, ValidationError, RateLimitError

try:
    pdf = get_pdf(id)
except NotFoundError:
    # handle error
    pass
```

### Exception Mapping

| Old Exception | New Exception |
|--------------|---------------|
| `PDFVariantesError` | `BaseAPIException` |
| `PDFNotFoundError` | `NotFoundError(resource_type="PDF", resource_id=id)` |
| `InvalidFileError` | `ValidationError(detail="...", field="file")` |
| `RateLimitError` | `RateLimitError(detail="...")` |

## 📚 Additional Resources

- See `api/exceptions.py` for all available exceptions
- See `api/handlers/exceptions.py` for exception handlers
- See `api/main.py` for app initialization
- See `REFACTORING_STATUS.md` for refactoring progress






