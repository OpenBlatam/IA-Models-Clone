# Middleware Guide - PDF Variantes

## ✅ Recommended Middleware

### `api/middleware.py` - **USE THIS**

The canonical middleware file containing all middleware components:

```python
from api.middleware import (
    setup_cors_middleware,
    setup_trusted_host_middleware,
    rate_limit_middleware,
    request_logging_middleware,
    performance_monitoring_middleware,
    security_middleware,
)
```

### `api/config.py` - **Middleware Setup**

The centralized middleware setup:

```python
from api.config import setup_middleware
from api.main import app

# Setup all middleware
setup_middleware(app)
```

**Features:**
- Centralized middleware configuration
- Optimized middleware stack
- Security middleware
- CORS configuration
- Performance monitoring
- Request logging

## 📦 Available Middleware

### Core Middleware (from `api/middleware.py`)

#### `setup_cors_middleware`
- CORS configuration
- Cross-origin resource sharing
- Security headers

#### `setup_trusted_host_middleware`
- Trusted host validation
- Security protection
- Host header validation

#### `rate_limit_middleware`
- Rate limiting
- Request throttling
- Abuse prevention

#### `request_logging_middleware`
- Request/response logging
- Performance tracking
- Debug information

#### `performance_monitoring_middleware`
- Performance metrics
- Response time tracking
- Resource monitoring

#### `security_middleware`
- Security headers
- XSS protection
- CSRF protection

### Optimized Middleware

#### `api/optimized_middleware.py`
- **Status**: ✅ Active (Optimized)
- **Purpose**: Ultra-optimized middleware stack
- **Usage**: Used by `api/config.py` via `setup_optimized_middleware`

## ⚠️ Deprecated Middleware Files

The following middleware files are **deprecated** and should not be used for new code:

### `middleware.py` (root)
- **Status**: Deprecated
- **Reason**: Duplicate of `api/middleware.py`
- **Migration**: Use `api.middleware` or `api.config.setup_middleware`

### `performance_middleware.py`
- **Status**: Deprecated
- **Reason**: Functionality moved to `api/middleware.py`
- **Migration**: Use `api.middleware.performance_monitoring_middleware`

### `ultra_middleware.py`
- **Status**: Deprecated
- **Reason**: Functionality moved to `api/middleware.py`
- **Migration**: Use `api.middleware` components

### `api/enhanced_middleware.py`
- **Status**: Deprecated (if exists)
- **Reason**: Use `api/middleware.py` instead
- **Migration**: Use `api.middleware` components

## 🏗️ Middleware Structure

```
pdf_variantes/
├── api/
│   ├── middleware.py              # ✅ Canonical middleware file
│   ├── optimized_middleware.py   # ✅ Active (optimized)
│   └── config.py                 # ✅ Middleware setup
├── middleware.py                  # ⚠️ Deprecated
├── performance_middleware.py      # ⚠️ Deprecated
└── ultra_middleware.py            # ⚠️ Deprecated
```

## 📝 Usage Examples

### Setting Up Middleware

```python
from api.main import create_application
from api.config import setup_middleware

app = create_application()
setup_middleware(app)
```

### Using Individual Middleware Components

```python
from api.middleware import setup_cors_middleware, setup_trusted_host_middleware
from fastapi import FastAPI

app = FastAPI()

# Setup specific middleware
setup_cors_middleware(app)
setup_trusted_host_middleware(app)
```

### Creating Custom Middleware

```python
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Your middleware logic
        response = await call_next(request)
        return response

app.add_middleware(CustomMiddleware)
```

## 🔄 Migration Guide

### From `middleware.py` (root)
```python
# Old
from middleware import LoggingMiddleware, RateLimitMiddleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# New
from api.config import setup_middleware
setup_middleware(app)
```

### From `performance_middleware.py`
```python
# Old
from performance_middleware import PerformanceMiddleware
app.add_middleware(PerformanceMiddleware)

# New
from api.middleware import performance_monitoring_middleware
# or
from api.config import setup_middleware
setup_middleware(app)
```

### From `ultra_middleware.py`
```python
# Old
from ultra_middleware import UltraFastRequestLogger
app.add_middleware(UltraFastRequestLogger)

# New
from api.middleware import request_logging_middleware
# or
from api.config import setup_middleware
setup_middleware(app)
```

## 📚 Additional Resources

- See `api/middleware.py` for all available middleware
- See `api/config.py` for middleware setup
- See `api/main.py` for app initialization
- See `REFACTORING_STATUS.md` for refactoring progress






