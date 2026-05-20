# Entry Points Guide - PDF Variantes

## ✅ Recommended Entry Point

### `run.py` - **USE THIS**

The recommended way to start the PDF Variantes API:

```bash
python run.py
```

This file:
- Uses the canonical `api/main.py` application
- Provides proper configuration
- Includes helpful startup messages
- Supports environment variables for configuration

**Environment Variables:**
- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8000`)
- `ENVIRONMENT` - Environment mode (default: `development`)
- `LOG_LEVEL` - Logging level (default: `info`)

## 📦 Programmatic Usage

### Import the App Directly

```python
from api.main import app

# Use the app instance
# app is a FastAPI application ready to use
```

### Create Application Factory

```python
from api.main import create_application

# Create a new app instance
app = create_application()
```

## ⚠️ Deprecated Entry Points

The following files are **deprecated** and should not be used for new code:

### `main.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `run.py` instead
- **Migration**: Use `run.py` or import from `api.main`

### `enhanced_main.py`
- **Status**: Deprecated
- **Reason**: Enhanced features should be in `api/main.py` or middleware
- **Migration**: Integrate features into `api/main.py` if needed

### `optimized_main.py`
- **Status**: Deprecated
- **Reason**: Optimizations should be in middleware
- **Migration**: Move optimizations to middleware layer

### `ultra_main.py`
- **Status**: Deprecated
- **Reason**: Ultra-optimizations should be in middleware/plugins
- **Migration**: Move optimizations to middleware or plugin system

### `start.py`
- **Status**: Review needed
- **Reason**: Uses `system.py` wrapper, may be useful for system-level initialization
- **Migration**: Consider consolidating with `run.py` or keeping as system entry point

## 🏗️ Architecture

```
pdf_variantes/
├── run.py              # ✅ Recommended entry point
├── api/
│   └── main.py        # ✅ Canonical FastAPI application
├── main.py            # ⚠️ Deprecated
├── enhanced_main.py   # ⚠️ Deprecated
├── optimized_main.py  # ⚠️ Deprecated
└── ultra_main.py     # ⚠️ Deprecated
```

## 📝 Best Practices

1. **For Development**: Use `python run.py`
2. **For Production**: Use `uvicorn api.main:app --host 0.0.0.0 --port 8000`
3. **For Testing**: Import `from api.main import app`
4. **For Integration**: Import `from api.main import create_application`

## 🔄 Migration Guide

If you're currently using deprecated entry points:

### From `main.py`
```python
# Old
from main import app

# New
from api.main import app
# or
python run.py
```

### From `enhanced_main.py`
```python
# Old
from enhanced_main import app

# New
from api.main import app
# Enhanced features should be added via middleware
```

### From `optimized_main.py` or `ultra_main.py`
```python
# Old
from optimized_main import app
from ultra_main import app

# New
from api.main import app
# Optimizations should be added via middleware
```

## 🚀 Quick Start

```bash
# Start the server
python run.py

# Or with custom settings
HOST=0.0.0.0 PORT=8080 ENVIRONMENT=production python run.py
```

## 📚 Additional Resources

- See `ARCHITECTURE.md` for architectural details
- See `REFACTORING_SUMMARY.md` for refactoring history
- See `api/main.py` for the canonical application implementation






