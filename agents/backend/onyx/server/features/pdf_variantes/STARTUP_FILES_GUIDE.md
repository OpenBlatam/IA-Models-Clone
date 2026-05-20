# Startup Files Guide - PDF Variantes

## ✅ Recommended Entry Points

### `run.py` - **USE THIS FOR PRODUCTION**

The canonical entry point for the PDF Variantes application:

```bash
python run.py
```

**Features:**
- Uses `api/main.py` (canonical FastAPI app)
- Production-ready configuration
- Proper logging setup
- Health checks enabled
- Standard FastAPI server

**Usage:**
```bash
# Standard run
python run.py

# With custom port
python run.py --port 8080

# With custom host
python run.py --host 0.0.0.0
```

## 📋 Alternative Startup Files

### `start.py` - System Initialization
- **Status**: ✅ Active (Different Purpose)
- **Purpose**: Full system initialization with all services
- **Use Case**: When you need the complete system with all features
- **Note**: Uses `system.py` for full initialization

```bash
python start.py
```

**When to use:**
- Testing the complete system
- Development with all features enabled
- System integration testing

### `start_api_and_debug.py` - API with Debugging Tools
- **Status**: ✅ Active (Development Tool)
- **Purpose**: Start API with enhanced debugging and monitoring
- **Use Case**: Development and debugging sessions
- **Features:**
  - Enhanced debugging tools
  - Health checker integration
  - Performance monitoring
  - Test suite runner

```bash
python start_api_and_debug.py
```

**When to use:**
- Active development
- Debugging API issues
- Performance analysis
- Testing new features

### `run_api_debug.py` - Simple API Debug
- **Status**: ✅ Active (Development Tool)
- **Purpose**: Simple API startup with debug mode enabled
- **Use Case**: Quick debugging sessions
- **Features:**
  - Debug mode enabled
  - Detailed error messages
  - Request/response logging

```bash
python run_api_debug.py
```

**When to use:**
- Quick debugging
- Testing API endpoints
- Development workflow

## ⚠️ Deprecated Entry Points

The following entry points are **deprecated** and should not be used for new code:

### `main.py`
- **Status**: Deprecated
- **Reason**: Use `run.py` instead
- **Migration**: Use `run.py` or `api/main.py` directly

### `enhanced_main.py`
- **Status**: Deprecated
- **Reason**: Features moved to `api/main.py`
- **Migration**: Use `run.py` (uses `api/main.py`)

### `optimized_main.py`
- **Status**: Deprecated
- **Reason**: Optimizations moved to middleware
- **Migration**: Use `run.py` (optimizations in middleware)

### `ultra_main.py`
- **Status**: Deprecated
- **Reason**: Features moved to `api/main.py`
- **Migration**: Use `run.py` (uses `api/main.py`)

## 🏗️ Startup Files Structure

```
pdf_variantes/
├── run.py                    # ✅ Canonical entry point (production)
├── start.py                  # ✅ Full system initialization
├── start_api_and_debug.py    # ✅ API with debugging tools
├── run_api_debug.py          # ✅ Simple API debug
├── main.py                   # ⚠️ Deprecated
├── enhanced_main.py          # ⚠️ Deprecated
├── optimized_main.py         # ⚠️ Deprecated
└── ultra_main.py            # ⚠️ Deprecated
```

## 📝 Usage Examples

### Production Deployment
```bash
# Use the canonical entry point
python run.py
```

### Development with Full System
```bash
# Start complete system with all features
python start.py
```

### Development with Debugging
```bash
# Start API with debugging tools
python start_api_and_debug.py

# Or simple debug mode
python run_api_debug.py
```

### Using uvicorn Directly
```bash
# Direct uvicorn (if you need more control)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔄 Migration Guide

### From `main.py`
```bash
# Old
python main.py

# New
python run.py
```

### From `enhanced_main.py`
```bash
# Old
python enhanced_main.py

# New
python run.py
# Enhanced features are now in api/main.py
```

### From `optimized_main.py`
```bash
# Old
python optimized_main.py

# New
python run.py
# Optimizations are now in middleware
```

### From `ultra_main.py`
```bash
# Old
python ultra_main.py

# New
python run.py
# Ultra features are now in api/main.py
```

## 🎯 Quick Reference

| File | Purpose | Status | When to Use |
|------|---------|--------|-------------|
| `run.py` | Production entry point | ✅ Canonical | Production, standard use |
| `start.py` | Full system init | ✅ Active | Complete system testing |
| `start_api_and_debug.py` | API + debugging | ✅ Active | Development, debugging |
| `run_api_debug.py` | Simple API debug | ✅ Active | Quick debugging |
| `main.py` | Basic entry point | ⚠️ Deprecated | Use `run.py` instead |
| `enhanced_main.py` | Enhanced version | ⚠️ Deprecated | Use `run.py` instead |
| `optimized_main.py` | Optimized version | ⚠️ Deprecated | Use `run.py` instead |
| `ultra_main.py` | Ultra-fast version | ⚠️ Deprecated | Use `run.py` instead |

## 📚 Additional Resources

- See `ENTRY_POINTS.md` for entry point details
- See `api/main.py` for the canonical FastAPI app
- See `REFACTORING_STATUS.md` for refactoring progress
- See `ARCHITECTURE.md` for system architecture






