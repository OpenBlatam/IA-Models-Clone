# Entry Points Guide - Addiction Recovery AI

## ✅ Recommended Entry Point

### `main.py` - **USE THIS FOR PRODUCTION**

The canonical entry point for the Addiction Recovery AI application:

```bash
python main.py
```

**Features:**
- Uses `core.app_factory.create_app()` (modular factory pattern)
- Production-ready configuration
- Proper lifespan management
- Centralized configuration via `config.app_config`

**Usage:**
```python
from core.app_factory import create_app
from config.app_config import get_config

app = create_app()

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.host, port=config.port)
```

## 📋 Alternative Entry Point

### `main_modular.py` - Module-Based Architecture
- **Status**: ✅ Active (Alternative Architecture)
- **Purpose**: Module loader and registry system
- **Use Case**: When you need the module-based architecture
- **Features**:
  - Module loader system
  - Module registry
  - Dynamic module loading

```bash
python main_modular.py
```

**When to use:**
- Testing module-based architecture
- Development with module system
- When you need dynamic module loading

**Note**: This uses a different architectural approach than `main.py`. Choose based on your needs.

## 🏗️ Entry Points Structure

```
addiction_recovery_ai/
├── main.py                    # ✅ Canonical entry point (production)
└── main_modular.py            # ✅ Alternative (module-based)
```

## 📝 Usage Examples

### Production Deployment
```bash
# Use the canonical entry point
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8020 --reload
```

### Development with Module System
```bash
# Use module-based entry point
python main_modular.py
```

### Using uvicorn Directly
```bash
# Direct uvicorn (if you need more control)
uvicorn main:app --host 0.0.0.0 --port 8020 --reload
```

## 🎯 Quick Reference

| File | Purpose | Status | When to Use |
|------|---------|--------|-------------|
| `main.py` | Production entry point | ✅ Canonical | Production, standard use |
| `main_modular.py` | Module-based architecture | ✅ Active | Module system needs |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `startup_docs/QUICK_REFERENCE.md` for quick start guide
- See `startup_docs/INSTALLATION_GUIDE.md` for installation details






