# Deprecated Files Guide - Addiction Recovery AI

This guide documents all deprecated files and their migration paths.

## ⚠️ Deprecated Files

### `api/recovery_api.py` - **DEPRECATED**

- **Status**: ⚠️ Deprecated (kept for backward compatibility)
- **Size**: 4,932+ lines (monolithic)
- **Reason**: Replaced by modular route structure
- **Replacement**: `api/recovery_api_refactored.py` and route modules in `api/routes/`

**Migration:**
```python
# ❌ DON'T USE
from api.recovery_api import router

# ✅ USE THIS INSTEAD
from api.recovery_api_refactored import router
# or
from api import router  # This imports from recovery_api_refactored
```

**Note**: This file is kept for backward compatibility but should not be used for new code. All new endpoints should be added to appropriate route modules in `api/routes/`.

## 📋 File Status Summary

| File | Status | Replacement | Notes |
|------|--------|-------------|-------|
| `api/recovery_api.py` | ⚠️ Deprecated | `api/recovery_api_refactored.py` | 4,932+ lines, monolithic |
| `main_modular.py` | ✅ Alternative | `main.py` (canonical) | Different architecture, use if needed |
| `api/health_advanced.py` | ✅ Active | `api/health.py` (standard) | AWS-specific, use when needed |

## 🔄 Migration Checklist

### From `recovery_api.py`
- [ ] Identify endpoints used from `recovery_api.py`
- [ ] Find corresponding route module in `api/routes/`
- [ ] Update imports to use route modules
- [ ] Test thoroughly
- [ ] Remove dependency on `recovery_api.py`

### Example Migration

**Before:**
```python
# Old monolithic import
from api.recovery_api import router
```

**After:**
```python
# New modular import
from api.recovery_api_refactored import router
# or
from api import router
```

## 📚 Additional Resources

- See `API_GUIDE.md` for API structure details
- See `REFACTORING_STATUS.md` for refactoring progress
- See `ENTRY_POINTS_GUIDE.md` for entry points






