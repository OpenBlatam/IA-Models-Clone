# Migration Guide - Content Redundancy Detector

## Overview

This guide helps you migrate from the legacy monolithic `routers.py` to the new modular `api/routes/` structure.

## Migration Status

- ✅ **91% Complete** (90+ of 99 endpoints migrated)
- ⏳ **9 endpoints remaining** (mostly duplicates)

## Quick Reference

### Old Structure (Deprecated)
```python
from routers import router
app.include_router(router, prefix="/api/v1")
```

### New Structure (Recommended)
```python
from api.routes import api_router
app.include_router(api_router, prefix="/api/v1")
```

## Endpoint Mapping

### Core Analysis
- `/analyze` → `api/routes/analysis.py`
- `/similarity` → `api/routes/similarity.py`
- `/quality` → `api/routes/quality.py`

### System
- `/health` → `api/routes/health.py`
- `/metrics` → `api/routes/metrics.py`
- `/stats` → `api/routes/stats.py`
- `/cache/*` → `api/routes/cache.py`
- `/` → `api/routes/root.py`

### AI/ML
- `/ai/sentiment` → `api/routes/ai_sentiment.py`
- `/ai/topics` → `api/routes/ai_topics.py`
- `/ai/semantic-similarity` → `api/routes/ai_semantic.py`
- `/ai/plagiarism` → `api/routes/ai_plagiarism.py`
- `/ai/language`, `/ai/entities`, etc. → `api/routes/ai_ml.py`
- `/ai/predict/*` → `api/routes/ai_predict.py`

### Advanced Features
- `/analytics/*` → `api/routes/analytics.py`
- `/monitoring/*` → `api/routes/monitoring.py`
- `/security/*` → `api/routes/security.py`
- `/cloud/*` → `api/routes/cloud.py`
- `/automation/*` → `api/routes/automation.py`
- `/multimodal/*` → `api/routes/multimodal.py`
- `/realtime/*` → `api/routes/realtime.py`
- `/training/*` → `api/routes/training.py`
- `/batch/*` → `api/routes/batch.py`
- `/export/*` → `api/routes/export.py`
- `/webhooks/*` → `api/routes/webhooks.py`

## Benefits of Migration

1. **Better Organization**: Domain-specific modules instead of one large file
2. **Easier Maintenance**: Smaller, focused files (~100-200 lines each)
3. **Clearer Structure**: Logical grouping of related endpoints
4. **Better Scalability**: Easy to add new endpoints in appropriate modules
5. **Improved Testability**: Isolated modules are easier to test

## Backward Compatibility

- Legacy router still available at `/api/v1/legacy`
- Modular routes take precedence (registered first)
- All existing endpoints continue to work
- No breaking changes

## Next Steps

1. Update imports to use `api.routes` instead of `routers`
2. Add new endpoints to appropriate modules in `api/routes/`
3. Eventually remove dependency on `routers.py` after full verification

## Questions?

See `REFACTORING_COMPLETE_FINAL.md` for detailed refactoring information.






