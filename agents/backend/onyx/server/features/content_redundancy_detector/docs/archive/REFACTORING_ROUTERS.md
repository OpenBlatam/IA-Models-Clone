# Routers Refactoring Plan

## Current State

### Issues Identified

1. **Monolithic `routers.py`** (88KB, ~2500 lines)
   - Contains all API endpoints in a single file
   - Difficult to maintain and test
   - Hard to navigate and understand

2. **Duplicate Router Structure**
   - `routers.py` - Monolithic router with all endpoints
   - `api/routes/` - Modular route structure (already exists but underutilized)
   - Both are being included in `app.py`, causing potential conflicts

3. **Inconsistent Patterns**
   - Some endpoints use the modular structure
   - Others are still in the monolithic file
   - Mixed import patterns

## Proposed Solution

### Phase 1: Audit and Document ✅
- [x] Identify all endpoints in `routers.py`
- [x] Map endpoints to appropriate modular route files
- [x] Document migration strategy

### Phase 2: Migrate Endpoints
- [ ] Move endpoints from `routers.py` to appropriate `api/routes/` modules
- [ ] Update imports to use new modular structure
- [ ] Ensure backward compatibility during migration

### Phase 3: Update App Registration
- [ ] Update `app.py` to use only modular routes
- [ ] Remove monolithic router dependency
- [ ] Test all endpoints still work

### Phase 4: Cleanup
- [ ] Deprecate or remove `routers.py`
- [ ] Update documentation
- [ ] Verify no broken imports

## Endpoint Mapping

### Core Analysis Endpoints
- `/analyze` → `api/routes/analysis.py` ✅ (already exists)
- `/similarity` → `api/routes/similarity.py` ✅ (already exists)
- `/quality` → `api/routes/quality.py` ✅ (already exists)

### AI/ML Endpoints
- `/ai/sentiment` → `api/routes/ai_sentiment.py` ✅ (already exists)
- `/ai/topics` → `api/routes/ai_topics.py` ✅ (already exists)
- `/ai/semantic` → `api/routes/ai_semantic.py` ✅ (already exists)
- `/ai/plagiarism` → `api/routes/ai_plagiarism.py` ✅ (already exists)
- `/ai/predict/*` → Need to create or migrate

### System Endpoints
- `/health` → `api/routes/health.py` ✅ (already exists)
- `/stats` → Need to migrate to `api/routes/metrics.py` or create new
- `/metrics` → `api/routes/metrics.py` ✅ (already exists)

### Batch Processing
- `/batch/*` → `api/routes/batch.py` ✅ (already exists)

### Webhooks
- `/webhooks/*` → `api/routes/webhooks.py` ✅ (already exists)

### Export
- `/export/*` → `api/routes/export.py` ✅ (already exists)

### Analytics
- `/analytics/*` → Need to create `api/routes/analytics.py`

### Cache Management
- `/cache/clear` → Need to migrate to appropriate module

## Implementation Strategy

1. **Incremental Migration**: Move endpoints one category at a time
2. **Backward Compatibility**: Keep old router working during migration
3. **Testing**: Verify each migrated endpoint works correctly
4. **Documentation**: Update API docs as we migrate

## Benefits

- **Better Organization**: Each route file has a single responsibility
- **Easier Testing**: Smaller modules are easier to test
- **Improved Maintainability**: Changes isolated to specific route files
- **Clearer Structure**: Easy to find and modify specific endpoints
- **Scalability**: Easy to add new endpoints in appropriate modules



