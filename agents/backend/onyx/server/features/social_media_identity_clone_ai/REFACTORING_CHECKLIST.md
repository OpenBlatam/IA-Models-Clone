# Refactoring Checklist: Step-by-Step Migration Guide

This checklist helps you systematically refactor the codebase using the helper functions we've created.

---

## Phase 1: Preparation (Day 1)

### Setup
- [ ] Review all helper function documentation
- [ ] Understand each helper's purpose and usage
- [ ] Set up test environment
- [ ] Create feature branch: `refactor/apply-helpers`

### Testing Setup
- [ ] Write unit tests for helper functions
- [ ] Ensure existing tests pass
- [ ] Set up integration test environment

---

## Phase 2: Core API Routes (Days 2-3)

### File: `api/routes.py`

#### Service Getters (Priority: High)
- [ ] Replace `get_analytics_service()` with `create_service_getter(AnalyticsService)`
- [ ] Replace `get_export_service()` with `create_service_getter(ExportService)`
- [ ] Replace `get_versioning_service()` with `create_service_getter(VersioningService)`
- [ ] Replace `get_batch_service()` with `create_service_getter(BatchService)`
- [ ] Replace `get_search_service()` with `create_service_getter(SearchService)`

**Expected Reduction:** ~20 lines → ~5 lines

#### extract_profile Endpoint
- [ ] Import cache helpers: `generate_cache_key`, `get_cache`
- [ ] Import response helpers: `success_response`
- [ ] Import serialization helpers: `serialize_model`
- [ ] Import metrics helpers: `track_operation`
- [ ] Import exception helpers: `validation_error`
- [ ] Replace `hashlib.md5()` with `generate_cache_key()`
- [ ] Replace `_response_cache` with `get_cache()`
- [ ] Replace response dict with `success_response()`
- [ ] Replace `profile.model_dump()` with `serialize_model(profile)`
- [ ] Replace `metrics.increment()` and `metrics.timer()` with `track_operation()`
- [ ] Replace `HTTPException` with `validation_error()`
- [ ] Test endpoint functionality
- [ ] Verify cache behavior
- [ ] Verify metrics tracking

**Expected Reduction:** ~60 lines → ~40 lines

#### build_identity Endpoint
- [ ] Import validation helpers: `validate_at_least_one`
- [ ] Import webhook helpers: `send_webhook`
- [ ] Replace validation check with `validate_at_least_one()`
- [ ] Replace metrics calls with `track_operation()`
- [ ] Replace `webhook_service.send_webhook()` with `send_webhook()`
- [ ] Replace response dict with `success_response()`
- [ ] Replace `identity.model_dump()` with `serialize_model(identity)`
- [ ] Test endpoint functionality

**Expected Reduction:** ~50 lines → ~45 lines

#### generate_content Endpoint
- [ ] Import exception helpers: `not_found`, `validation_error`
- [ ] Import validation helpers: `validate_platform`, `validate_content_type`
- [ ] Import serialization helpers: `serialize_model`
- [ ] Import webhook helpers: `send_webhook`
- [ ] Replace `HTTPException` with `not_found()` and `validation_error()`
- [ ] Replace enum validation with `validate_platform()` and `validate_content_type()`
- [ ] Replace `generated.model_dump()` with `serialize_model(generated)`
- [ ] Replace `webhook_service.send_webhook()` with `send_webhook()`
- [ ] Test endpoint functionality

**Expected Reduction:** ~80 lines → ~70 lines

#### get_identity Endpoint
- [ ] Import cache helpers: `generate_cache_key`, `get_cache`
- [ ] Import exception helpers: `not_found`
- [ ] Import serialization helpers: `serialize_model`
- [ ] Replace cache key generation
- [ ] Replace `HTTPException` with `not_found()`
- [ ] Replace `identity.model_dump()` with `serialize_model(identity)`
- [ ] Test endpoint functionality

**Expected Reduction:** ~30 lines → ~25 lines

#### get_generated_content Endpoint
- [ ] Import exception helpers: `not_found`
- [ ] Import serialization helpers: `serialize_models`
- [ ] Import response helpers: `success_response`
- [ ] Replace `HTTPException` with `not_found()`
- [ ] Replace `[c.model_dump() for c in content_list]` with `serialize_models(content_list)`
- [ ] Replace response dict with `success_response()`
- [ ] Test endpoint functionality

**Expected Reduction:** ~30 lines → ~25 lines

---

## Phase 3: Storage Service (Days 4-5)

### File: `services/storage_service.py`

#### save_identity Method
- [ ] Import database helpers: `db_transaction`, `upsert_model`
- [ ] Import datetime helpers: `utcnow`
- [ ] Replace `get_db_session()` with `db_transaction()`
- [ ] Replace update/create pattern with `upsert_model()`
- [ ] Remove manual `datetime.utcnow()` calls (handled by upsert_model)
- [ ] Remove manual `db.commit()` and logging (handled by db_transaction)
- [ ] Test method functionality
- [ ] Verify database transactions

**Expected Reduction:** ~75 lines → ~45 lines

#### _save_social_profile Method
- [ ] Import database helpers: `upsert_model`
- [ ] Import serialization helpers: `serialize_models`
- [ ] Replace update/create pattern with `upsert_model()`
- [ ] Replace `[v.model_dump() for v in profile.videos]` with `serialize_models(profile.videos)`
- [ ] Test method functionality

**Expected Reduction:** ~50 lines → ~30 lines

#### get_identity Method
- [ ] Import database helpers: `db_transaction`, `query_one`
- [ ] Replace `get_db_session()` with `db_transaction(auto_commit=False)`
- [ ] Replace query with `query_one()`
- [ ] Test method functionality

**Expected Reduction:** ~50 lines → ~40 lines

#### get_generated_content Method
- [ ] Import database helpers: `db_transaction`, `query_many`
- [ ] Replace `get_db_session()` with `db_transaction(auto_commit=False)`
- [ ] Replace query with `query_many()`
- [ ] Test method functionality

**Expected Reduction:** ~30 lines → ~25 lines

#### save_generated_content Method
- [ ] Import database helpers: `db_transaction`
- [ ] Import datetime helpers: `utcnow`
- [ ] Replace `get_db_session()` with `db_transaction()`
- [ ] Replace `datetime.utcnow()` with `utcnow()`
- [ ] Test method functionality

**Expected Reduction:** ~20 lines → ~15 lines

---

## Phase 4: Content Generator Service (Day 6)

### File: `services/content_generator.py`

#### ID Generation
- [ ] Import ID helpers: `generate_id`
- [ ] Replace all `str(uuid.uuid4())` with `generate_id("content")`
- [ ] Test ID generation

**Expected Reduction:** ~3 lines → ~1 line per occurrence (20+ occurrences)

#### Datetime Operations
- [ ] Import datetime helpers: `now`, `utcnow`
- [ ] Replace `datetime.now()` with `now()`
- [ ] Replace `datetime.utcnow()` with `utcnow()`
- [ ] Test datetime operations

**Expected Reduction:** Multiple occurrences

#### Cache Operations
- [ ] Import cache helpers: `generate_cache_key`
- [ ] Import cache manager: `ResponseCache`
- [ ] Replace `hashlib.md5()` with `generate_cache_key()`
- [ ] Replace dict cache with `ResponseCache`
- [ ] Test cache behavior

**Expected Reduction:** ~10 lines → ~5 lines per method

---

## Phase 5: Other Services (Days 7-8)

### Files to Refactor:
- [ ] `services/identity_analyzer.py`
- [ ] `services/profile_extractor.py`
- [ ] `services/webhook_service.py`
- [ ] `services/versioning_service.py`
- [ ] `services/batch_service.py`

### Patterns to Apply:
- [ ] Replace `str(uuid.uuid4())` with `generate_id()`
- [ ] Replace `datetime.now()`/`datetime.utcnow()` with helpers
- [ ] Replace cache patterns with helpers
- [ ] Replace error handling with helpers
- [ ] Replace logging patterns with helpers

---

## Phase 6: Database Services (Days 9-10)

### Files to Refactor:
- [ ] `analytics/analytics_service.py`
- [ ] `notifications/notification_service.py`
- [ ] `scheduler/scheduler_service.py`
- [ ] `ab_testing/ab_test_service.py`
- [ ] `alerts/alert_service.py`
- [ ] `collaboration/collaboration_service.py`

### Patterns to Apply:
- [ ] Replace `get_db_session()` with `db_transaction()`
- [ ] Replace update/create patterns with `upsert_model()`
- [ ] Replace queries with `query_one()` and `query_many()`
- [ ] Replace `datetime.utcnow()` with `utcnow()`
- [ ] Replace `str(uuid.uuid4())` with `generate_id()`

---

## Phase 7: Testing & Validation (Days 11-12)

### Unit Tests
- [ ] Test all refactored endpoints
- [ ] Test all refactored service methods
- [ ] Test helper functions in isolation
- [ ] Test error handling
- [ ] Test cache behavior
- [ ] Test database transactions

### Integration Tests
- [ ] Test complete workflows
- [ ] Test API endpoints end-to-end
- [ ] Test database operations
- [ ] Test webhook delivery
- [ ] Test metrics tracking

### Performance Tests
- [ ] Benchmark cache performance
- [ ] Benchmark database operations
- [ ] Compare before/after performance
- [ ] Verify no performance regressions

---

## Phase 8: Documentation & Cleanup (Day 13)

### Documentation
- [ ] Update API documentation
- [ ] Update code comments
- [ ] Create migration guide for team
- [ ] Document new patterns

### Code Cleanup
- [ ] Remove unused imports
- [ ] Remove commented code
- [ ] Run linter and fix issues
- [ ] Format code consistently

### Review
- [ ] Code review with team
- [ ] Address review comments
- [ ] Final testing
- [ ] Merge to main branch

---

## Success Metrics

### Code Quality
- [ ] Code reduction: Target 30-40% reduction in repetitive code
- [ ] Consistency: All similar patterns use same helpers
- [ ] Maintainability: Changes can be made in one place

### Functionality
- [ ] All tests pass
- [ ] No breaking changes
- [ ] Performance maintained or improved
- [ ] Error handling improved

### Team Adoption
- [ ] Team understands new patterns
- [ ] Documentation is clear
- [ ] Examples are helpful
- [ ] Future code uses helpers

---

## Rollback Plan

If issues arise:
1. [ ] Revert to previous commit
2. [ ] Document issues encountered
3. [ ] Fix helper functions if needed
4. [ ] Re-apply refactoring incrementally

---

## Notes

- Refactor incrementally, one file at a time
- Test after each major change
- Commit frequently with clear messages
- Review with team before merging
- Monitor production after deployment

---

## Estimated Timeline

- **Total Days:** 13 days
- **Total Hours:** ~65-80 hours
- **Files to Refactor:** ~30-40 files
- **Lines of Code Affected:** ~2000-3000 lines
- **Expected Reduction:** ~600-900 lines

---

## Quick Reference: Helper Imports

```python
# Cache
from ..utils.cache_helpers import generate_cache_key
from ..utils.cache_manager import get_cache

# Responses
from .response_helpers import success_response, paginated_response

# Exceptions
from .exception_helpers import not_found, validation_error, internal_error

# Validation
from ..utils.validation_helpers import validate_platform, validate_content_type, validate_at_least_one

# Serialization
from ..utils.serialization_helpers import serialize_model, serialize_models

# Metrics
from ..utils.metrics_helpers import track_operation, increment_metric

# Webhooks
from ..utils.webhook_helpers import send_webhook

# Database
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model, get_or_create
from ..db.query_helpers import query_one, query_many

# IDs & Datetime
from ..utils.id_helpers import generate_id
from ..utils.datetime_helpers import now, utcnow, utcnow_iso

# Logging
from ..utils.logging_helpers import log_operation

# Services
from ..core.service_factory import create_service_getter
```








