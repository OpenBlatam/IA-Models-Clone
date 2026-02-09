# Quick Start Guide: Using Helper Functions

**Get started using helpers in 5 minutes!**

---

## 🚀 Most Common Helpers (Copy-Paste Ready)

### 1. Cache Key Generation

```python
# BEFORE
cache_key = hashlib.md5(f"key_{var1}_{var2}".encode()).hexdigest()

# AFTER
from ..utils.cache_helpers import generate_cache_key
cache_key = generate_cache_key("key", var1, var2)
```

### 2. Response Formatting

```python
# BEFORE
return {"success": True, "data": data, "stats": stats}

# AFTER
from .response_helpers import success_response
return success_response(data=data, metadata={"stats": stats})
```

### 3. Model Serialization

```python
# BEFORE
"profile": profile.model_dump()

# AFTER
from ..utils.serialization_helpers import serialize_model
"profile": serialize_model(profile)
```

### 4. Platform Handler

```python
# BEFORE
if platform == "tiktok":
    result = await extract_tiktok(username)
elif platform == "instagram":
    result = await extract_instagram(username)

# AFTER
from ..utils.platform_helpers import execute_for_platform
result = await execute_for_platform(
    platform,
    {"tiktok": extract_tiktok, "instagram": extract_instagram},
    username
)
```

### 5. Database Upsert

```python
# BEFORE
existing = db.query(Model).filter_by(id=x).first()
if existing:
    existing.field = value
    existing.updated_at = datetime.utcnow()
else:
    db.add(Model(id=x, field=value))

# AFTER
from ..db.model_helpers import upsert_model
upsert_model(db, Model, {"id": x}, {"field": value})
```

### 6. Safe Dictionary Access

```python
# BEFORE
value = data.get("key", {}).get("nested", "default")

# AFTER
from ..utils.dict_helpers import nested_get
value = nested_get(data, "key", "nested", default="default")
```

### 7. First Non-None Value

```python
# BEFORE
value = val1 if val1 else (val2 if val2 else default)

# AFTER
from ..utils.condition_helpers import coalesce
value = coalesce(val1, val2, default="default")
```

### 8. Safe List Mapping

```python
# BEFORE
results = []
for item in items:
    try:
        results.append(process(item))
    except Exception:
        continue

# AFTER
from ..utils.collection_helpers import safe_map
results = safe_map(items, process, operation="process_items")
```

---

## 📋 Common Patterns Cheat Sheet

### API Endpoints

```python
from .response_helpers import success_response
from .exception_helpers import not_found, validation_error
from ..utils.serialization_helpers import serialize_model
from ..utils.cache_helpers import generate_cache_key
from ..utils.metrics_helpers import track_operation

@router.get("/endpoint")
async def my_endpoint(id: str):
    with track_operation("my_operation"):
        # Your code here
        return success_response(data=serialize_model(result))
```

### Database Operations

```python
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model
from ..db.query_helpers import query_one

with db_transaction(log_operation="save_data") as db:
    upsert_model(db, Model, {"id": x}, {"field": value})
    # Auto-commit, auto-rollback, auto-logging
```

### Error Handling

```python
from ..utils.error_handling_helpers import handle_errors

@handle_errors("my_operation", error_types=(ValueError,))
def my_function():
    # Your code - errors handled automatically
    pass
```

### Async Operations

```python
from ..utils.async_helpers import safe_map_async, retry_async

# Process multiple items safely
results = await safe_map_async(items, process_item, max_concurrent=5)

# With retry
result = await retry_async(fetch_data, max_attempts=3, url=url)
```

---

## 🎯 Quick Reference by Task

### I need to...

**Generate a cache key:**
```python
from ..utils.cache_helpers import generate_cache_key
key = generate_cache_key("prefix", var1, var2)
```

**Format an API response:**
```python
from .response_helpers import success_response
return success_response(data=..., metadata=...)
```

**Serialize a model:**
```python
from ..utils.serialization_helpers import serialize_model
data = serialize_model(model)
```

**Handle a database upsert:**
```python
from ..db.model_helpers import upsert_model
upsert_model(db, Model, identifier, update_data)
```

**Get nested dict value:**
```python
from ..utils.dict_helpers import nested_get
value = nested_get(data, "key1", "key2", default=None)
```

**Process list safely:**
```python
from ..utils.collection_helpers import safe_map
results = safe_map(items, func, operation="name")
```

**Get first non-None:**
```python
from ..utils.condition_helpers import coalesce
value = coalesce(val1, val2, default="default")
```

**Execute for platform:**
```python
from ..utils.platform_helpers import execute_for_platform
result = await execute_for_platform(platform, handlers, *args)
```

**Track metrics:**
```python
from ..utils.metrics_helpers import track_operation
with track_operation("name", tags={...}):
    # code
```

**Handle errors:**
```python
from ..utils.error_handling_helpers import handle_errors
@handle_errors("operation")
def my_func():
    pass
```

---

## 🔄 Migration Pattern

For any repetitive code:

1. **Identify the pattern**
2. **Find the helper** (check HELPERS_SUMMARY.md)
3. **Import the helper**
4. **Replace the code**
5. **Test**

Example:

```python
# Step 1: Identify
cache_key = hashlib.md5(f"key_{x}".encode()).hexdigest()

# Step 2: Find helper (cache_helpers.generate_cache_key)
# Step 3: Import
from ..utils.cache_helpers import generate_cache_key

# Step 4: Replace
cache_key = generate_cache_key("key", x)

# Step 5: Test
```

---

## 📚 Full Documentation

- **All helpers:** [HELPERS_SUMMARY.md](./HELPERS_SUMMARY.md)
- **Examples:** [REAL_CODE_REFACTORING.py](./REAL_CODE_REFACTORING.py)
- **Complete guide:** [ULTIMATE_REFACTORING_SUMMARY.md](./ULTIMATE_REFACTORING_SUMMARY.md)
- **Index:** [REFACTORING_INDEX.md](./REFACTORING_INDEX.md)

---

## ✅ Checklist: First Helper Usage

- [ ] Import the helper
- [ ] Replace old code
- [ ] Test functionality
- [ ] Verify no errors
- [ ] Commit changes

**That's it! You're using helpers! 🎉**








