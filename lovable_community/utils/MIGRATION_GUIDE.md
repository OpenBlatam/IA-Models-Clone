# Migration Guide - From Old Code to Refactored Code

## Overview

This guide helps you migrate from the old problematic code to the new refactored `RecordStorage` implementation.

## Key Changes

### 1. Context Managers

**Old Code:**
```python
def read(self):
    f = open(self.file_path, 'r')
    data = json.load(f)
    f.close()
    return data['records']
```

**New Code:**
```python
def read(self):
    with open(self.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('records', [])
```

**Migration:** No code changes needed - the new implementation handles this automatically.

### 2. Indentation Fixes

**Old Code:**
```python
def read(self):
    if 'records' in data:
    return data['records']  # Indentation error
    return []
```

**New Code:**
```python
def read(self):
    if 'records' in data:
        return data['records']  # Correct indentation
    return []
```

**Migration:** The new code fixes this automatically - no changes needed.

### 3. Update Method - Merging vs Replacing

**Old Code (Problematic):**
```python
def update(self, record_id, updates):
    records = self.read()
    for record in records:
        if record['id'] == record_id:
            record = updates  # ❌ Replaces entire record
            break
    self.write(records)
```

**New Code (Fixed):**
```python
def update(self, record_id, updates):
    records = self.read()
    for i, record in enumerate(records):
        if record.get('id') == record_id:
            records[i].update(updates)  # ✅ Merges updates
            break
    self.write(records)
```

**Migration Impact:** 
- ✅ **Good News**: The new code preserves existing fields automatically
- ⚠️ **Check**: If your code relied on `update()` replacing the entire record, you'll need to review that logic

### 4. Error Handling

**Old Code:**
```python
def write(self, records):
    f = open(self.file_path, 'w')
    json.dump({"records": records}, f)
    f.close()
```

**New Code:**
```python
def write(self, records):
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    try:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump({"records": records}, f, indent=2, ensure_ascii=False)
    except (IOError, OSError) as e:
        raise RuntimeError(f"Cannot write storage file: {e}") from e
```

**Migration:** Add error handling to your code:

```python
# Old way
storage.write(records)

# New way (recommended)
try:
    storage.write(records)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"File error: {e}")
```

## Step-by-Step Migration

### Step 1: Update Imports

```python
# Old
from your_module import RecordStorage

# New
from utils.record_storage import RecordStorage
```

### Step 2: Review Update Calls

Check all places where you call `update()`:

```python
# This will now merge instead of replace
storage.update("1", {"age": 31})
# Result: {"id": "1", "name": "Alice", "age": 31}  # name preserved!
```

If you need to replace the entire record, use `write()` instead:

```python
# To replace entire record
records = storage.read()
for i, r in enumerate(records):
    if r['id'] == "1":
        records[i] = {"id": "1", "age": 31}  # New record
        break
storage.write(records)
```

### Step 3: Add Error Handling

Wrap file operations in try/except:

```python
# Old
records = storage.read()

# New
try:
    records = storage.read()
except RuntimeError as e:
    logger.error(f"Failed to read: {e}")
    records = []
```

### Step 4: Update Return Value Checks

The new `write()` and `update()` methods return booleans:

```python
# Old
storage.write(records)  # No return value

# New
success = storage.write(records)
if not success:
    print("Write failed")
```

### Step 5: Test Your Code

Run your tests to ensure everything works:

```bash
python utils/record_storage_validator.py
pytest tests/test_record_storage.py
```

## Common Migration Patterns

### Pattern 1: Handling Missing Records

**Old:**
```python
records = storage.read()
for record in records:
    if record['id'] == "1":
        # update
        break
```

**New:**
```python
success = storage.update("1", {"age": 31})
if not success:
    print("Record not found")
```

### Pattern 2: Conditional Updates

**Old:**
```python
records = storage.read()
for record in records:
    if record['id'] == "1":
        if some_condition:
            record = new_data  # Replaces
        break
storage.write(records)
```

**New:**
```python
if some_condition:
    # Replace entire record
    records = storage.read()
    for i, r in enumerate(records):
        if r['id'] == "1":
            records[i] = new_data
            break
    storage.write(records)
else:
    # Merge updates
    storage.update("1", {"age": 31})
```

### Pattern 3: Error Recovery

**Old:**
```python
storage.write(records)  # May fail silently
```

**New:**
```python
try:
    storage.write(records)
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    # Handle invalid data
except RuntimeError as e:
    logger.error(f"File error: {e}")
    # Retry or use backup
```

## Breaking Changes

### 1. Return Values

- `write()` now returns `bool` (was `None`)
- `update()` now returns `bool` (was `None`)

### 2. Exceptions

- New methods raise `ValueError` for invalid inputs
- New methods raise `RuntimeError` for file operations
- Old code may not have handled these

### 3. Update Behavior

- `update()` now merges instead of replacing
- This is a **fix**, but may change behavior if you relied on replacement

## Testing Your Migration

1. **Run the validator:**
   ```bash
   python utils/record_storage_validator.py
   ```

2. **Run your existing tests:**
   ```bash
   pytest your_tests.py
   ```

3. **Check for errors:**
   - Look for `ValueError` exceptions
   - Look for `RuntimeError` exceptions
   - Verify update behavior matches expectations

## Rollback Plan

If you need to rollback:

1. Keep a backup of your old code
2. The new code is backward compatible for basic operations
3. Only update behavior changed (merging vs replacing)

## Support

If you encounter issues:

1. Check `REFACTORED_CODE_SUMMARY.md` for implementation details
2. Review `record_storage_example.py` for before/after comparisons
3. Run `record_storage_validator.py` to verify requirements
4. Check test files for usage examples

## Checklist

- [ ] Update imports
- [ ] Review all `update()` calls
- [ ] Add error handling
- [ ] Update return value checks
- [ ] Test with validator
- [ ] Run existing tests
- [ ] Verify update behavior
- [ ] Update documentation

## Success Criteria

✅ All file operations use context managers  
✅ Indentation is correct  
✅ Updates merge instead of replace  
✅ Error handling is comprehensive  
✅ All tests pass  
✅ No linter errors  


