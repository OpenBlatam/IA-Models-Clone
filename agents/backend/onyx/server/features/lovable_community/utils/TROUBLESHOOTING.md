# Troubleshooting Guide - Record Storage

Common issues and solutions when using RecordStorage.

## Common Issues

### Issue 1: File Not Found Error

**Symptoms:**
```
RuntimeError: Cannot read storage file: [Errno 2] No such file or directory
```

**Solution:**
The file is automatically created if it doesn't exist. If you see this error:
1. Check file path permissions
2. Ensure parent directory exists and is writable
3. Check disk space

**Code:**
```python
storage = RecordStorage("data/records.json")  # Auto-creates if missing
```

### Issue 2: JSON Decode Error

**Symptoms:**
```
RuntimeError: Cannot parse storage file: Expecting value: line 1 column 1 (char 0)
```

**Solution:**
The file may be corrupted or empty. Options:
1. Delete the file and let it recreate
2. Check file contents manually
3. Use backup if available

**Code:**
```python
import os
if os.path.exists("data/records.json"):
    os.remove("data/records.json")
storage = RecordStorage("data/records.json")  # Creates fresh file
```

### Issue 3: Update Not Working

**Symptoms:**
Updates don't seem to persist or fields are lost.

**Solution:**
The new `update()` method merges instead of replaces. If you need to replace:
1. Use `write()` to replace entire record
2. Or manually read, modify, and write

**Code:**
```python
# Merging (default behavior)
storage.update("1", {"age": 31})  # Preserves other fields

# Replacing entire record
records = storage.read()
for i, r in enumerate(records):
    if r['id'] == "1":
        records[i] = {"id": "1", "age": 31}  # New record
        break
storage.write(records)
```

### Issue 4: ValueError on Invalid Input

**Symptoms:**
```
ValueError: records must be a list
ValueError: record_id must be a non-empty string
```

**Solution:**
Validate inputs before calling methods:

**Code:**
```python
# Validate before write
if isinstance(data, list):
    storage.write(data)
else:
    print("Error: data must be a list")

# Validate before update
if record_id and isinstance(record_id, str):
    storage.update(record_id, updates)
else:
    print("Error: invalid record_id")
```

### Issue 5: Permission Denied

**Symptoms:**
```
RuntimeError: Cannot write storage file: [Errno 13] Permission denied
```

**Solution:**
1. Check file/directory permissions
2. Ensure user has write access
3. Check if file is locked by another process

**Code:**
```python
from pathlib import Path

file_path = Path("data/records.json")
if file_path.exists():
    if not os.access(file_path, os.W_OK):
        print("File is not writable")
        # Fix permissions or use different location
```

### Issue 6: Records Disappearing

**Symptoms:**
Records seem to disappear after operations.

**Solution:**
This usually happens when:
1. `write()` replaces all records (by design)
2. File is being accessed by multiple processes
3. Exception occurs mid-operation

**Code:**
```python
# Always check return values
success = storage.write(records)
if not success:
    print("Write failed - records not saved")

# Use transactions/backups for critical data
backup_path = storage.backup()  # If using AdvancedRecordStorage
try:
    storage.write(records)
except Exception as e:
    storage.restore(backup_path)  # Restore on error
```

### Issue 7: ID Field Missing

**Symptoms:**
```
ValueError: record must contain an 'id' field
```

**Solution:**
All records must have an 'id' field:

**Code:**
```python
# Correct
record = {"id": "1", "name": "Alice"}
storage.add(record)

# Incorrect
record = {"name": "Alice"}  # Missing 'id'
storage.add(record)  # Raises ValueError
```

### Issue 8: Concurrent Access Issues

**Symptoms:**
Data corruption or lost updates when multiple processes access the file.

**Solution:**
1. Use file locking for concurrent access
2. Implement retry logic
3. Consider using a database for concurrent access

**Code:**
```python
import fcntl  # Unix only
import time

def safe_write(storage, records, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(storage.file_path, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock file
                storage.write(records)
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))  # Backoff
    return False
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

storage = RecordStorage("data/records.json")
# Now you'll see detailed logs
```

### Inspect File Contents

```python
import json
from pathlib import Path

file_path = Path("data/records.json")
if file_path.exists():
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(json.dumps(data, indent=2))
```

### Verify Record Structure

```python
records = storage.read()
for i, record in enumerate(records):
    if not isinstance(record, dict):
        print(f"Invalid record at index {i}: {record}")
    if 'id' not in record:
        print(f"Record at index {i} missing 'id': {record}")
```

### Check File Size

```python
from pathlib import Path

file_path = Path("data/records.json")
if file_path.exists():
    size = file_path.stat().st_size
    print(f"File size: {size} bytes")
    if size == 0:
        print("Warning: File is empty")
```

## Performance Issues

### Large Files

If working with large numbers of records (>1000), consider:
1. Using a database instead
2. Implementing pagination
3. Using batch operations (AdvancedRecordStorage)

### Slow Operations

If operations are slow:
1. Check disk I/O performance
2. Consider using SSD
3. Implement caching
4. Use batch operations

## Getting Help

1. **Check Documentation**: See `README_RECORD_STORAGE.md`
2. **Run Validator**: `python utils/record_storage_validator.py`
3. **Check Examples**: See `record_storage_usage.py`
4. **Review Tests**: See `tests/test_record_storage.py` for usage patterns

## Prevention Best Practices

1. **Always use try/except** for file operations
2. **Validate inputs** before calling methods
3. **Check return values** from write/update operations
4. **Use backups** for critical data
5. **Handle exceptions** gracefully
6. **Test with sample data** before production use


