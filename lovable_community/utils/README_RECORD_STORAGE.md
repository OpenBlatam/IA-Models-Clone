# Record Storage - Complete Guide

## Overview

The `RecordStorage` class provides a robust, file-based storage system for managing records in JSON format. It has been fully refactored to follow Python best practices with proper error handling, context managers, and input validation.

## Quick Start

```python
from utils.record_storage import RecordStorage

# Create storage instance
storage = RecordStorage("data/my_records.json")

# Write records
records = [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
]
storage.write(records)

# Read records
all_records = storage.read()

# Get specific record
record = storage.get("1")

# Update record
storage.update("1", {"age": 31})

# Add new record
storage.add({"id": "3", "name": "Charlie", "age": 35})

# Delete record
storage.delete("2")
```

## Features

### ✅ Context Managers
All file operations use `with` statements to ensure files are properly closed:

```python
# Automatically handles file closing
with open(self.file_path, 'r') as f:
    data = json.load(f)
```

### ✅ Proper Error Handling
Comprehensive error handling with specific exception types:

```python
try:
    records = storage.read()
except RuntimeError as e:
    print(f"Error reading file: {e}")
```

### ✅ Input Validation
All methods validate inputs before processing:

```python
# Raises ValueError if invalid
storage.write("not a list")  # ValueError
storage.update("", {"age": 30})  # ValueError
```

### ✅ Correct Record Updates
The `update()` method merges changes instead of replacing:

```python
# Original record
{"id": "1", "name": "Alice", "age": 30}

# Update only age
storage.update("1", {"age": 31})

# Result: {"id": "1", "name": "Alice", "age": 31}
# Name is preserved!
```

## API Reference

### `__init__(file_path: str)`
Initialize storage with a file path.

**Parameters:**
- `file_path`: Path to JSON file (created if doesn't exist)

**Raises:**
- `ValueError`: If file_path is invalid

### `read() -> List[Dict[str, Any]]`
Read all records from file.

**Returns:**
- List of record dictionaries

**Raises:**
- `RuntimeError`: If file cannot be read or contains invalid JSON

### `write(records: List[Dict[str, Any]]) -> bool`
Write records to file, replacing all existing records.

**Parameters:**
- `records`: List of record dictionaries

**Returns:**
- `True` if successful

**Raises:**
- `ValueError`: If records is not a list or contains invalid items
- `RuntimeError`: If file cannot be written

### `update(record_id: str, updates: Dict[str, Any]) -> bool`
Update a specific record by merging updates.

**Parameters:**
- `record_id`: ID of record to update
- `updates`: Dictionary of fields to update

**Returns:**
- `True` if record was found and updated, `False` otherwise

**Raises:**
- `ValueError`: If record_id or updates are invalid
- `RuntimeError`: If file operations fail

### `add(record: Dict[str, Any]) -> bool`
Add a new record to storage.

**Parameters:**
- `record`: Record dictionary (must contain 'id' field)

**Returns:**
- `True` if added, `False` if ID already exists

**Raises:**
- `ValueError`: If record is invalid or missing 'id'

### `delete(record_id: str) -> bool`
Delete a record by ID.

**Parameters:**
- `record_id`: ID of record to delete

**Returns:**
- `True` if deleted, `False` if not found

**Raises:**
- `ValueError`: If record_id is invalid

### `get(record_id: str) -> Optional[Dict[str, Any]]`
Get a specific record by ID.

**Parameters:**
- `record_id`: ID of record to retrieve

**Returns:**
- Record dictionary or `None` if not found

**Raises:**
- `ValueError`: If record_id is invalid

## Advanced Features

For advanced features like batch operations, filtering, and backup/restore, see `record_storage_advanced.py`:

```python
from utils.record_storage_advanced import AdvancedRecordStorage

storage = AdvancedRecordStorage("data/advanced.json")

# Batch operations
storage.batch_add([record1, record2, record3])
storage.batch_update([{"id": "1", "age": 31}])
storage.batch_delete(["1", "2"])

# Query operations
active_users = storage.find_all(active=True)
user = storage.find_one(name="Alice")
count = storage.count(role="admin")

# Filter with custom predicate
adults = storage.filter(lambda r: r.get("age", 0) >= 18)

# Backup and restore
backup_path = storage.backup()
storage.restore(backup_path)

# Schema validation
schema = {
    "required": ["id", "name"],
    "types": {"id": str, "name": str, "age": int}
}
storage.add_with_validation(record, schema)
```

## Error Handling Examples

```python
from utils.record_storage import RecordStorage

storage = RecordStorage("data/example.json")

# Handle read errors
try:
    records = storage.read()
except RuntimeError as e:
    print(f"Failed to read: {e}")

# Handle write errors
try:
    storage.write(records)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Write failed: {e}")

# Handle update errors
try:
    success = storage.update("1", {"age": 31})
    if not success:
        print("Record not found")
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

## Best Practices

1. **Always use try/except** for file operations
2. **Validate data** before writing
3. **Check return values** from update/delete operations
4. **Use backups** for important data (see AdvancedRecordStorage)
5. **Handle None** when using `get()`

## Testing

Run the test suite:

```bash
pytest tests/test_record_storage.py -v
```

Run the interactive demo:

```bash
python utils/record_storage_demo.py
```

## File Structure

```
utils/
├── record_storage.py              # Main implementation
├── record_storage_advanced.py      # Advanced features
├── record_storage_example.py      # Before/after comparison
├── record_storage_usage.py        # Usage examples
├── record_storage_demo.py         # Interactive demo
└── RECORD_STORAGE_REFACTORING.md  # Refactoring documentation

tests/
└── test_record_storage.py         # Comprehensive unit tests
```

## Migration from Old Code

If you have code using the old problematic version:

1. **Replace imports**: Use the new `RecordStorage` class
2. **Update error handling**: Catch `ValueError` and `RuntimeError`
3. **Check return values**: `write()` and `update()` now return booleans
4. **Update logic**: `update()` now merges instead of replacing

## Performance Notes

- File operations are synchronous
- For large datasets, consider using a database
- Batch operations are more efficient than individual operations
- Backup operations copy the entire file

## License

Part of the Lovable Community feature set.


