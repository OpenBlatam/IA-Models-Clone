# RecordStorage - Refactored Implementation

## Overview

`RecordStorage` is a robust, production-ready Python class for managing JSON-based record storage with proper error handling, type safety, and resource management.

## Features

✅ **Context Managers**: All file operations use `with` statements for automatic resource cleanup  
✅ **Type Safety**: Full type hints and input validation  
✅ **Error Handling**: Comprehensive exception handling with clear error messages  
✅ **Data Integrity**: Preserves record IDs and merges updates correctly  
✅ **UTF-8 Encoding**: Proper encoding for international character support  

## Installation

No external dependencies required beyond Python standard library:
- `json`
- `pathlib`
- `typing`

## Quick Start

```python
from record_storage import RecordStorage

# Initialize storage
storage = RecordStorage("data.json")

# Write records
records = [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
]
storage.write(records)

# Read records
all_records = storage.read()
print(f"Found {len(all_records)} records")

# Update a record
storage.update("1", {"age": 31, "city": "New York"})

# Read updated record
updated = storage.read()
alice = next(r for r in updated if r["id"] == "1")
print(alice)  # {'id': '1', 'name': 'Alice', 'age': 31, 'city': 'New York'}
```

## API Reference

### `__init__(file_path: str)`

Initialize the storage system.

**Parameters:**
- `file_path` (str): Path to the JSON file for storing records

**Raises:**
- `ValueError`: If file_path is empty or invalid

**Example:**
```python
storage = RecordStorage("my_data.json")
```

### `read() -> List[Dict[str, Any]]`

Read all records from the storage file.

**Returns:**
- `List[Dict[str, Any]]`: List of records. Returns empty list if file doesn't exist or is invalid.

**Raises:**
- `RuntimeError`: If file cannot be read or contains invalid JSON

**Example:**
```python
records = storage.read()
for record in records:
    print(record["name"])
```

### `write(records: List[Dict[str, Any]]) -> bool`

Write records to the storage file, replacing all existing records.

**Parameters:**
- `records` (List[Dict[str, Any]]): List of record dictionaries to write

**Returns:**
- `bool`: True if write was successful

**Raises:**
- `TypeError`: If records is not a list
- `ValueError`: If records contains invalid elements
- `RuntimeError`: If file cannot be written

**Example:**
```python
new_records = [
    {"id": "1", "name": "Alice"},
    {"id": "2", "name": "Bob"}
]
success = storage.write(new_records)
```

### `update(record_id: str, updates: Dict[str, Any]) -> bool`

Update a specific record by ID.

**Parameters:**
- `record_id` (str): The ID of the record to update
- `updates` (Dict[str, Any]): Dictionary of fields to update

**Returns:**
- `bool`: True if record was found and updated, False if record not found

**Raises:**
- `ValueError`: If record_id is empty or updates is not a dictionary
- `TypeError`: If argument types are incorrect
- `RuntimeError`: If file operations fail

**Example:**
```python
# Update age and add city
success = storage.update("1", {"age": 31, "city": "New York"})

if success:
    print("Record updated successfully")
else:
    print("Record not found")
```

## Error Handling

The class provides comprehensive error handling:

### Input Validation

```python
# Invalid input types
try:
    storage.write("not a list")  # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")

try:
    storage.update("1", "not a dict")  # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```

### File Operations

```python
# File read errors are handled gracefully
try:
    records = storage.read()
except RuntimeError as e:
    print(f"Failed to read: {e}")
```

### Record Updates

```python
# Updating non-existent record returns False
result = storage.update("999", {"key": "value"})
if not result:
    print("Record not found")
```

## Best Practices

### 1. Always Use Context Managers

The refactored code uses `with` statements for all file operations, ensuring files are always closed properly:

```python
# ✅ Good (automatic in RecordStorage)
with open(self.file_path, 'r') as f:
    data = json.load(f)

# ❌ Bad (manual file handling)
f = open(self.file_path, 'r')
data = json.load(f)
f.close()  # May not execute if exception occurs
```

### 2. Validate Inputs

Always validate user inputs before processing:

```python
# ✅ Good
if not isinstance(records, list):
    raise TypeError("records must be a list")

# ❌ Bad
json.dump(records, f)  # May fail with unclear error
```

### 3. Preserve Data Integrity

The `update` method preserves the record ID even if updates try to change it:

```python
# ID is always preserved
storage.update("1", {"id": "999", "name": "New Name"})
# Record still has id="1", not "999"
```

### 4. Handle Exceptions Appropriately

Use specific exception types and provide clear error messages:

```python
try:
    records = storage.read()
except json.JSONDecodeError as e:
    raise RuntimeError(f"Invalid JSON: {e}") from e
except (IOError, OSError) as e:
    raise RuntimeError(f"File error: {e}") from e
```

## Testing

Run the test suite:

```bash
python test_record_storage.py
```

The test suite covers:
- Reading empty files
- Writing and reading records
- Updating records
- Handling nonexistent records
- Input validation
- Context manager file closure
- ID preservation during updates

## File Structure

The storage file uses the following JSON structure:

```json
{
  "records": [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
  ]
}
```

## Migration from Old Code

If you're migrating from code that doesn't use context managers:

### Before (Problematic)
```python
def read(self):
    f = open(self.file_path, 'r')
    data = json.load(f)
    f.close()
    if 'records' in data:
    return data['records']  # Indentation error
    return []
```

### After (Refactored)
```python
def read(self) -> List[Dict[str, Any]]:
    if not self.file_path.exists():
        return []
    try:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict) or 'records' not in data:
            return []
        records = data.get('records', [])
        if not isinstance(records, list):
            return []
        return records
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {e}") from e
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error reading: {e}") from e
```

## Performance Considerations

- File operations are atomic (complete read/write operations)
- No unnecessary file I/O
- Efficient JSON parsing
- UTF-8 encoding for international support

## Security Notes

- Input validation prevents injection attacks
- File paths are validated
- No arbitrary code execution risks
- Safe JSON parsing with error handling

## License

This code follows Python best practices and is ready for production use.


