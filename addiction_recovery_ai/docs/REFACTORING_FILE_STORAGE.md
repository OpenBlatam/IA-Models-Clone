# File Storage Refactoring Guide

## Overview
This document shows the refactoring of a file-based storage class to follow Python best practices.

## Issues Addressed

### 1. Context Managers (`with` statement)
**Before (Problematic):**
```python
def write(self, data):
    f = open(self.file_path, 'w')
    json.dump(data, f)
    f.close()  # May not execute if exception occurs
```

**After (Refactored):**
```python
def write(self, data: List[Dict[str, Any]]) -> None:
    try:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise IOError(f"Failed to write to file {self.file_path}: {str(e)}")
```

### 2. Indentation Issues
**Before (Problematic):**
```python
def read(self):
    if os.path.exists(self.file_path):
    data = []
    f = open(self.file_path, 'r')
    data = json.load(f)
    f.close()
    return data
```

**After (Refactored):**
```python
def read(self) -> List[Dict[str, Any]]:
    if not os.path.exists(self.file_path):
        return []
    
    try:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("File does not contain a valid list")
        
        return data
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(...)
```

### 3. Update Function Errors
**Before (Problematic):**
```python
def update(self, record_id, updates):
    records = self.read()
    for record in records:
        if record['id'] == record_id:
            record.update(updates)
            break
    # Missing: Writing updated records back to file
    # Missing: Error handling
    # Missing: Validation
```

**After (Refactored):**
```python
def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
    if not isinstance(record_id, str):
        raise TypeError("record_id must be a string")
    
    if not record_id:
        raise ValueError("record_id cannot be empty")
    
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")
    
    try:
        records = self.read()
        
        found = False
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            
            if record.get('id') == record_id:
                records[i].update(updates)
                found = True
                break
        
        if found:
            self.write(records)  # Properly write back to file
            return True
        
        return False
    except (IOError, json.JSONDecodeError, ValueError) as e:
        raise IOError(f"Failed to update record: {str(e)}")
```

## Key Improvements

### 1. Context Managers
- All file operations use `with` statements
- Automatic file closing even if exceptions occur
- Prevents resource leaks

### 2. Error Handling
- Type validation for all inputs
- Proper exception handling with meaningful messages
- Handles edge cases (file not found, invalid JSON, etc.)

### 3. Code Quality
- Type hints for better IDE support and documentation
- Proper indentation throughout
- Clear docstrings
- Input validation before processing

### 4. Functionality
- `update` method properly writes records back to file
- Safe handling of missing files
- Validation of data structures

## Usage Example

```python
from utils.file_storage import FileStorage

# Initialize storage
storage = FileStorage("data/records.json")

# Write initial data
storage.write([
    {"id": "1", "name": "John", "age": 30},
    {"id": "2", "name": "Jane", "age": 25}
])

# Read data
records = storage.read()
print(records)

# Update a record
success = storage.update("1", {"age": 31})
if success:
    print("Record updated successfully")

# Add a new record
storage.add({"id": "3", "name": "Bob", "age": 28})

# Get a specific record
record = storage.get("2")
print(record)

# Delete a record
storage.delete("3")
```

## Best Practices Applied

1. **Context Managers**: Always use `with` for file operations
2. **Error Handling**: Comprehensive try-except blocks
3. **Type Hints**: Full type annotations for clarity
4. **Input Validation**: Validate all inputs before processing
5. **Documentation**: Clear docstrings for all methods
6. **Resource Safety**: Automatic cleanup of resources
7. **Encoding**: Explicit UTF-8 encoding for text files
8. **Atomic Operations**: Read-modify-write pattern for updates


