# Before and After Comparison

## Complete Code Comparison

### ❌ BEFORE - Problematic Code

```python
class RecordStorage:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        # ❌ PROBLEMA 1: No usa context manager
        f = open(self.file_path, 'r')
        data = json.load(f)
        f.close()
        # ❌ PROBLEMA 2: Indentación incorrecta
        if 'records' in data:
        return data['records']  # Indentación incorrecta
        return []
    
    def write(self, records):
        # ❌ PROBLEMA 1: No usa context manager
        # ❌ PROBLEMA 2: Sin validación de entrada
        f = open(self.file_path, 'w')
        json.dump({"records": records}, f)
        f.close()
    
    def update(self, record_id, updates):
        # ❌ PROBLEMA 1: Indentación incorrecta
        records = self.read()
        for record in records:
            if record['id'] == record_id:
                # ❌ PROBLEMA 2: Reemplaza todo el registro
                record = updates  # Esto no funciona correctamente
                break
        # ❌ PROBLEMA 3: write fuera del contexto correcto
        self.write(records)
```

### ✅ AFTER - Refactored Code

```python
import json
from pathlib import Path
from typing import Dict, Any, List


class RecordStorage:
    def __init__(self, file_path: str):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
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
            raise RuntimeError(f"Invalid JSON in storage file: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error reading storage file: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        if not isinstance(records, list):
            raise TypeError("records must be a list")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Element at index {i} is not a valid dictionary")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error writing to storage file: {e}") from e
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Error serializing records: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
        
        if not updates:
            return False
        
        try:
            records = self.read()
            
            record_found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                if record.get('id') == record_id:
                    original_id = record.get('id')
                    records[i].update(updates)  # ✅ Fusiona actualizaciones
                    if 'id' not in records[i] or records[i].get('id') != original_id:
                        records[i]['id'] = original_id  # ✅ Preserva ID
                    record_found = True
                    break
            
            if not record_found:
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error updating record: {e}") from e
        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during update: {e}") from e
```

## Detailed Issue Breakdown

### Issue 1: Context Managers

**Problem:**
```python
# ❌ BEFORE
f = open(self.file_path, 'r')
data = json.load(f)
f.close()  # May not execute if exception occurs
```

**Solution:**
```python
# ✅ AFTER
with open(self.file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
# File automatically closed, even if exception occurs
```

**Why it matters:**
- Files are always closed, even if exceptions occur
- Prevents resource leaks
- More Pythonic and readable

### Issue 2: Indentation in `read()`

**Problem:**
```python
# ❌ BEFORE
if 'records' in data:
return data['records']  # Syntax error - incorrect indentation
return []
```

**Solution:**
```python
# ✅ AFTER
if not isinstance(data, dict) or 'records' not in data:
    return []
    
records = data.get('records', [])
if not isinstance(records, list):
    return []
    
return records
```

**Why it matters:**
- Correct Python syntax
- Better validation of data structure
- More robust error handling

### Issue 3: Record Handling in `update()`

**Problem:**
```python
# ❌ BEFORE
for record in records:
    if record['id'] == record_id:
        record = updates  # ❌ This doesn't modify the list!
        break
self.write(records)  # Records not actually updated
```

**Solution:**
```python
# ✅ AFTER
for i, record in enumerate(records):
    if record.get('id') == record_id:
        original_id = record.get('id')
        records[i].update(updates)  # ✅ Actually modifies the list element
        if 'id' not in records[i] or records[i].get('id') != original_id:
            records[i]['id'] = original_id  # ✅ Preserves ID
        record_found = True
        break

if not record_found:
    return False

with open(self.file_path, 'w', encoding='utf-8') as f:
    json.dump({"records": records}, f, indent=2, ensure_ascii=False)
```

**Why it matters:**
- Actually modifies the record in the list
- Merges updates instead of replacing
- Preserves the record ID
- Uses context manager for file writing

### Issue 4: Error Handling

**Problem:**
```python
# ❌ BEFORE
def write(self, records):
    f = open(self.file_path, 'w')
    json.dump({"records": records}, f)
    f.close()
    # No validation, no error handling
```

**Solution:**
```python
# ✅ AFTER
def write(self, records: List[Dict[str, Any]]) -> bool:
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Element at index {i} is not a valid dictionary")
    
    try:
        data = {"records": records}
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error writing to storage file: {e}") from e
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Error serializing records: {e}") from e
```

**Why it matters:**
- Validates inputs before processing
- Provides clear error messages
- Handles different types of errors appropriately
- Prevents crashes from invalid data

## Test Results Comparison

### Before (Would Fail)
```python
# These would cause errors:
storage.write("not a list")  # ❌ No validation
storage.update("1", "not a dict")  # ❌ No validation
# File might not close if exception occurs
# Records might not actually update
```

### After (Handles Gracefully)
```python
# These are properly handled:
try:
    storage.write("not a list")  # ✅ Raises TypeError with clear message
except TypeError as e:
    print(f"Error: {e}")  # "records must be a list"

try:
    storage.update("1", "not a dict")  # ✅ Raises TypeError
except TypeError as e:
    print(f"Error: {e}")  # "updates must be a dictionary"

# Files always close properly
# Records update correctly
```

## Summary of Improvements

| Aspect | Before | After |
|--------|--------|-------|
| File Operations | Manual `open()`/`close()` | Context managers (`with`) |
| Indentation | Incorrect in `read()` | Correct throughout |
| Record Updates | Doesn't work (reassigns variable) | Works (modifies list element) |
| ID Preservation | No protection | ID always preserved |
| Input Validation | None | Comprehensive |
| Error Handling | None | Comprehensive with specific exceptions |
| Type Hints | None | Full type annotations |
| Encoding | Not specified | UTF-8 explicitly |
| File Initialization | Not handled | Auto-initializes if missing |

## Key Takeaways

1. **Always use context managers** for file operations
2. **Validate inputs** before processing
3. **Use enumerate()** when modifying list elements
4. **Preserve critical fields** like IDs during updates
5. **Handle exceptions** with specific types and clear messages
6. **Add type hints** for better code documentation
7. **Test thoroughly** to catch issues early


