# Record Storage - Quick Start Guide

## 🚀 Get Started in 30 Seconds

```python
from utils.record_storage import RecordStorage

# Create storage
storage = RecordStorage("data/my_data.json")

# Write data
storage.write([
    {"id": "1", "name": "Alice"},
    {"id": "2", "name": "Bob"}
])

# Read data
records = storage.read()

# Update record
storage.update("1", {"name": "Alice Updated"})

# Get specific record
record = storage.get("1")

# Add new record
storage.add({"id": "3", "name": "Charlie"})

# Delete record
storage.delete("2")
```

## 📋 All Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `read()` | Read all records | `List[Dict]` |
| `write(records)` | Write records (replaces all) | `bool` |
| `get(id)` | Get one record by ID | `Dict` or `None` |
| `add(record)` | Add new record | `bool` |
| `update(id, updates)` | Update record (merges) | `bool` |
| `delete(id)` | Delete record | `bool` |

## ✨ Key Features

- ✅ **Safe file handling** - Uses context managers
- ✅ **Error handling** - Comprehensive validation
- ✅ **Smart updates** - Merges instead of replacing
- ✅ **Type hints** - Full type support
- ✅ **Logging** - Built-in logging

## 🎯 Common Patterns

### Pattern 1: Check if record exists
```python
record = storage.get("1")
if record:
    print(f"Found: {record['name']}")
else:
    print("Not found")
```

### Pattern 2: Update if exists, add if not
```python
if storage.get("1"):
    storage.update("1", {"age": 31})
else:
    storage.add({"id": "1", "age": 31})
```

### Pattern 3: Error handling
```python
try:
    storage.write(records)
except ValueError as e:
    print(f"Invalid data: {e}")
except RuntimeError as e:
    print(f"File error: {e}")
```

## 🔥 Advanced Features

For batch operations, filtering, and more:

```python
from utils.record_storage_advanced import AdvancedRecordStorage

storage = AdvancedRecordStorage("data/advanced.json")

# Batch operations
storage.batch_add([r1, r2, r3])
storage.batch_update([{"id": "1", "age": 31}])

# Query
users = storage.find_all(role="admin")
count = storage.count(active=True)

# Filter
adults = storage.filter(lambda r: r.get("age", 0) >= 18)

# Backup
backup_path = storage.backup()
storage.restore(backup_path)
```

## 📚 More Resources

- **Full Documentation**: `README_RECORD_STORAGE.md`
- **Refactoring Guide**: `RECORD_STORAGE_REFACTORING.md`
- **Examples**: `record_storage_usage.py`
- **Interactive Demo**: `record_storage_demo.py`
- **Tests**: `tests/test_record_storage.py`

## 🎬 Run the Demo

```bash
python utils/record_storage_demo.py
```

## ✅ All Requirements Met

1. ✅ Context managers (`with` statements)
2. ✅ Correct indentation
3. ✅ Proper record merging in `update()`
4. ✅ Comprehensive error handling
5. ✅ Input validation
6. ✅ Type hints
7. ✅ Documentation


