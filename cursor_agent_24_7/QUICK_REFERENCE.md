# RecordStorage - Quick Reference Guide

## 🚀 One-Minute Start

```python
from record_storage import RecordStorage

# Create storage
storage = RecordStorage("data.json")

# Write records
storage.write([
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
])

# Read records
records = storage.read()

# Update a record
storage.update("1", {"age": 31, "city": "New York"})
```

## 📋 Method Cheat Sheet

### `read() -> List[Dict[str, Any]]`
Read all records from file.
```python
records = storage.read()
```

### `write(records: List[Dict[str, Any]]) -> bool`
Write records to file (replaces all).
```python
success = storage.write([{"id": "1", "name": "Alice"}])
```

### `update(record_id: str, updates: Dict[str, Any]) -> bool`
Update a specific record by ID.
```python
success = storage.update("1", {"age": 31})
```

## ⚠️ Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `TypeError: records must be a list` | Passing non-list to `write()` | Use `storage.write([...])` not `storage.write(...)` |
| `ValueError: record_id must be a non-empty string` | Empty or invalid ID | Use `storage.update("valid_id", {...})` |
| `TypeError: updates must be a dictionary` | Passing non-dict to `update()` | Use `storage.update("1", {...})` not `storage.update("1", ...)` |
| `RuntimeError: Invalid JSON` | Corrupted file | File will be auto-initialized on next write |

## ✅ Best Practices Checklist

- [ ] Always use context managers (✅ Already implemented)
- [ ] Validate inputs before processing (✅ Already implemented)
- [ ] Handle exceptions appropriately (✅ Already implemented)
- [ ] Use type hints (✅ Already implemented)
- [ ] Preserve record IDs during updates (✅ Already implemented)

## 🔍 Common Patterns

### Pattern 1: Check if record exists
```python
records = storage.read()
exists = any(r.get("id") == "1" for r in records)
```

### Pattern 2: Find record by ID
```python
records = storage.read()
record = next((r for r in records if r.get("id") == "1"), None)
```

### Pattern 3: Filter records
```python
records = storage.read()
filtered = [r for r in records if r.get("status") == "active"]
```

### Pattern 4: Add new record
```python
records = storage.read()
records.append({"id": "3", "name": "Charlie"})
storage.write(records)
```

### Pattern 5: Delete record
```python
records = storage.read()
records = [r for r in records if r.get("id") != "1"]
storage.write(records)
```

## 📊 Performance Tips

- **Read operations**: Fast, cached in memory during session
- **Write operations**: Atomic (complete or nothing)
- **Update operations**: Reads all, updates one, writes all
- **Large datasets**: Consider batching updates

## 🛡️ Safety Features

✅ **Automatic file closure** - Context managers ensure files always close  
✅ **Input validation** - Type checking prevents errors  
✅ **ID preservation** - Record IDs cannot be accidentally changed  
✅ **Error handling** - Clear error messages for debugging  
✅ **UTF-8 encoding** - Supports international characters  

## 📁 File Structure

```
{
  "records": [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
  ]
}
```

## 🎯 Quick Troubleshooting

**Q: Records not updating?**  
A: Check that you're using `records[i].update()` not `record = updates`

**Q: File not found error?**  
A: File is auto-created on first write

**Q: JSON decode error?**  
A: File might be corrupted, will auto-initialize on next write

**Q: ID changed after update?**  
A: IDs are automatically preserved, even if update tries to change them

## 📚 Related Files

- `README_RECORD_STORAGE.md` - Full documentation
- `example_usage.py` - Basic examples
- `advanced_examples.py` - Advanced patterns
- `test_record_storage.py` - Test suite


