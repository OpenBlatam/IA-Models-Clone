"""
Record Storage - Quick Reference Card

Copy-paste ready code snippets for common operations.
"""

from utils.record_storage import RecordStorage


# ============================================================================
# BASIC OPERATIONS
# ============================================================================

def basic_crud():
    """Basic CRUD operations."""
    storage = RecordStorage("data/basic.json")
    
    # CREATE - Add records
    storage.write([
        {"id": "1", "name": "Alice"},
        {"id": "2", "name": "Bob"}
    ])
    
    # READ - Get all records
    all_records = storage.read()
    
    # READ - Get one record
    record = storage.get("1")
    
    # UPDATE - Update a record (merges, doesn't replace)
    storage.update("1", {"age": 30})  # Preserves "name"!
    
    # DELETE - Remove a record
    storage.delete("2")
    
    # CREATE - Add new record
    storage.add({"id": "3", "name": "Charlie"})


# ============================================================================
# ERROR HANDLING PATTERNS
# ============================================================================

def error_handling_read():
    """Error handling for read operations."""
    storage = RecordStorage("data/example.json")
    
    try:
        records = storage.read()
        return records
    except RuntimeError as e:
        print(f"Error reading: {e}")
        return []


def error_handling_write():
    """Error handling for write operations."""
    storage = RecordStorage("data/example.json")
    
    try:
        storage.write([{"id": "1", "data": "test"}])
        return True
    except ValueError as e:
        print(f"Invalid input: {e}")
        return False
    except RuntimeError as e:
        print(f"File error: {e}")
        return False


def error_handling_update():
    """Error handling for update operations."""
    storage = RecordStorage("data/example.json")
    
    try:
        success = storage.update("1", {"age": 30})
        if not success:
            print("Record not found")
        return success
    except ValueError as e:
        print(f"Invalid input: {e}")
        return False
    except RuntimeError as e:
        print(f"File error: {e}")
        return False


# ============================================================================
# COMMON PATTERNS
# ============================================================================

def check_and_update():
    """Check if record exists, then update or add."""
    storage = RecordStorage("data/example.json")
    
    record_id = "1"
    updates = {"age": 31}
    
    existing = storage.get(record_id)
    if existing:
        storage.update(record_id, updates)
    else:
        storage.add({"id": record_id, **updates})


def update_or_create():
    """Update if exists, create if not."""
    storage = RecordStorage("data/example.json")
    
    record = {"id": "1", "name": "Alice", "age": 30}
    
    if storage.get(record["id"]):
        storage.update(record["id"], record)
    else:
        storage.add(record)


def batch_operations():
    """Perform multiple operations."""
    storage = RecordStorage("data/example.json")
    
    # Read all
    records = storage.read()
    
    # Update multiple
    for record in records:
        if record.get("status") == "pending":
            storage.update(record["id"], {"status": "processed"})
    
    # Add new ones
    new_records = [
        {"id": "new1", "data": "value1"},
        {"id": "new2", "data": "value2"}
    ]
    for record in new_records:
        storage.add(record)


def filter_records():
    """Filter records after reading."""
    storage = RecordStorage("data/example.json")
    
    all_records = storage.read()
    
    # Filter by condition
    active = [r for r in all_records if r.get("active", False)]
    adults = [r for r in all_records if r.get("age", 0) >= 18]
    recent = [r for r in all_records if "2024" in r.get("date", "")]
    
    return active, adults, recent


def safe_update():
    """Safe update with validation."""
    storage = RecordStorage("data/example.json")
    
    record_id = "1"
    updates = {"age": 31}
    
    # Validate before update
    if not record_id or not isinstance(updates, dict):
        return False
    
    # Check record exists
    if not storage.get(record_id):
        return False
    
    # Update
    return storage.update(record_id, updates)


# ============================================================================
# ADVANCED PATTERNS
# ============================================================================

def transaction_like():
    """Simulate transaction-like behavior."""
    storage = RecordStorage("data/example.json")
    
    try:
        # Read current state
        records = storage.read()
        
        # Make changes
        for i, record in enumerate(records):
            if record.get("id") == "1":
                records[i]["updated"] = True
        
        # Write all at once (atomic)
        storage.write(records)
        return True
    except Exception as e:
        print(f"Transaction failed: {e}")
        return False


def backup_before_write():
    """Create backup before major write."""
    import shutil
    from pathlib import Path
    
    storage = RecordStorage("data/example.json")
    backup_path = Path(str(storage.file_path) + ".backup")
    
    # Create backup
    if storage.file_path.exists():
        shutil.copy2(storage.file_path, backup_path)
    
    try:
        # Perform operation
        storage.write([{"id": "1", "data": "new"}])
    except Exception:
        # Restore on error
        if backup_path.exists():
            shutil.copy2(backup_path, storage.file_path)
        raise


def conditional_update():
    """Update only if condition met."""
    storage = RecordStorage("data/example.json")
    
    record_id = "1"
    updates = {"status": "active"}
    
    record = storage.get(record_id)
    if record and record.get("can_update", True):
        return storage.update(record_id, updates)
    return False


# ============================================================================
# QUICK SNIPPETS
# ============================================================================

SNIPPETS = {
    "basic_usage": """
from utils.record_storage import RecordStorage

storage = RecordStorage("data.json")
storage.write([{"id": "1", "name": "Alice"}])
records = storage.read()
""",
    
    "error_handling": """
try:
    storage.write(records)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"File error: {e}")
""",
    
    "update_pattern": """
# Update merges, doesn't replace
storage.update("1", {"age": 31})  # Preserves other fields
""",
    
    "check_exists": """
record = storage.get("1")
if record:
    print(f"Found: {record}")
else:
    print("Not found")
""",
    
    "add_if_not_exists": """
if not storage.get("1"):
    storage.add({"id": "1", "data": "value"})
"""
}


if __name__ == "__main__":
    print("Record Storage - Quick Reference")
    print("=" * 50)
    print("\nAvailable snippets:")
    for name in SNIPPETS:
        print(f"  - {name}")
    print("\nSee code above for implementation examples.")


