"""
Record Storage - Usage Examples

This file demonstrates how to use the RecordStorage class with various examples.
"""

from pathlib import Path
from .record_storage import RecordStorage


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    storage = RecordStorage("data/example_storage.json")
    
    initial_records = [
        {"id": "1", "name": "Alice", "email": "alice@example.com", "age": 30},
        {"id": "2", "name": "Bob", "email": "bob@example.com", "age": 25},
        {"id": "3", "name": "Charlie", "email": "charlie@example.com", "age": 35}
    ]
    
    storage.write(initial_records)
    print(f"✓ Written {len(initial_records)} records")
    
    records = storage.read()
    print(f"✓ Read {len(records)} records")
    for record in records:
        print(f"  - {record['name']} ({record['id']})")
    
    print()


def example_update_record():
    """Example of updating a record."""
    print("=" * 60)
    print("Example 2: Updating a Record")
    print("=" * 60)
    
    storage = RecordStorage("data/example_storage.json")
    
    print("Before update:")
    record = storage.get("1")
    if record:
        print(f"  {record}")
    
    updates = {"age": 31, "city": "New York"}
    success = storage.update("1", updates)
    
    if success:
        print("✓ Record updated successfully")
        print("After update:")
        updated_record = storage.get("1")
        if updated_record:
            print(f"  {updated_record}")
    else:
        print("✗ Record not found")
    
    print()


def example_add_record():
    """Example of adding a new record."""
    print("=" * 60)
    print("Example 3: Adding a New Record")
    print("=" * 60)
    
    storage = RecordStorage("data/example_storage.json")
    
    new_record = {
        "id": "4",
        "name": "David",
        "email": "david@example.com",
        "age": 28,
        "city": "Los Angeles"
    }
    
    success = storage.add(new_record)
    if success:
        print(f"✓ Added new record: {new_record['name']}")
        print(f"  Total records: {len(storage.read())}")
    else:
        print("✗ Record with this ID already exists")
    
    print()


def example_delete_record():
    """Example of deleting a record."""
    print("=" * 60)
    print("Example 4: Deleting a Record")
    print("=" * 60)
    
    storage = RecordStorage("data/example_storage.json")
    
    print(f"Records before deletion: {len(storage.read())}")
    
    success = storage.delete("2")
    if success:
        print("✓ Record deleted successfully")
        print(f"Records after deletion: {len(storage.read())}")
    else:
        print("✗ Record not found")
    
    print()


def example_error_handling():
    """Example of error handling."""
    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)
    
    storage = RecordStorage("data/example_storage.json")
    
    try:
        storage.write("not a list")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.update("", {"age": 30})
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.add({"name": "No ID"})
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    print()


def example_complex_workflow():
    """Example of a complex workflow."""
    print("=" * 60)
    print("Example 6: Complex Workflow")
    print("=" * 60)
    
    storage = RecordStorage("data/workflow_storage.json")
    
    users = [
        {"id": "user1", "name": "Alice", "role": "admin", "active": True},
        {"id": "user2", "name": "Bob", "role": "user", "active": True},
        {"id": "user3", "name": "Charlie", "role": "user", "active": False}
    ]
    
    storage.write(users)
    print(f"✓ Initialized with {len(users)} users")
    
    storage.update("user1", {"last_login": "2024-01-15"})
    print("✓ Updated user1 with last_login")
    
    storage.update("user2", {"active": False})
    print("✓ Deactivated user2")
    
    storage.add({"id": "user4", "name": "David", "role": "user", "active": True})
    print("✓ Added new user4")
    
    active_users = [u for u in storage.read() if u.get("active", False)]
    print(f"✓ Active users: {len(active_users)}")
    
    for user in active_users:
        print(f"  - {user['name']} ({user['id']})")
    
    print()


def example_data_persistence():
    """Example showing data persistence across instances."""
    print("=" * 60)
    print("Example 7: Data Persistence")
    print("=" * 60)
    
    file_path = "data/persistence_storage.json"
    
    storage1 = RecordStorage(file_path)
    storage1.write([
        {"id": "1", "data": "First instance"}
    ])
    print("✓ Written data from first instance")
    
    storage2 = RecordStorage(file_path)
    records = storage2.read()
    print(f"✓ Read data from second instance: {len(records)} records")
    print(f"  Data: {records[0]['data']}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Record Storage Usage Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_update_record()
        example_add_record()
        example_delete_record()
        example_error_handling()
        example_complex_workflow()
        example_data_persistence()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()

