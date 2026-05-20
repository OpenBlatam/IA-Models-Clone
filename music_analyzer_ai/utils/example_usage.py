"""
Example Usage of Refactored File Storage
=========================================

This file demonstrates how to use the refactored FileStorage class
with proper error handling and best practices.
"""

from file_storage_refactored import FileStorage
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_operations():
    """Example of basic CRUD operations."""
    print("\n=== Basic Operations Example ===")
    
    storage = FileStorage("data/example_records.json")
    
    # Write records
    record1 = {
        "id": "record_1",
        "name": "First Record",
        "value": 100,
        "timestamp": "2024-01-01T00:00:00"
    }
    
    record2 = {
        "id": "record_2",
        "name": "Second Record",
        "value": 200,
        "timestamp": "2024-01-02T00:00:00"
    }
    
    try:
        storage.write(record1)
        storage.write(record2)
        print("✓ Records written successfully")
    except Exception as e:
        print(f"✗ Error writing records: {e}")
        return
    
    # Read all records
    try:
        all_records = storage.read_all()
        print(f"✓ Total records: {len(all_records)}")
    except Exception as e:
        print(f"✗ Error reading records: {e}")
        return
    
    # Read specific record
    try:
        record = storage.read("record_1")
        if record:
            print(f"✓ Found record: {record['name']}")
    except Exception as e:
        print(f"✗ Error reading record: {e}")
    
    # Update record
    try:
        success = storage.update("record_1", {"value": 150, "updated": True})
        if success:
            print("✓ Record updated successfully")
        else:
            print("✗ Record not found for update")
    except Exception as e:
        print(f"✗ Error updating record: {e}")
    
    # Delete record
    try:
        success = storage.delete("record_2")
        if success:
            print("✓ Record deleted successfully")
        else:
            print("✗ Record not found for deletion")
    except Exception as e:
        print(f"✗ Error deleting record: {e}")
    
    # Get count
    try:
        count = storage.count()
        print(f"✓ Current record count: {count}")
    except Exception as e:
        print(f"✗ Error getting count: {e}")


def example_error_handling():
    """Example of proper error handling."""
    print("\n=== Error Handling Example ===")
    
    storage = FileStorage("data/error_example.json")
    
    # Test invalid input - empty record
    try:
        storage.write({})
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {e}")
    
    # Test invalid input - wrong type
    try:
        storage.write("not a dict")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {e}")
    
    # Test invalid record_id type
    try:
        storage.read(123)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {e}")
    
    # Test update with empty data
    try:
        storage.update("record_1", {})
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught expected ValueError: {e}")
    
    # Test update with non-existent record
    try:
        result = storage.update("non_existent", {"key": "value"})
        if not result:
            print("✓ Correctly returned False for non-existent record")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def example_batch_operations():
    """Example of batch operations."""
    print("\n=== Batch Operations Example ===")
    
    storage = FileStorage("data/batch_example.json")
    
    # Write multiple records
    records = [
        {"id": f"record_{i}", "name": f"Record {i}", "value": i * 10}
        for i in range(1, 6)
    ]
    
    try:
        for record in records:
            storage.write(record)
        print(f"✓ Wrote {len(records)} records")
    except Exception as e:
        print(f"✗ Error writing batch: {e}")
        return
    
    # Read all and process
    try:
        all_records = storage.read_all()
        total_value = sum(r.get("value", 0) for r in all_records)
        print(f"✓ Total value of all records: {total_value}")
    except Exception as e:
        print(f"✗ Error processing batch: {e}")
    
    # Update multiple records
    try:
        updates = [
            ("record_1", {"status": "active"}),
            ("record_2", {"status": "active"}),
            ("record_3", {"status": "inactive"})
        ]
        
        for record_id, update_data in updates:
            storage.update(record_id, update_data)
        
        print("✓ Updated multiple records")
    except Exception as e:
        print(f"✗ Error updating batch: {e}")


def example_file_management():
    """Example of file management operations."""
    print("\n=== File Management Example ===")
    
    storage = FileStorage("data/management_example.json")
    
    # Check if file exists
    exists = storage.exists()
    print(f"File exists: {exists}")
    
    # Write some data
    try:
        storage.write({"id": "test", "data": "test"})
        print("✓ Wrote test record")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Get count
    count = storage.count()
    print(f"✓ Record count: {count}")
    
    # Clear all records
    try:
        storage.clear()
        print("✓ Cleared all records")
        print(f"✓ New count: {storage.count()}")
    except Exception as e:
        print(f"✗ Error clearing: {e}")


if __name__ == "__main__":
    print("File Storage Refactored - Usage Examples")
    print("=" * 50)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Run examples
    example_basic_operations()
    example_error_handling()
    example_batch_operations()
    example_file_management()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


