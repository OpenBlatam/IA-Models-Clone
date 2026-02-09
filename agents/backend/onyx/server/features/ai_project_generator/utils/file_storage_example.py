"""
File Storage Usage Example
==========================

Example demonstrating the refactored FileStorage class usage.
"""

from file_storage import FileStorage
import tempfile
import os


def example_usage():
    """Demonstrate FileStorage usage."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        storage = FileStorage(tmp_path)
        
        initial_records = [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25},
            {"id": "3", "name": "Charlie", "age": 35}
        ]
        
        storage.write(initial_records)
        print("✅ Records written successfully")
        
        records = storage.read()
        print(f"✅ Read {len(records)} records")
        
        storage.update("2", {"age": 26, "city": "New York"})
        print("✅ Record updated successfully")
        
        updated_records = storage.read()
        updated_bob = next((r for r in updated_records if r.get('id') == '2'), None)
        print(f"✅ Updated record: {updated_bob}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    example_usage()


