"""
Example usage of the refactored FileStorage class
Demonstrates proper file operations with context managers and error handling
"""

from utils.file_storage import FileStorage
import os


def main():
    """Demonstrate FileStorage usage"""
    
    print("=" * 60)
    print("FileStorage Example - Refactored with Best Practices")
    print("=" * 60)
    
    # Initialize storage
    file_path = "data/example_records.json"
    storage = FileStorage(file_path)
    
    print(f"\n1. Initializing storage at: {file_path}")
    print(f"   Directory created automatically: {os.path.dirname(file_path)}")
    
    # Write initial data
    print("\n2. Writing initial data...")
    initial_data = [
        {"id": "1", "name": "John Doe", "age": 30, "city": "New York"},
        {"id": "2", "name": "Jane Smith", "age": 25, "city": "Los Angeles"},
        {"id": "3", "name": "Bob Johnson", "age": 35, "city": "Chicago"}
    ]
    
    try:
        storage.write(initial_data)
        print("   ✓ Data written successfully")
    except Exception as e:
        print(f"   ✗ Error writing data: {e}")
        return
    
    # Read data
    print("\n3. Reading data...")
    try:
        records = storage.read()
        print(f"   ✓ Read {len(records)} records")
        for record in records:
            print(f"      - {record['name']} (ID: {record['id']})")
    except Exception as e:
        print(f"   ✗ Error reading data: {e}")
        return
    
    # Update a record
    print("\n4. Updating record ID '1'...")
    try:
        success = storage.update("1", {"age": 31, "status": "active"})
        if success:
            print("   ✓ Record updated successfully")
            updated_record = storage.get("1")
            print(f"      Updated record: {updated_record}")
        else:
            print("   ✗ Record not found")
    except Exception as e:
        print(f"   ✗ Error updating record: {e}")
    
    # Add a new record
    print("\n5. Adding a new record...")
    try:
        storage.add({"id": "4", "name": "Alice Brown", "age": 28, "city": "Miami"})
        print("   ✓ Record added successfully")
    except Exception as e:
        print(f"   ✗ Error adding record: {e}")
    
    # Get a specific record
    print("\n6. Getting record ID '2'...")
    try:
        record = storage.get("2")
        if record:
            print(f"   ✓ Found record: {record}")
        else:
            print("   ✗ Record not found")
    except Exception as e:
        print(f"   ✗ Error getting record: {e}")
    
    # Delete a record
    print("\n7. Deleting record ID '3'...")
    try:
        success = storage.delete("3")
        if success:
            print("   ✓ Record deleted successfully")
        else:
            print("   ✗ Record not found")
    except Exception as e:
        print(f"   ✗ Error deleting record: {e}")
    
    # Read final state
    print("\n8. Final state of records...")
    try:
        final_records = storage.read()
        print(f"   Total records: {len(final_records)}")
        for record in final_records:
            print(f"      - {record['name']} (ID: {record['id']}, Age: {record.get('age', 'N/A')})")
    except Exception as e:
        print(f"   ✗ Error reading final state: {e}")
    
    # Demonstrate error handling
    print("\n9. Demonstrating error handling...")
    
    print("   a) Invalid data type:")
    try:
        storage.write("not a list")
    except TypeError as e:
        print(f"      ✓ Caught TypeError: {e}")
    
    print("   b) Invalid record ID type:")
    try:
        storage.update(123, {"age": 30})
    except TypeError as e:
        print(f"      ✓ Caught TypeError: {e}")
    
    print("   c) Empty record ID:")
    try:
        storage.update("", {"age": 30})
    except ValueError as e:
        print(f"      ✓ Caught ValueError: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"\nCleaned up: {file_path}")


if __name__ == "__main__":
    main()


