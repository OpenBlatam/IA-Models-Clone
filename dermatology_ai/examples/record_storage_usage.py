"""
Example usage of the refactored RecordStorage class
Demonstrates proper usage patterns and error handling
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.record_storage import RecordStorage


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    storage = RecordStorage("example_data.json")
    
    initial_records = [
        {"id": "1", "name": "Alice", "age": 30, "city": "New York"},
        {"id": "2", "name": "Bob", "age": 25, "city": "London"},
        {"id": "3", "name": "Charlie", "age": 35, "city": "Tokyo"}
    ]
    
    storage.write(initial_records)
    print(f"✓ Written {len(initial_records)} records")
    
    all_records = storage.read()
    print(f"✓ Read {len(all_records)} records")
    for record in all_records:
        print(f"  - {record['name']} (ID: {record['id']})")
    
    Path("example_data.json").unlink(missing_ok=True)
    print()


def example_update_merging():
    """Demonstrates that update merges instead of replacing"""
    print("=== Update Merging Example ===")
    
    storage = RecordStorage("example_data.json")
    
    initial_record = {
        "id": "1",
        "name": "Alice",
        "age": 30,
        "city": "New York",
        "email": "alice@example.com",
        "phone": "123-456-7890"
    }
    
    storage.write([initial_record])
    print("Initial record:")
    print(f"  {storage.read()[0]}")
    
    storage.update("1", {"age": 31, "city": "Boston"})
    updated = storage.read()[0]
    print("\nAfter updating age and city:")
    print(f"  {updated}")
    
    print("\n✓ All original fields preserved:")
    print(f"  - name: {updated.get('name')} (preserved)")
    print(f"  - email: {updated.get('email')} (preserved)")
    print(f"  - phone: {updated.get('phone')} (preserved)")
    print(f"  - age: {updated.get('age')} (updated)")
    print(f"  - city: {updated.get('city')} (updated)")
    
    Path("example_data.json").unlink(missing_ok=True)
    print()


def example_error_handling():
    """Demonstrates error handling"""
    print("=== Error Handling Example ===")
    
    storage = RecordStorage("example_data.json")
    
    print("Testing input validation:")
    
    try:
        storage.write("not a list")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.write([{"id": "1"}, "not a dict"])
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.update("", {"name": "Test"})
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.update("1", "not a dict")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    result = storage.update("999", {"name": "Test"})
    print(f"✓ Update non-existent record returns: {result}")
    
    Path("example_data.json").unlink(missing_ok=True)
    print()


def example_context_manager_safety():
    """Demonstrates context manager safety"""
    print("=== Context Manager Safety Example ===")
    
    storage = RecordStorage("example_data.json")
    
    print("Writing records with context manager...")
    storage.write([{"id": "1", "name": "Test"}])
    
    print("Reading records with context manager...")
    records = storage.read()
    print(f"✓ Successfully read {len(records)} records")
    
    print("✓ Files are automatically closed even if exceptions occur")
    print("  (Context managers ensure proper cleanup)")
    
    Path("example_data.json").unlink(missing_ok=True)
    print()


def example_unicode_support():
    """Demonstrates Unicode support"""
    print("=== Unicode Support Example ===")
    
    storage = RecordStorage("example_data.json")
    
    unicode_records = [
        {"id": "1", "name": "José", "city": "São Paulo", "note": "Café español"},
        {"id": "2", "name": "François", "city": "München", "note": "Über alles"},
        {"id": "3", "name": "李", "city": "北京", "note": "中文测试"}
    ]
    
    storage.write(unicode_records)
    print("✓ Written records with Unicode characters")
    
    read_records = storage.read()
    for record in read_records:
        print(f"  - {record['name']} from {record['city']}")
    
    Path("example_data.json").unlink(missing_ok=True)
    print()


def example_complete_workflow():
    """Complete workflow example"""
    print("=== Complete Workflow Example ===")
    
    storage = RecordStorage("workflow_data.json")
    
    print("1. Creating initial records...")
    records = [
        {"id": "user_1", "name": "John", "role": "admin", "active": True},
        {"id": "user_2", "name": "Jane", "role": "user", "active": True},
        {"id": "user_3", "name": "Bob", "role": "user", "active": False}
    ]
    storage.write(records)
    print(f"   ✓ Created {len(records)} users")
    
    print("\n2. Reading all records...")
    all_users = storage.read()
    print(f"   ✓ Found {len(all_users)} users")
    
    print("\n3. Updating user status...")
    storage.update("user_3", {"active": True})
    updated_user = next(u for u in storage.read() if u["id"] == "user_3")
    print(f"   ✓ User {updated_user['name']} is now active: {updated_user['active']}")
    
    print("\n4. Updating user role...")
    storage.update("user_2", {"role": "moderator"})
    updated_user = next(u for u in storage.read() if u["id"] == "user_2")
    print(f"   ✓ User {updated_user['name']} role updated to: {updated_user['role']}")
    print(f"   ✓ Original fields preserved: name={updated_user['name']}, active={updated_user['active']}")
    
    print("\n5. Final state:")
    final_records = storage.read()
    for user in final_records:
        print(f"   - {user['name']}: {user['role']} (active: {user['active']})")
    
    Path("workflow_data.json").unlink(missing_ok=True)
    print()


if __name__ == "__main__":
    print("RecordStorage Usage Examples\n")
    print("=" * 50 + "\n")
    
    example_basic_usage()
    example_update_merging()
    example_error_handling()
    example_context_manager_safety()
    example_unicode_support()
    example_complete_workflow()
    
    print("=" * 50)
    print("All examples completed successfully!")
    print("\nKey takeaways:")
    print("  ✓ Context managers ensure files are always closed")
    print("  ✓ Updates merge fields instead of replacing entire records")
    print("  ✓ Input validation prevents invalid data")
    print("  ✓ Error handling provides clear error messages")
    print("  ✓ Unicode characters are properly supported")


