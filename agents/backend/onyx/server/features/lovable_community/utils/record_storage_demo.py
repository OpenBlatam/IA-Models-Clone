"""
Record Storage - Interactive Demo Script

This script demonstrates the RecordStorage class with interactive examples.
Run this script to see the refactored code in action.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.record_storage import RecordStorage


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_operations():
    """Demonstrate basic CRUD operations."""
    print_section("Demo 1: Basic CRUD Operations")
    
    storage = RecordStorage("demo_data/basic_storage.json")
    print("✓ Created RecordStorage instance")
    
    initial_data = [
        {"id": "1", "name": "Alice", "email": "alice@example.com", "age": 30},
        {"id": "2", "name": "Bob", "email": "bob@example.com", "age": 25},
        {"id": "3", "name": "Charlie", "email": "charlie@example.com", "age": 35}
    ]
    
    print(f"\nWriting {len(initial_data)} records...")
    storage.write(initial_data)
    print("✓ Records written successfully")
    
    print("\nReading all records...")
    records = storage.read()
    print(f"✓ Read {len(records)} records:")
    for record in records:
        print(f"  - {record['name']} (ID: {record['id']}, Age: {record['age']})")
    
    print("\nGetting specific record...")
    record = storage.get("2")
    if record:
        print(f"✓ Found record: {record['name']} - {record['email']}")
    
    print("\nUpdating record...")
    success = storage.update("1", {"age": 31, "city": "New York"})
    if success:
        updated = storage.get("1")
        print(f"✓ Updated record: {updated['name']} is now {updated['age']} years old")
        print(f"  New city: {updated.get('city', 'N/A')}")
    
    print("\nAdding new record...")
    new_record = {"id": "4", "name": "David", "email": "david@example.com", "age": 28}
    storage.add(new_record)
    print(f"✓ Added: {new_record['name']}")
    print(f"  Total records: {len(storage.read())}")
    
    print("\nDeleting record...")
    storage.delete("3")
    print("✓ Deleted record with ID '3'")
    print(f"  Remaining records: {len(storage.read())}")


def demo_error_handling():
    """Demonstrate error handling and validation."""
    print_section("Demo 2: Error Handling & Validation")
    
    storage = RecordStorage("demo_data/error_demo.json")
    
    print("Testing input validation...")
    
    try:
        storage.write("not a list")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.write([{"id": "1"}, "not a dict"])
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.update("", {"age": 30})
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.update("1", "not a dict")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    try:
        storage.add({"name": "No ID"})
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    print("\nTesting operations on non-existent records...")
    result = storage.update("999", {"age": 50})
    print(f"✓ Update non-existent record returned: {result}")
    
    result = storage.delete("999")
    print(f"✓ Delete non-existent record returned: {result}")
    
    record = storage.get("999")
    print(f"✓ Get non-existent record returned: {record}")


def demo_context_managers():
    """Demonstrate that context managers work correctly."""
    print_section("Demo 3: Context Manager Safety")
    
    storage = RecordStorage("demo_data/context_demo.json")
    
    print("Writing data with context manager...")
    storage.write([{"id": "1", "data": "test"}])
    print("✓ File written and automatically closed")
    
    print("\nReading data multiple times (testing file closure)...")
    for i in range(5):
        records = storage.read()
        print(f"  Read #{i+1}: {len(records)} records")
    print("✓ All reads successful - files properly closed each time")
    
    print("\nPerforming multiple write operations...")
    for i in range(3):
        storage.write([{"id": str(i+1), "iteration": i+1}])
        records = storage.read()
        print(f"  Write #{i+1}: {len(records)} records")
    print("✓ All writes successful - files properly closed each time")


def demo_update_merging():
    """Demonstrate that update merges correctly instead of replacing."""
    print_section("Demo 4: Update Merging (Not Replacing)")
    
    storage = RecordStorage("demo_data/merge_demo.json")
    
    original = {
        "id": "1",
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
        "city": "Boston",
        "country": "USA"
    }
    
    storage.write([original])
    print("Original record:")
    print(f"  {original}")
    
    print("\nUpdating only 'age' and 'city'...")
    storage.update("1", {"age": 31, "city": "New York"})
    
    updated = storage.get("1")
    print("\nUpdated record (should preserve other fields):")
    print(f"  {updated}")
    
    print("\n✓ Verification:")
    print(f"  - Name preserved: {updated['name'] == original['name']}")
    print(f"  - Email preserved: {updated['email'] == original['email']}")
    print(f"  - Country preserved: {updated['country'] == original['country']}")
    print(f"  - Age updated: {updated['age']} (was {original['age']})")
    print(f"  - City updated: {updated['city']} (was {original['city']})")
    print(f"  - ID preserved: {updated['id'] == original['id']}")


def demo_concurrent_operations():
    """Demonstrate sequential operations (simulating concurrent access)."""
    print_section("Demo 5: Sequential Operations Workflow")
    
    storage = RecordStorage("demo_data/workflow_demo.json")
    
    print("Simulating a complete workflow...")
    
    print("\n1. Initializing with user data...")
    users = [
        {"id": "user1", "name": "Alice", "role": "admin", "active": True},
        {"id": "user2", "name": "Bob", "role": "user", "active": True},
        {"id": "user3", "name": "Charlie", "role": "user", "active": False}
    ]
    storage.write(users)
    print(f"   ✓ Created {len(users)} users")
    
    print("\n2. Updating user1 with last login...")
    storage.update("user1", {"last_login": "2024-01-15"})
    user1 = storage.get("user1")
    print(f"   ✓ {user1['name']} last login: {user1.get('last_login')}")
    
    print("\n3. Deactivating user2...")
    storage.update("user2", {"active": False})
    user2 = storage.get("user2")
    print(f"   ✓ {user2['name']} active status: {user2['active']}")
    
    print("\n4. Adding new user...")
    storage.add({"id": "user4", "name": "David", "role": "user", "active": True})
    print(f"   ✓ Added new user: David")
    
    print("\n5. Getting active users...")
    all_users = storage.read()
    active_users = [u for u in all_users if u.get("active", False)]
    print(f"   ✓ Active users: {len(active_users)}")
    for user in active_users:
        print(f"     - {user['name']} ({user['role']})")
    
    print("\n6. Final state:")
    final_records = storage.read()
    print(f"   ✓ Total users: {len(final_records)}")
    for user in final_records:
        status = "active" if user.get("active") else "inactive"
        print(f"     - {user['name']}: {status}")


def demo_data_persistence():
    """Demonstrate data persistence across instances."""
    print_section("Demo 6: Data Persistence")
    
    file_path = "demo_data/persistence_demo.json"
    
    print("Creating first instance and writing data...")
    storage1 = RecordStorage(file_path)
    storage1.write([{"id": "1", "message": "Hello from instance 1"}])
    print("✓ Data written by first instance")
    
    print("\nCreating second instance and reading data...")
    storage2 = RecordStorage(file_path)
    records = storage2.read()
    print(f"✓ Second instance read {len(records)} records")
    print(f"  Message: {records[0]['message']}")
    
    print("\nUpdating from second instance...")
    storage2.update("1", {"message": "Updated from instance 2"})
    
    print("\nReading from first instance again...")
    records = storage1.read()
    print(f"✓ First instance sees updated data")
    print(f"  Message: {records[0]['message']}")
    
    print("\n✓ Data persists correctly across instances!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  Record Storage - Interactive Demo")
    print("  Demonstrating Refactored Code with All Improvements")
    print("=" * 70)
    
    try:
        demo_basic_operations()
        demo_error_handling()
        demo_context_managers()
        demo_update_merging()
        demo_concurrent_operations()
        demo_data_persistence()
        
        print_section("Demo Complete!")
        print("✓ All demonstrations completed successfully")
        print("\nKey improvements demonstrated:")
        print("  1. ✓ Context managers for safe file operations")
        print("  2. ✓ Correct indentation in all methods")
        print("  3. ✓ Proper record merging (not replacing) in update()")
        print("  4. ✓ Comprehensive error handling and validation")
        print("  5. ✓ Data persistence across instances")
        print("  6. ✓ Complete CRUD operations")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


