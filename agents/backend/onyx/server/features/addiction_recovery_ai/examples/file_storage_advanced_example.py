"""
Advanced Examples - FileStorage Usage Patterns
Demonstrates advanced usage scenarios and best practices
"""

from utils.file_storage import FileStorage
import os
from typing import List, Dict, Any


class AdvancedFileStorage(FileStorage):
    """
    Extended FileStorage with additional utility methods
    """
    
    def bulk_update(self, updates: List[Dict[str, str]]) -> Dict[str, bool]:
        """
        Update multiple records at once
        
        Args:
            updates: List of dicts with 'id' and 'data' keys
            
        Returns:
            Dictionary mapping record IDs to success status
        """
        results = {}
        records = self.read()
        
        for update_item in updates:
            record_id = update_item.get('id')
            update_data = update_item.get('data', {})
            
            if not record_id:
                results[record_id] = False
                continue
            
            found = False
            for i, record in enumerate(records):
                if isinstance(record, dict) and record.get('id') == record_id:
                    records[i].update(update_data)
                    found = True
                    results[record_id] = True
                    break
            
            if not found:
                results[record_id] = False
        
        if any(results.values()):
            self.write(records)
        
        return results
    
    def find_records(self, **criteria) -> List[Dict[str, Any]]:
        """
        Find records matching criteria
        
        Args:
            **criteria: Key-value pairs to match
            
        Returns:
            List of matching records
        """
        records = self.read()
        matches = []
        
        for record in records:
            if not isinstance(record, dict):
                continue
            
            match = all(
                record.get(key) == value
                for key, value in criteria.items()
            )
            
            if match:
                matches.append(record)
        
        return matches
    
    def count_records(self) -> int:
        """Get total number of records"""
        return len(self.read())
    
    def clear_all(self) -> None:
        """Clear all records from storage"""
        self.write([])
    
    def backup(self, backup_path: str) -> bool:
        """
        Create a backup of current data
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if backup successful
        """
        try:
            records = self.read()
            backup_storage = FileStorage(backup_path)
            backup_storage.write(records)
            return True
        except Exception:
            return False
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore data from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful
        """
        try:
            backup_storage = FileStorage(backup_path)
            records = backup_storage.read()
            self.write(records)
            return True
        except Exception:
            return False


def example_basic_operations():
    """Basic CRUD operations"""
    print("=" * 60)
    print("Example 1: Basic CRUD Operations")
    print("=" * 60)
    
    storage = FileStorage("data/basic_example.json")
    
    # Create
    storage.write([
        {"id": "1", "name": "Alice", "role": "admin"},
        {"id": "2", "name": "Bob", "role": "user"},
        {"id": "3", "name": "Charlie", "role": "user"}
    ])
    print("✓ Created records")
    
    # Read
    records = storage.read()
    print(f"✓ Read {len(records)} records")
    
    # Update
    storage.update("1", {"role": "super_admin", "active": True})
    print("✓ Updated record")
    
    # Delete
    storage.delete("3")
    print("✓ Deleted record")
    
    # Get
    record = storage.get("2")
    print(f"✓ Retrieved record: {record}")
    
    # Cleanup
    if os.path.exists("data/basic_example.json"):
        os.remove("data/basic_example.json")


def example_error_handling():
    """Error handling patterns"""
    print("\n" + "=" * 60)
    print("Example 2: Error Handling")
    print("=" * 60)
    
    storage = FileStorage("data/error_example.json")
    
    # Invalid data type
    try:
        storage.write("not a list")
    except TypeError as e:
        print(f"✓ Caught TypeError: {e}")
    
    # Invalid record ID
    try:
        storage.update(123, {"name": "Test"})
    except TypeError as e:
        print(f"✓ Caught TypeError: {e}")
    
    # Empty updates
    storage.write([{"id": "1", "name": "Test"}])
    try:
        storage.update("1", {})
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    # Cleanup
    if os.path.exists("data/error_example.json"):
        os.remove("data/error_example.json")


def example_advanced_operations():
    """Advanced operations with extended class"""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Operations")
    print("=" * 60)
    
    storage = AdvancedFileStorage("data/advanced_example.json")
    
    # Initial data
    storage.write([
        {"id": "1", "name": "Alice", "age": 30, "city": "NYC"},
        {"id": "2", "name": "Bob", "age": 25, "city": "LA"},
        {"id": "3", "name": "Charlie", "age": 35, "city": "NYC"}
    ])
    
    # Bulk update
    bulk_updates = [
        {"id": "1", "data": {"age": 31, "status": "active"}},
        {"id": "2", "data": {"status": "inactive"}}
    ]
    results = storage.bulk_update(bulk_updates)
    print(f"✓ Bulk update results: {results}")
    
    # Find records
    nyc_residents = storage.find_records(city="NYC")
    print(f"✓ Found {len(nyc_residents)} NYC residents")
    
    # Count records
    count = storage.count_records()
    print(f"✓ Total records: {count}")
    
    # Backup
    if storage.backup("data/backup.json"):
        print("✓ Backup created")
    
    # Restore
    storage.clear_all()
    print(f"✓ Cleared all ({storage.count_records()} records)")
    
    if storage.restore("data/backup.json"):
        print(f"✓ Restored ({storage.count_records()} records)")
    
    # Cleanup
    for file in ["data/advanced_example.json", "data/backup.json"]:
        if os.path.exists(file):
            os.remove(file)


def example_transaction_pattern():
    """Transaction-like pattern for atomic operations"""
    print("\n" + "=" * 60)
    print("Example 4: Transaction Pattern")
    print("=" * 60)
    
    storage = FileStorage("data/transaction_example.json")
    
    def atomic_update(record_id: str, updates: Dict[str, Any]) -> bool:
        """Atomic update with rollback on error"""
        try:
            # Read current state
            records = storage.read()
            
            # Find and update
            found = False
            for i, record in enumerate(records):
                if isinstance(record, dict) and record.get('id') == record_id:
                    # Create backup of original
                    original = record.copy()
                    
                    # Apply update
                    records[i].update(updates)
                    
                    # Validate (example: age must be positive)
                    if 'age' in records[i] and records[i]['age'] < 0:
                        # Rollback
                        records[i] = original
                        print(f"✗ Validation failed, rolled back")
                        return False
                    
                    found = True
                    break
            
            if found:
                # Commit: write to file
                storage.write(records)
                print(f"✓ Transaction committed")
                return True
            else:
                print(f"✗ Record not found")
                return False
                
        except Exception as e:
            print(f"✗ Transaction failed: {e}")
            return False
    
    # Setup
    storage.write([
        {"id": "1", "name": "Alice", "age": 30}
    ])
    
    # Valid update
    atomic_update("1", {"age": 31})
    
    # Invalid update (will rollback)
    atomic_update("1", {"age": -5})
    
    # Verify
    record = storage.get("1")
    print(f"✓ Final age: {record['age']} (should be 31, not -5)")
    
    # Cleanup
    if os.path.exists("data/transaction_example.json"):
        os.remove("data/transaction_example.json")


def example_concurrent_safety():
    """Patterns for handling concurrent access"""
    print("\n" + "=" * 60)
    print("Example 5: Concurrent Access Patterns")
    print("=" * 60)
    
    storage = FileStorage("data/concurrent_example.json")
    
    def safe_increment(record_id: str, field: str) -> bool:
        """
        Safely increment a numeric field
        Handles race conditions by reading-modify-write
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                records = storage.read()
                
                for i, record in enumerate(records):
                    if isinstance(record, dict) and record.get('id') == record_id:
                        current_value = record.get(field, 0)
                        
                        if isinstance(current_value, (int, float)):
                            records[i][field] = current_value + 1
                            storage.write(records)
                            return True
                        else:
                            raise ValueError(f"Field {field} is not numeric")
                
                return False
                
            except (IOError, ValueError) as e:
                if attempt == max_retries - 1:
                    print(f"✗ Failed after {max_retries} attempts: {e}")
                    return False
                # Retry
                continue
        
        return False
    
    # Setup
    storage.write([
        {"id": "1", "name": "Counter", "count": 0}
    ])
    
    # Increment
    for _ in range(5):
        safe_increment("1", "count")
    
    record = storage.get("1")
    print(f"✓ Final count: {record['count']} (should be 5)")
    
    # Cleanup
    if os.path.exists("data/concurrent_example.json"):
        os.remove("data/concurrent_example.json")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Advanced FileStorage Examples")
    print("=" * 60)
    
    example_basic_operations()
    example_error_handling()
    example_advanced_operations()
    example_transaction_pattern()
    example_concurrent_safety()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


