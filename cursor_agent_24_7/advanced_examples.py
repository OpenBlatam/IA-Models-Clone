from record_storage import RecordStorage
from typing import List, Dict, Any, Optional
import json


class AdvancedRecordStorage(RecordStorage):
    def find_by_field(self, field: str, value: Any) -> List[Dict[str, Any]]:
        records = self.read()
        return [r for r in records if r.get(field) == value]
    
    def find_by_fields(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        records = self.read()
        matches = []
        for record in records:
            if all(record.get(k) == v for k, v in criteria.items()):
                matches.append(record)
        return matches
    
    def bulk_update(self, updates: Dict[str, Dict[str, Any]]) -> int:
        updated_count = 0
        for record_id, update_data in updates.items():
            if self.update(record_id, update_data):
                updated_count += 1
        return updated_count
    
    def get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        records = self.read()
        return next((r for r in records if r.get("id") == record_id), None)
    
    def exists(self, record_id: str) -> bool:
        return self.get_by_id(record_id) is not None
    
    def count(self) -> int:
        return len(self.read())
    
    def export_to_json(self, output_path: str) -> bool:
        try:
            records = self.read()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def import_from_json(self, input_path: str, merge: bool = False) -> int:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_records = data.get("records", [])
            if not imported_records:
                return 0
            
            if merge:
                existing = self.read()
                existing_ids = {r.get("id") for r in existing}
                new_records = [r for r in imported_records if r.get("id") not in existing_ids]
                all_records = existing + new_records
                self.write(all_records)
                return len(new_records)
            else:
                self.write(imported_records)
                return len(imported_records)
        except Exception:
            return 0


def example_bulk_operations():
    print("=== Bulk Operations Example ===")
    storage = AdvancedRecordStorage("bulk_data.json")
    
    initial_records = [
        {"id": "1", "name": "Alice", "department": "Engineering", "status": "active"},
        {"id": "2", "name": "Bob", "department": "Engineering", "status": "active"},
        {"id": "3", "name": "Charlie", "department": "Sales", "status": "active"},
    ]
    storage.write(initial_records)
    print(f"✅ Created {len(initial_records)} initial records")
    
    engineering_team = storage.find_by_field("department", "Engineering")
    print(f"✅ Found {len(engineering_team)} engineers")
    
    updates = {
        "1": {"status": "promoted", "level": "senior"},
        "2": {"status": "promoted", "level": "senior"},
    }
    updated = storage.bulk_update(updates)
    print(f"✅ Bulk updated {updated} records")
    
    active_engineers = storage.find_by_fields({"department": "Engineering", "status": "promoted"})
    print(f"✅ Found {len(active_engineers)} promoted engineers")


def example_import_export():
    print("\n=== Import/Export Example ===")
    storage = AdvancedRecordStorage("import_export_data.json")
    
    source_records = [
        {"id": "1", "name": "Alice", "age": 30},
        {"id": "2", "name": "Bob", "age": 25},
    ]
    storage.write(source_records)
    print(f"✅ Created {len(source_records)} records")
    
    if storage.export_to_json("backup.json"):
        print("✅ Exported to backup.json")
    
    new_storage = AdvancedRecordStorage("imported_data.json")
    imported = new_storage.import_from_json("backup.json", merge=False)
    print(f"✅ Imported {imported} records to new storage")
    
    print(f"✅ New storage has {new_storage.count()} records")


def example_search_operations():
    print("\n=== Search Operations Example ===")
    storage = AdvancedRecordStorage("search_data.json")
    
    records = [
        {"id": "1", "name": "Alice", "city": "New York", "age": 30},
        {"id": "2", "name": "Bob", "city": "Los Angeles", "age": 25},
        {"id": "3", "name": "Charlie", "city": "New York", "age": 35},
    ]
    storage.write(records)
    print(f"✅ Created {len(records)} records")
    
    ny_residents = storage.find_by_field("city", "New York")
    print(f"✅ Found {len(ny_residents)} New York residents")
    
    for record in ny_residents:
        print(f"   - {record['name']} (age {record.get('age', 'N/A')})")


def example_data_validation():
    print("\n=== Data Validation Example ===")
    storage = AdvancedRecordStorage("validation_data.json")
    
    try:
        storage.write("not a list")
    except TypeError as e:
        print(f"✅ Caught TypeError: {e}")
    
    try:
        storage.write([{"id": "1"}, "not a dict"])
    except ValueError as e:
        print(f"✅ Caught ValueError: {e}")
    
    try:
        storage.update("", {"key": "value"})
    except ValueError as e:
        print(f"✅ Caught ValueError for empty ID: {e}")
    
    try:
        storage.update("1", "not a dict")
    except TypeError as e:
        print(f"✅ Caught TypeError: {e}")


def example_record_lifecycle():
    print("\n=== Record Lifecycle Example ===")
    storage = AdvancedRecordStorage("lifecycle_data.json")
    
    record = {"id": "1", "name": "Test User", "status": "new"}
    
    if not storage.exists("1"):
        records = storage.read()
        records.append(record)
        storage.write(records)
        print("✅ Created new record")
    
    found = storage.get_by_id("1")
    print(f"✅ Retrieved record: {found}")
    
    storage.update("1", {"status": "active", "last_login": "2024-01-01"})
    print("✅ Updated record")
    
    updated = storage.get_by_id("1")
    print(f"✅ Updated record: {updated}")
    
    print(f"✅ Total records: {storage.count()}")


if __name__ == "__main__":
    example_bulk_operations()
    example_import_export()
    example_search_operations()
    example_data_validation()
    example_record_lifecycle()
    
    print("\n✅ All advanced examples completed!")


