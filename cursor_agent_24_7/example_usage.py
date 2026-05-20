from record_storage import RecordStorage


def main():
    storage = RecordStorage("example_data.json")
    
    print("=== Example: Writing Records ===")
    records = [
        {"id": "1", "name": "Alice", "age": 30, "city": "New York"},
        {"id": "2", "name": "Bob", "age": 25, "city": "Los Angeles"},
        {"id": "3", "name": "Charlie", "age": 35, "city": "Chicago"}
    ]
    
    storage.write(records)
    print(f"✓ Written {len(records)} records")
    
    print("\n=== Example: Reading Records ===")
    all_records = storage.read()
    print(f"✓ Read {len(all_records)} records")
    for record in all_records:
        print(f"  - {record['name']} (ID: {record['id']})")
    
    print("\n=== Example: Updating a Record ===")
    result = storage.update("1", {"age": 31, "status": "active"})
    if result:
        print("✓ Record updated successfully")
        updated = storage.read()
        alice = next(r for r in updated if r["id"] == "1")
        print(f"  Updated record: {alice}")
    else:
        print("✗ Record not found")
    
    print("\n=== Example: Error Handling ===")
    try:
        storage.write("invalid input")
    except TypeError as e:
        print(f"✓ Caught TypeError: {e}")
    
    try:
        storage.update("", {"key": "value"})
    except ValueError as e:
        print(f"✓ Caught ValueError: {e}")
    
    print("\n=== Final State ===")
    final_records = storage.read()
    for record in final_records:
        print(f"  {record}")


if __name__ == "__main__":
    main()


