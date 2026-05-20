from record_storage import RecordStorage
import sys


def print_menu():
    print("\n" + "="*50)
    print("RecordStorage Interactive Demo")
    print("="*50)
    print("1. Add a new record")
    print("2. View all records")
    print("3. Update a record")
    print("4. Search records")
    print("5. Delete a record")
    print("6. Clear all records")
    print("7. Show statistics")
    print("8. Exit")
    print("="*50)


def add_record(storage):
    print("\n--- Add New Record ---")
    try:
        record_id = input("Enter record ID: ").strip()
        if not record_id:
            print("❌ ID cannot be empty")
            return
        
        name = input("Enter name: ").strip()
        age = input("Enter age (optional): ").strip()
        
        record = {"id": record_id, "name": name}
        if age:
            try:
                record["age"] = int(age)
            except ValueError:
                print("⚠️  Age must be a number, skipping...")
        
        records = storage.read()
        if any(r.get("id") == record_id for r in records):
            print(f"❌ Record with ID '{record_id}' already exists")
            return
        
        records.append(record)
        storage.write(records)
        print(f"✅ Record '{record_id}' added successfully")
    except Exception as e:
        print(f"❌ Error: {e}")


def view_records(storage):
    print("\n--- All Records ---")
    records = storage.read()
    if not records:
        print("No records found")
        return
    
    for i, record in enumerate(records, 1):
        print(f"\n{i}. Record ID: {record.get('id', 'N/A')}")
        for key, value in record.items():
            if key != 'id':
                print(f"   {key}: {value}")


def update_record(storage):
    print("\n--- Update Record ---")
    record_id = input("Enter record ID to update: ").strip()
    if not record_id:
        print("❌ ID cannot be empty")
        return
    
    records = storage.read()
    record = next((r for r in records if r.get("id") == record_id), None)
    
    if not record:
        print(f"❌ Record with ID '{record_id}' not found")
        return
    
    print(f"\nCurrent record: {record}")
    print("\nEnter new values (press Enter to skip):")
    
    updates = {}
    for key in record.keys():
        if key == 'id':
            continue
        new_value = input(f"  {key} (current: {record.get(key, 'N/A')}): ").strip()
        if new_value:
            if key == 'age':
                try:
                    updates[key] = int(new_value)
                except ValueError:
                    print(f"⚠️  Invalid age, skipping...")
            else:
                updates[key] = new_value
    
    if not updates:
        print("No updates provided")
        return
    
    if storage.update(record_id, updates):
        print(f"✅ Record '{record_id}' updated successfully")
    else:
        print(f"❌ Failed to update record '{record_id}'")


def search_records(storage):
    print("\n--- Search Records ---")
    search_term = input("Enter search term: ").strip().lower()
    if not search_term:
        print("❌ Search term cannot be empty")
        return
    
    records = storage.read()
    matches = []
    
    for record in records:
        for key, value in record.items():
            if search_term in str(value).lower():
                matches.append(record)
                break
    
    if not matches:
        print(f"No records found matching '{search_term}'")
        return
    
    print(f"\nFound {len(matches)} matching record(s):")
    for i, record in enumerate(matches, 1):
        print(f"\n{i}. {record}")


def delete_record(storage):
    print("\n--- Delete Record ---")
    record_id = input("Enter record ID to delete: ").strip()
    if not record_id:
        print("❌ ID cannot be empty")
        return
    
    records = storage.read()
    original_count = len(records)
    records = [r for r in records if r.get("id") != record_id]
    
    if len(records) == original_count:
        print(f"❌ Record with ID '{record_id}' not found")
        return
    
    storage.write(records)
    print(f"✅ Record '{record_id}' deleted successfully")


def clear_all(storage):
    print("\n--- Clear All Records ---")
    confirm = input("Are you sure? This will delete ALL records (yes/no): ").strip().lower()
    if confirm == 'yes':
        storage.write([])
        print("✅ All records cleared")
    else:
        print("❌ Operation cancelled")


def show_statistics(storage):
    print("\n--- Statistics ---")
    records = storage.read()
    print(f"Total records: {len(records)}")
    
    if records:
        all_keys = set()
        for record in records:
            all_keys.update(record.keys())
        
        print(f"\nFields used: {', '.join(sorted(all_keys))}")
        
        if 'age' in all_keys:
            ages = [r.get('age') for r in records if isinstance(r.get('age'), (int, float))]
            if ages:
                print(f"\nAge statistics:")
                print(f"  Min: {min(ages)}")
                print(f"  Max: {max(ages)}")
                print(f"  Average: {sum(ages) / len(ages):.2f}")


def main():
    print("Initializing RecordStorage...")
    storage = RecordStorage("demo_data.json")
    print("✅ Storage initialized")
    
    while True:
        print_menu()
        choice = input("\nSelect an option (1-8): ").strip()
        
        try:
            if choice == '1':
                add_record(storage)
            elif choice == '2':
                view_records(storage)
            elif choice == '3':
                update_record(storage)
            elif choice == '4':
                search_records(storage)
            elif choice == '5':
                delete_record(storage)
            elif choice == '6':
                clear_all(storage)
            elif choice == '7':
                show_statistics(storage)
            elif choice == '8':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-8.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


