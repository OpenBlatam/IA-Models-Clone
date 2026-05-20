import json
import tempfile
import os
from pathlib import Path
from record_storage import RecordStorage


def test_read_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        records = storage.read()
        assert records == [], "Should return empty list for new file"
        print("✓ test_read_empty_file passed")
    finally:
        os.unlink(temp_path)


def test_write_and_read():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        test_records = [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25}
        ]
        
        result = storage.write(test_records)
        assert result is True, "Write should return True"
        
        records = storage.read()
        assert len(records) == 2, "Should read 2 records"
        assert records[0]["name"] == "Alice", "First record should be Alice"
        print("✓ test_write_and_read passed")
    finally:
        os.unlink(temp_path)


def test_update_record():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        test_records = [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25}
        ]
        
        storage.write(test_records)
        
        result = storage.update("1", {"age": 31, "city": "New York"})
        assert result is True, "Update should return True"
        
        records = storage.read()
        updated_record = next(r for r in records if r["id"] == "1")
        assert updated_record["age"] == 31, "Age should be updated"
        assert updated_record["name"] == "Alice", "Name should be preserved"
        assert updated_record["city"] == "New York", "City should be added"
        assert updated_record["id"] == "1", "ID should be preserved"
        print("✓ test_update_record passed")
    finally:
        os.unlink(temp_path)


def test_update_nonexistent_record():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        test_records = [{"id": "1", "name": "Alice"}]
        storage.write(test_records)
        
        result = storage.update("999", {"age": 30})
        assert result is False, "Update should return False for nonexistent record"
        print("✓ test_update_nonexistent_record passed")
    finally:
        os.unlink(temp_path)


def test_error_handling_invalid_input():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        
        try:
            storage.write("not a list")
            assert False, "Should raise TypeError"
        except TypeError:
            print("✓ test_error_handling_invalid_input (TypeError) passed")
        
        try:
            storage.write([{"id": "1"}, "not a dict"])
            assert False, "Should raise ValueError"
        except ValueError:
            print("✓ test_error_handling_invalid_input (ValueError) passed")
        
        try:
            storage.update("", {"key": "value"})
            assert False, "Should raise ValueError for empty record_id"
        except ValueError:
            print("✓ test_error_handling_invalid_input (empty record_id) passed")
        
        try:
            storage.update("1", "not a dict")
            assert False, "Should raise TypeError"
        except TypeError:
            print("✓ test_error_handling_invalid_input (updates not dict) passed")
    finally:
        os.unlink(temp_path)


def test_context_manager_file_closure():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        test_records = [{"id": "1", "name": "Test"}]
        storage.write(test_records)
        
        with open(temp_path, 'r') as f:
            data = json.load(f)
            assert "records" in data, "File should be properly written"
        
        print("✓ test_context_manager_file_closure passed")
    finally:
        os.unlink(temp_path)


def test_preserve_id_on_update():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        storage = RecordStorage(temp_path)
        test_records = [{"id": "1", "name": "Alice"}]
        storage.write(test_records)
        
        storage.update("1", {"id": "999", "age": 30})
        
        records = storage.read()
        updated_record = records[0]
        assert updated_record["id"] == "1", "ID should be preserved even if update tries to change it"
        assert updated_record["age"] == 30, "Other fields should be updated"
        print("✓ test_preserve_id_on_update passed")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("Running RecordStorage tests...\n")
    
    test_read_empty_file()
    test_write_and_read()
    test_update_record()
    test_update_nonexistent_record()
    test_error_handling_invalid_input()
    test_context_manager_file_closure()
    test_preserve_id_on_update()
    
    print("\n✅ All tests passed!")


