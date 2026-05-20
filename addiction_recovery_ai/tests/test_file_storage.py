"""
Tests for FileStorage class
Demonstrates proper usage and error handling
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from utils.file_storage import FileStorage


class TestFileStorage:
    """Test suite for FileStorage class"""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def storage(self, temp_file):
        """Create a FileStorage instance"""
        return FileStorage(temp_file)
    
    def test_write_and_read(self, storage):
        """Test writing and reading data"""
        data = [
            {"id": "1", "name": "John", "age": 30},
            {"id": "2", "name": "Jane", "age": 25}
        ]
        
        storage.write(data)
        result = storage.read()
        
        assert result == data
        assert len(result) == 2
    
    def test_write_empty_list(self, storage):
        """Test writing empty list"""
        storage.write([])
        result = storage.read()
        assert result == []
    
    def test_read_nonexistent_file(self, temp_file):
        """Test reading from non-existent file"""
        os.unlink(temp_file)
        storage = FileStorage(temp_file)
        result = storage.read()
        assert result == []
    
    def test_update_existing_record(self, storage):
        """Test updating an existing record"""
        initial_data = [
            {"id": "1", "name": "John", "age": 30},
            {"id": "2", "name": "Jane", "age": 25}
        ]
        storage.write(initial_data)
        
        success = storage.update("1", {"age": 31, "city": "NYC"})
        assert success is True
        
        updated = storage.read()
        assert updated[0]["age"] == 31
        assert updated[0]["city"] == "NYC"
        assert updated[0]["name"] == "John"
    
    def test_update_nonexistent_record(self, storage):
        """Test updating a non-existent record"""
        initial_data = [{"id": "1", "name": "John"}]
        storage.write(initial_data)
        
        success = storage.update("999", {"age": 30})
        assert success is False
        
        result = storage.read()
        assert result == initial_data
    
    def test_add_record(self, storage):
        """Test adding a new record"""
        initial_data = [{"id": "1", "name": "John"}]
        storage.write(initial_data)
        
        storage.add({"id": "2", "name": "Jane"})
        
        result = storage.read()
        assert len(result) == 2
        assert result[1]["name"] == "Jane"
    
    def test_delete_record(self, storage):
        """Test deleting a record"""
        initial_data = [
            {"id": "1", "name": "John"},
            {"id": "2", "name": "Jane"},
            {"id": "3", "name": "Bob"}
        ]
        storage.write(initial_data)
        
        success = storage.delete("2")
        assert success is True
        
        result = storage.read()
        assert len(result) == 2
        assert all(r["id"] != "2" for r in result)
    
    def test_delete_nonexistent_record(self, storage):
        """Test deleting a non-existent record"""
        initial_data = [{"id": "1", "name": "John"}]
        storage.write(initial_data)
        
        success = storage.delete("999")
        assert success is False
        
        result = storage.read()
        assert len(result) == 1
    
    def test_get_record(self, storage):
        """Test getting a specific record"""
        initial_data = [
            {"id": "1", "name": "John", "age": 30},
            {"id": "2", "name": "Jane", "age": 25}
        ]
        storage.write(initial_data)
        
        record = storage.get("1")
        assert record is not None
        assert record["name"] == "John"
        assert record["age"] == 30
    
    def test_get_nonexistent_record(self, storage):
        """Test getting a non-existent record"""
        initial_data = [{"id": "1", "name": "John"}]
        storage.write(initial_data)
        
        record = storage.get("999")
        assert record is None
    
    def test_write_invalid_data_type(self, storage):
        """Test writing invalid data type"""
        with pytest.raises(TypeError, match="data must be a list"):
            storage.write("not a list")
    
    def test_write_invalid_list_items(self, storage):
        """Test writing list with non-dict items"""
        with pytest.raises(ValueError, match="All items in data must be dictionaries"):
            storage.write([1, 2, 3])
    
    def test_update_invalid_record_id_type(self, storage):
        """Test update with invalid record_id type"""
        with pytest.raises(TypeError, match="record_id must be a string"):
            storage.update(123, {"age": 30})
    
    def test_update_empty_record_id(self, storage):
        """Test update with empty record_id"""
        with pytest.raises(ValueError, match="record_id cannot be empty"):
            storage.update("", {"age": 30})
    
    def test_update_invalid_updates_type(self, storage):
        """Test update with invalid updates type"""
        with pytest.raises(TypeError, match="updates must be a dictionary"):
            storage.update("1", "not a dict")
    
    def test_update_empty_updates(self, storage):
        """Test update with empty updates"""
        storage.write([{"id": "1", "name": "John"}])
        with pytest.raises(ValueError, match="updates cannot be empty"):
            storage.update("1", {})
    
    def test_add_invalid_record_type(self, storage):
        """Test adding invalid record type"""
        with pytest.raises(TypeError, match="record must be a dictionary"):
            storage.add("not a dict")
    
    def test_add_empty_record(self, storage):
        """Test adding empty record"""
        with pytest.raises(ValueError, match="record cannot be empty"):
            storage.add({})
    
    def test_delete_invalid_record_id_type(self, storage):
        """Test delete with invalid record_id type"""
        with pytest.raises(TypeError, match="record_id must be a string"):
            storage.delete(123)
    
    def test_get_invalid_record_id_type(self, storage):
        """Test get with invalid record_id type"""
        with pytest.raises(TypeError, match="record_id must be a string"):
            storage.get(123)
    
    def test_context_manager_usage(self, temp_file):
        """Test that context managers are used properly"""
        storage = FileStorage(temp_file)
        data = [{"id": "1", "name": "Test"}]
        
        storage.write(data)
        
        with open(temp_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_directory_creation(self):
        """Test that directories are created automatically"""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "subdir", "data.json")
            storage = FileStorage(file_path)
            
            data = [{"id": "1", "name": "Test"}]
            storage.write(data)
            
            assert os.path.exists(file_path)
            result = storage.read()
            assert result == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


