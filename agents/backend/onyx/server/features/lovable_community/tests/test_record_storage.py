"""
Unit tests for RecordStorage class

Tests all functionality including error handling, validation, and file operations.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from ..utils.record_storage import RecordStorage


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def storage(temp_dir):
    """Create a RecordStorage instance with a temporary file."""
    file_path = temp_dir / "test_storage.json"
    return RecordStorage(str(file_path))


@pytest.fixture
def sample_records():
    """Sample records for testing."""
    return [
        {"id": "1", "name": "Alice", "age": 30},
        {"id": "2", "name": "Bob", "age": 25},
        {"id": "3", "name": "Charlie", "age": 35}
    ]


class TestRecordStorageInitialization:
    """Test initialization and file creation."""
    
    def test_init_creates_file_if_not_exists(self, temp_dir):
        """Test that file is created if it doesn't exist."""
        file_path = temp_dir / "new_storage.json"
        storage = RecordStorage(str(file_path))
        
        assert file_path.exists()
        assert storage.read() == []
    
    def test_init_with_invalid_path_raises_error(self):
        """Test that invalid file path raises ValueError."""
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            RecordStorage("")
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            RecordStorage(None)
    
    def test_init_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created if they don't exist."""
        file_path = temp_dir / "nested" / "deep" / "storage.json"
        storage = RecordStorage(str(file_path))
        
        assert file_path.exists()
        assert file_path.parent.exists()


class TestRecordStorageRead:
    """Test read operations."""
    
    def test_read_empty_file(self, storage):
        """Test reading from an empty file."""
        records = storage.read()
        assert records == []
    
    def test_read_with_records(self, storage, sample_records):
        """Test reading records from file."""
        storage.write(sample_records)
        records = storage.read()
        
        assert len(records) == 3
        assert records == sample_records
    
    def test_read_invalid_json_raises_error(self, storage):
        """Test that invalid JSON raises RuntimeError."""
        with open(storage.file_path, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(RuntimeError, match="Cannot parse storage file"):
            storage.read()
    
    def test_read_missing_records_key(self, storage):
        """Test reading file without 'records' key."""
        with open(storage.file_path, 'w') as f:
            json.dump({"data": []}, f)
        
        records = storage.read()
        assert records == []
    
    def test_read_records_not_list(self, storage):
        """Test reading file where 'records' is not a list."""
        with open(storage.file_path, 'w') as f:
            json.dump({"records": "not a list"}, f)
        
        records = storage.read()
        assert records == []


class TestRecordStorageWrite:
    """Test write operations."""
    
    def test_write_records(self, storage, sample_records):
        """Test writing records to file."""
        result = storage.write(sample_records)
        
        assert result is True
        assert storage.read() == sample_records
    
    def test_write_invalid_type_raises_error(self, storage):
        """Test that writing non-list raises ValueError."""
        with pytest.raises(ValueError, match="records must be a list"):
            storage.write("not a list")
        
        with pytest.raises(ValueError, match="records must be a list"):
            storage.write({"not": "a list"})
    
    def test_write_invalid_record_type_raises_error(self, storage):
        """Test that writing list with non-dict items raises ValueError."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            storage.write(["not", "a", "dict"])
    
    def test_write_overwrites_existing(self, storage, sample_records):
        """Test that write overwrites existing records."""
        storage.write(sample_records)
        new_records = [{"id": "4", "name": "David", "age": 40}]
        storage.write(new_records)
        
        assert storage.read() == new_records
        assert len(storage.read()) == 1


class TestRecordStorageUpdate:
    """Test update operations."""
    
    def test_update_existing_record(self, storage, sample_records):
        """Test updating an existing record."""
        storage.write(sample_records)
        
        updates = {"age": 31, "city": "New York"}
        result = storage.update("1", updates)
        
        assert result is True
        records = storage.read()
        updated_record = next(r for r in records if r["id"] == "1")
        
        assert updated_record["name"] == "Alice"
        assert updated_record["age"] == 31
        assert updated_record["city"] == "New York"
        assert updated_record["id"] == "1"
    
    def test_update_preserves_id(self, storage, sample_records):
        """Test that update preserves the original ID."""
        storage.write(sample_records)
        
        updates = {"id": "999", "age": 31}
        storage.update("1", updates)
        
        records = storage.read()
        updated_record = next(r for r in records if r["id"] == "1")
        assert updated_record["id"] == "1"
        assert updated_record["age"] == 31
    
    def test_update_nonexistent_record(self, storage, sample_records):
        """Test updating a non-existent record."""
        storage.write(sample_records)
        
        result = storage.update("999", {"age": 50})
        assert result is False
        assert storage.read() == sample_records
    
    def test_update_invalid_record_id_raises_error(self, storage):
        """Test that invalid record_id raises ValueError."""
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.update("", {"age": 30})
        
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.update(None, {"age": 30})
    
    def test_update_invalid_updates_raises_error(self, storage):
        """Test that invalid updates raises ValueError."""
        with pytest.raises(ValueError, match="updates must be a dictionary"):
            storage.update("1", "not a dict")
        
        with pytest.raises(ValueError, match="updates must be a dictionary"):
            storage.update("1", None)
    
    def test_update_empty_updates_returns_false(self, storage, sample_records):
        """Test that empty updates returns False."""
        storage.write(sample_records)
        result = storage.update("1", {})
        assert result is False


class TestRecordStorageAdd:
    """Test add operations."""
    
    def test_add_new_record(self, storage, sample_records):
        """Test adding a new record."""
        storage.write(sample_records)
        
        new_record = {"id": "4", "name": "David", "age": 40}
        result = storage.add(new_record)
        
        assert result is True
        records = storage.read()
        assert len(records) == 4
        assert new_record in records
    
    def test_add_duplicate_id_returns_false(self, storage, sample_records):
        """Test that adding record with existing ID returns False."""
        storage.write(sample_records)
        
        duplicate = {"id": "1", "name": "Duplicate", "age": 99}
        result = storage.add(duplicate)
        
        assert result is False
        assert len(storage.read()) == 3
    
    def test_add_invalid_record_raises_error(self, storage):
        """Test that invalid record raises ValueError."""
        with pytest.raises(ValueError, match="record must be a dictionary"):
            storage.add("not a dict")
        
        with pytest.raises(ValueError, match="record must contain an 'id' field"):
            storage.add({"name": "No ID"})


class TestRecordStorageDelete:
    """Test delete operations."""
    
    def test_delete_existing_record(self, storage, sample_records):
        """Test deleting an existing record."""
        storage.write(sample_records)
        
        result = storage.delete("2")
        
        assert result is True
        records = storage.read()
        assert len(records) == 2
        assert not any(r["id"] == "2" for r in records)
    
    def test_delete_nonexistent_record(self, storage, sample_records):
        """Test deleting a non-existent record."""
        storage.write(sample_records)
        
        result = storage.delete("999")
        assert result is False
        assert len(storage.read()) == 3
    
    def test_delete_invalid_id_raises_error(self, storage):
        """Test that invalid record_id raises ValueError."""
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.delete("")
        
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.delete(None)


class TestRecordStorageGet:
    """Test get operations."""
    
    def test_get_existing_record(self, storage, sample_records):
        """Test getting an existing record."""
        storage.write(sample_records)
        
        record = storage.get("2")
        assert record is not None
        assert record["id"] == "2"
        assert record["name"] == "Bob"
    
    def test_get_nonexistent_record(self, storage, sample_records):
        """Test getting a non-existent record."""
        storage.write(sample_records)
        
        record = storage.get("999")
        assert record is None
    
    def test_get_invalid_id_raises_error(self, storage):
        """Test that invalid record_id raises ValueError."""
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.get("")
        
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.get(None)


class TestRecordStorageErrorHandling:
    """Test error handling and edge cases."""
    
    def test_context_manager_closes_file(self, storage, sample_records):
        """Test that context manager properly closes files."""
        storage.write(sample_records)
        
        records = storage.read()
        assert records == sample_records
        
        records2 = storage.read()
        assert records2 == sample_records
    
    def test_concurrent_operations(self, storage, sample_records):
        """Test that operations work correctly in sequence."""
        storage.write(sample_records)
        storage.update("1", {"age": 31})
        storage.add({"id": "4", "name": "David"})
        storage.delete("2")
        
        records = storage.read()
        assert len(records) == 3
        assert any(r["id"] == "1" and r["age"] == 31 for r in records)
        assert any(r["id"] == "4" for r in records)
        assert not any(r["id"] == "2" for r in records)


