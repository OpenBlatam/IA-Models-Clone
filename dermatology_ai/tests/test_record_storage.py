"""
Tests for RecordStorage class
Tests the refactored record storage implementation
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from utils.record_storage import RecordStorage


class TestRecordStorage:
    """Test suite for RecordStorage class"""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        fd, path = tempfile.mkstemp(suffix='.json')
        yield path
        Path(path).unlink(missing_ok=True)
    
    @pytest.fixture
    def storage(self, temp_file):
        """Create a RecordStorage instance with temp file"""
        return RecordStorage(temp_file)
    
    def test_initialization_creates_file(self, temp_file):
        """Test that initialization creates file if it doesn't exist"""
        storage = RecordStorage(temp_file)
        assert Path(temp_file).exists()
        
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data == {"records": []}
    
    def test_initialization_with_invalid_path(self):
        """Test that initialization raises error with invalid path"""
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            RecordStorage("")
        
        with pytest.raises(ValueError, match="file_path must be a non-empty string"):
            RecordStorage(None)
    
    def test_read_empty_file(self, storage):
        """Test reading from empty file"""
        records = storage.read()
        assert records == []
    
    def test_write_and_read(self, storage):
        """Test writing and reading records"""
        test_records = [
            {"id": "1", "name": "Test 1"},
            {"id": "2", "name": "Test 2"}
        ]
        
        storage.write(test_records)
        records = storage.read()
        
        assert len(records) == 2
        assert records[0]["id"] == "1"
        assert records[1]["id"] == "2"
    
    def test_write_invalid_input(self, storage):
        """Test that write validates input"""
        with pytest.raises(ValueError, match="records must be a list"):
            storage.write("not a list")
        
        with pytest.raises(ValueError, match="is not a dictionary"):
            storage.write(["not a dict", "also not a dict"])
    
    def test_update_existing_record(self, storage):
        """Test updating an existing record"""
        initial_records = [
            {"id": "1", "name": "Original", "value": 10},
            {"id": "2", "name": "Other", "value": 20}
        ]
        storage.write(initial_records)
        
        result = storage.update("1", {"name": "Updated", "value": 15})
        
        assert result is True
        records = storage.read()
        assert records[0]["name"] == "Updated"
        assert records[0]["value"] == 15
        assert records[0]["id"] == "1"  # ID preserved
        assert records[1] == initial_records[1]  # Other record unchanged
    
    def test_update_nonexistent_record(self, storage):
        """Test updating a non-existent record"""
        initial_records = [{"id": "1", "name": "Test"}]
        storage.write(initial_records)
        
        result = storage.update("999", {"name": "Updated"})
        
        assert result is False
        records = storage.read()
        assert records == initial_records
    
    def test_update_merges_instead_of_replaces(self, storage):
        """Test that update merges fields instead of replacing entire record"""
        initial_records = [
            {"id": "1", "name": "Original", "value": 10, "extra": "preserved"}
        ]
        storage.write(initial_records)
        
        storage.update("1", {"value": 20})
        records = storage.read()
        
        assert records[0]["name"] == "Original"  # Preserved
        assert records[0]["value"] == 20  # Updated
        assert records[0]["extra"] == "preserved"  # Preserved
        assert records[0]["id"] == "1"  # ID preserved
    
    def test_update_invalid_input(self, storage):
        """Test that update validates input"""
        with pytest.raises(ValueError, match="record_id must be a non-empty string"):
            storage.update("", {"name": "Test"})
        
        with pytest.raises(ValueError, match="updates must be a dictionary"):
            storage.update("1", "not a dict")
        
        result = storage.update("1", {})
        assert result is False
    
    def test_context_manager_handles_exceptions(self, storage):
        """Test that context managers properly handle exceptions"""
        # Create a file that will cause read error
        with open(storage.file_path, 'w') as f:
            f.write("invalid json {")
        
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            storage.read()
    
    def test_file_operations_use_context_managers(self, storage, temp_file):
        """Test that all file operations use context managers"""
        # This test verifies that files are properly closed
        # by checking that we can write and read multiple times
        for i in range(5):
            records = [{"id": str(i), "iteration": i}]
            storage.write(records)
            read_records = storage.read()
            assert len(read_records) == 1
    
    def test_encoding_handles_unicode(self, storage):
        """Test that file operations handle unicode correctly"""
        unicode_records = [
            {"id": "1", "name": "Test Español", "description": "Descripción con ñ y acentos"}
        ]
        
        storage.write(unicode_records)
        records = storage.read()
        
        assert records[0]["name"] == "Test Español"
        assert records[0]["description"] == "Descripción con ñ y acentos"
    
    def test_multiple_updates(self, storage):
        """Test multiple sequential updates"""
        initial_records = [{"id": "1", "value": 0}]
        storage.write(initial_records)
        
        for i in range(1, 6):
            storage.update("1", {"value": i})
            records = storage.read()
            assert records[0]["value"] == i

