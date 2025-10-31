"""
Simple test data management and cleanup system.
"""
import os
import json
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Generator, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import pickle
import sqlite3
from pathlib import Path
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import CopywritingInput, CopywritingOutput, Feedback
from tests.test_utils import TestDataFactory


@dataclass
class TestDataEntry:
    """Represents a test data entry."""
    id: str
    data: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    category: str
    tags: List[str]
    size_bytes: int


class SimpleTestDataManager:
    """Simple test data management system."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="test_data_")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.metadata_file = os.path.join(self.base_dir, "metadata.json")
        self.entries: Dict[str, TestDataEntry] = {}
        self._ensure_directories()
        self._load_metadata()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_metadata(self):
        """Load metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    for entry_data in metadata.get('entries', []):
                        entry = TestDataEntry(
                            id=entry_data['id'],
                            data=entry_data['data'],
                            created_at=datetime.fromisoformat(entry_data['created_at']),
                            last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                            access_count=entry_data['access_count'],
                            category=entry_data['category'],
                            tags=entry_data['tags'],
                            size_bytes=entry_data['size_bytes']
                        )
                        self.entries[entry.id] = entry
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata to file."""
        metadata = {
            'entries': [
                {
                    'id': entry.id,
                    'data': entry.data,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'access_count': entry.access_count,
                    'category': entry.category,
                    'tags': entry.tags,
                    'size_bytes': entry.size_bytes
                }
                for entry in self.entries.values()
            ]
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_entry(self, data: Dict[str, Any], category: str = "test", tags: List[str] = None) -> str:
        """Create a new test data entry."""
        entry_id = hashlib.md5(str(data).encode()).hexdigest()[:16]
        
        # Convert datetime objects to ISO strings for JSON serialization
        serializable_data = self._make_json_serializable(data)
        size_bytes = len(json.dumps(serializable_data).encode())
        
        entry = TestDataEntry(
            id=entry_id,
            data=serializable_data,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            category=category,
            tags=tags or [],
            size_bytes=size_bytes
        )
        
        self.entries[entry_id] = entry
        self._save_metadata()
        
        return entry_id
    
    def _make_json_serializable(self, obj):
        """Convert datetime objects to ISO strings for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def get_entry(self, entry_id: str) -> Optional[TestDataEntry]:
        """Get a test data entry by ID."""
        if entry_id in self.entries:
            entry = self.entries[entry_id]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._save_metadata()
            return entry
        return None
    
    def list_entries(self, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[TestDataEntry]:
        """List test data entries with optional filtering."""
        entries = list(self.entries.values())
        
        if category:
            entries = [e for e in entries if e.category == category]
        
        if tags:
            entries = [e for e in entries if all(tag in e.tags for tag in tags)]
        
        return entries
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a test data entry."""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self._save_metadata()
            return True
        return False
    
    def cleanup_old_entries(self, max_age_hours: int = 24) -> int:
        """Clean up entries older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_entries = [
            entry_id for entry_id, entry in self.entries.items()
            if entry.created_at < cutoff_time
        ]
        
        for entry_id in old_entries:
            del self.entries[entry_id]
        
        if old_entries:
            self._save_metadata()
        
        return len(old_entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored data."""
        if not self.entries:
            return {
                "total_entries": 0,
                "total_size_bytes": 0,
                "categories": {},
                "tags": {}
            }
        
        total_size = sum(entry.size_bytes for entry in self.entries.values())
        
        categories = {}
        for entry in self.entries.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        
        tags = {}
        for entry in self.entries.values():
            for tag in entry.tags:
                tags[tag] = tags.get(tag, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "total_size_bytes": total_size,
            "categories": categories,
            "tags": tags
        }
    
    def cleanup(self):
        """Clean up all test data."""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        self.entries.clear()


class TestDataManagerTests:
    """Test cases for test data manager."""
    
    @pytest.fixture
    def data_manager(self):
        """Create a test data manager."""
        manager = SimpleTestDataManager()
        yield manager
        manager.cleanup()
    
    def test_data_manager_initialization(self, data_manager):
        """Test data manager initialization."""
        assert data_manager is not None
        assert data_manager.base_dir is not None
        assert data_manager.data_dir is not None
        assert data_manager.metadata_file is not None
        assert isinstance(data_manager.entries, dict)
    
    def test_create_entry(self, data_manager):
        """Test creating a test data entry."""
        test_data = {"test": "data", "value": 123}
        entry_id = data_manager.create_entry(test_data, "test_category", ["tag1", "tag2"])
        
        assert entry_id is not None
        assert entry_id in data_manager.entries
        
        entry = data_manager.entries[entry_id]
        assert entry.data == test_data
        assert entry.category == "test_category"
        assert entry.tags == ["tag1", "tag2"]
        assert entry.access_count == 0
        assert entry.size_bytes > 0
    
    def test_get_entry(self, data_manager):
        """Test getting a test data entry."""
        test_data = {"test": "data"}
        entry_id = data_manager.create_entry(test_data)
        
        entry = data_manager.get_entry(entry_id)
        assert entry is not None
        assert entry.data == test_data
        assert entry.access_count == 1  # Should be incremented
        
        # Test getting non-existent entry
        non_existent = data_manager.get_entry("non_existent")
        assert non_existent is None
    
    def test_list_entries(self, data_manager):
        """Test listing test data entries."""
        # Create multiple entries
        data_manager.create_entry({"data1": "value1"}, "category1", ["tag1"])
        data_manager.create_entry({"data2": "value2"}, "category2", ["tag2"])
        data_manager.create_entry({"data3": "value3"}, "category1", ["tag1", "tag3"])
        
        # List all entries
        all_entries = data_manager.list_entries()
        assert len(all_entries) == 3
        
        # List by category
        category1_entries = data_manager.list_entries(category="category1")
        assert len(category1_entries) == 2
        
        # List by tags
        tag1_entries = data_manager.list_entries(tags=["tag1"])
        assert len(tag1_entries) == 2
        
        # List by category and tags
        filtered_entries = data_manager.list_entries(category="category1", tags=["tag1"])
        assert len(filtered_entries) == 2
    
    def test_delete_entry(self, data_manager):
        """Test deleting a test data entry."""
        test_data = {"test": "data"}
        entry_id = data_manager.create_entry(test_data)
        
        assert entry_id in data_manager.entries
        
        # Delete entry
        result = data_manager.delete_entry(entry_id)
        assert result is True
        assert entry_id not in data_manager.entries
        
        # Try to delete non-existent entry
        result = data_manager.delete_entry("non_existent")
        assert result is False
    
    def test_cleanup_old_entries(self, data_manager):
        """Test cleaning up old entries."""
        # Create an entry
        test_data = {"test": "data"}
        entry_id = data_manager.create_entry(test_data)
        
        # Clean up entries older than 0 hours (should clean up all)
        cleaned_count = data_manager.cleanup_old_entries(max_age_hours=0)
        assert cleaned_count == 1
        assert len(data_manager.entries) == 0
    
    def test_get_stats(self, data_manager):
        """Test getting statistics."""
        # Test empty stats
        stats = data_manager.get_stats()
        assert stats["total_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["categories"] == {}
        assert stats["tags"] == {}
        
        # Create some entries
        data_manager.create_entry({"data1": "value1"}, "category1", ["tag1"])
        data_manager.create_entry({"data2": "value2"}, "category2", ["tag2"])
        data_manager.create_entry({"data3": "value3"}, "category1", ["tag1", "tag3"])
        
        # Get stats
        stats = data_manager.get_stats()
        assert stats["total_entries"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["categories"]["category1"] == 2
        assert stats["categories"]["category2"] == 1
        assert stats["tags"]["tag1"] == 2
        assert stats["tags"]["tag2"] == 1
        assert stats["tags"]["tag3"] == 1
    
    def test_metadata_persistence(self, data_manager):
        """Test metadata persistence."""
        # Create an entry
        test_data = {"test": "data"}
        entry_id = data_manager.create_entry(test_data, "test_category", ["tag1"])
        
        # Create a new manager instance (should load metadata)
        new_manager = SimpleTestDataManager(data_manager.base_dir)
        
        # Check that entry was loaded
        assert entry_id in new_manager.entries
        entry = new_manager.entries[entry_id]
        assert entry.data == test_data
        assert entry.category == "test_category"
        assert entry.tags == ["tag1"]
        
        new_manager.cleanup()
    
    def test_copywriting_data_integration(self, data_manager):
        """Test integration with copywriting data."""
        # Create copywriting input
        copywriting_input = TestDataFactory.create_copywriting_input()
        input_data = copywriting_input.model_dump()
        
        # Store in data manager
        entry_id = data_manager.create_entry(input_data, "copywriting_input", ["test", "input"])
        
        # Retrieve and validate
        entry = data_manager.get_entry(entry_id)
        assert entry is not None
        assert entry.category == "copywriting_input"
        assert entry.tags == ["test", "input"]
        
        # Validate data structure
        assert "product_description" in entry.data
        assert "target_platform" in entry.data
        assert "content_type" in entry.data
    
    def test_feedback_data_integration(self, data_manager):
        """Test integration with feedback data."""
        # Create feedback
        feedback = TestDataFactory.create_feedback()
        feedback_data = feedback.model_dump()
        
        # Store in data manager
        entry_id = data_manager.create_entry(feedback_data, "feedback", ["test", "feedback"])
        
        # Retrieve and validate
        entry = data_manager.get_entry(entry_id)
        assert entry is not None
        assert entry.category == "feedback"
        assert entry.tags == ["test", "feedback"]
        
        # Validate data structure
        assert "type" in entry.data
        assert "score" in entry.data
        assert "comments" in entry.data
    
    def test_data_manager_cleanup(self, data_manager):
        """Test data manager cleanup."""
        # Create some entries
        data_manager.create_entry({"data1": "value1"})
        data_manager.create_entry({"data2": "value2"})
        
        assert len(data_manager.entries) == 2
        
        # Cleanup
        data_manager.cleanup()
        
        # Check that base directory is removed
        assert not os.path.exists(data_manager.base_dir)
        assert len(data_manager.entries) == 0
    
    def test_entry_access_tracking(self, data_manager):
        """Test entry access tracking."""
        test_data = {"test": "data"}
        entry_id = data_manager.create_entry(test_data)

        # Access entry multiple times
        entry1 = data_manager.get_entry(entry_id)
        assert entry1.access_count == 1
        
        entry2 = data_manager.get_entry(entry_id)
        assert entry2.access_count == 2
        
        entry3 = data_manager.get_entry(entry_id)
        assert entry3.access_count == 3
        
        # Check last accessed time
        assert entry1.last_accessed <= entry2.last_accessed <= entry3.last_accessed
    
    def test_entry_size_calculation(self, data_manager):
        """Test entry size calculation."""
        # Test small data
        small_data = {"test": "data"}
        entry_id1 = data_manager.create_entry(small_data)
        entry1 = data_manager.entries[entry_id1]
        assert entry1.size_bytes > 0
        
        # Test larger data
        large_data = {"test": "data" * 1000}
        entry_id2 = data_manager.create_entry(large_data)
        entry2 = data_manager.entries[entry_id2]
        assert entry2.size_bytes > entry1.size_bytes
    
    def test_category_filtering(self, data_manager):
        """Test category filtering."""
        # Create entries with different categories
        data_manager.create_entry({"data1": "value1"}, "category1")
        data_manager.create_entry({"data2": "value2"}, "category2")
        data_manager.create_entry({"data3": "value3"}, "category1")
        
        # Filter by category
        category1_entries = data_manager.list_entries(category="category1")
        assert len(category1_entries) == 2
        
        category2_entries = data_manager.list_entries(category="category2")
        assert len(category2_entries) == 1
        
        # Filter by non-existent category
        non_existent_entries = data_manager.list_entries(category="non_existent")
        assert len(non_existent_entries) == 0
    
    def test_tag_filtering(self, data_manager):
        """Test tag filtering."""
        # Create entries with different tags
        data_manager.create_entry({"data1": "value1"}, "test", ["tag1"])
        data_manager.create_entry({"data2": "value2"}, "test", ["tag2"])
        data_manager.create_entry({"data3": "value3"}, "test", ["tag1", "tag2"])

        # Filter by single tag
        tag1_entries = data_manager.list_entries(tags=["tag1"])
        assert len(tag1_entries) == 2

        tag2_entries = data_manager.list_entries(tags=["tag2"])
        assert len(tag2_entries) == 2

        # Filter by multiple tags - should return entries that have ALL specified tags
        both_tags_entries = data_manager.list_entries(tags=["tag1", "tag2"])
        assert len(both_tags_entries) == 1
        
        # Filter by non-existent tag
        non_existent_entries = data_manager.list_entries(tags=["non_existent"])
        assert len(non_existent_entries) == 0
