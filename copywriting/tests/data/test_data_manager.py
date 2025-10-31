"""
Advanced test data management and cleanup system.
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

from tests.base import BaseTestClass
from tests.config.test_config import test_config_manager, test_data_config


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
    ttl_seconds: Optional[int] = None


class TestDataManager:
    """Advanced test data management system."""
    
    def __init__(self, cache_size: int = 1000, cleanup_interval: int = 100):
        self.cache_size = cache_size
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, TestDataEntry] = {}
        self._access_counter = 0
        self._temp_dir = None
        self._db_path = None
        self._db_connection = None
        
        # Initialize temporary directory
        self._setup_temp_directory()
        self._setup_database()
    
    def _setup_temp_directory(self):
        """Setup temporary directory for test data."""
        self._temp_dir = tempfile.mkdtemp(prefix="copywriting_test_")
        os.makedirs(self._temp_dir, exist_ok=True)
    
    def _setup_database(self):
        """Setup SQLite database for test data management."""
        self._db_path = os.path.join(self._temp_dir, "test_data.db")
        self._db_connection = sqlite3.connect(self._db_path)
        
        # Create tables
        cursor = self._db_connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                id TEXT PRIMARY KEY,
                data BLOB,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER,
                category TEXT,
                tags TEXT,
                size_bytes INTEGER,
                ttl_seconds INTEGER
            )
        """)
        self._db_connection.commit()
    
    def _generate_id(self, data: Dict[str, Any], category: str) -> str:
        """Generate unique ID for test data."""
        content = json.dumps(data, sort_keys=True) + category
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_size(self, data: Dict[str, Any]) -> int:
        """Calculate size of data in bytes."""
        return len(pickle.dumps(data))
    
    def _is_expired(self, entry: TestDataEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl_seconds is None:
            return False
        
        expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.now() > expiry_time
    
    def _cleanup_expired(self):
        """Cleanup expired entries."""
        expired_ids = []
        for entry_id, entry in self._cache.items():
            if self._is_expired(entry):
                expired_ids.append(entry_id)
        
        for entry_id in expired_ids:
            del self._cache[entry_id]
    
    def _cleanup_lru(self):
        """Cleanup least recently used entries."""
        if len(self._cache) <= self.cache_size:
            return
        
        # Sort by last accessed time and access count
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        # Remove oldest entries
        entries_to_remove = len(self._cache) - self.cache_size
        for i in range(entries_to_remove):
            entry_id = sorted_entries[i][0]
            del self._cache[entry_id]
    
    def _cleanup_if_needed(self):
        """Cleanup cache if needed."""
        self._access_counter += 1
        
        if self._access_counter % self.cleanup_interval == 0:
            self._cleanup_expired()
            self._cleanup_lru()
    
    def store(self, data: Dict[str, Any], category: str, tags: List[str] = None, ttl_seconds: int = None) -> str:
        """Store test data."""
        if tags is None:
            tags = []
        
        entry_id = self._generate_id(data, category)
        now = datetime.now()
        
        entry = TestDataEntry(
            id=entry_id,
            data=data,
            created_at=now,
            last_accessed=now,
            access_count=1,
            category=category,
            tags=tags,
            size_bytes=self._calculate_size(data),
            ttl_seconds=ttl_seconds
        )
        
        self._cache[entry_id] = entry
        self._cleanup_if_needed()
        
        return entry_id
    
    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get test data by ID."""
        if entry_id not in self._cache:
            return None
        
        entry = self._cache[entry_id]
        
        if self._is_expired(entry):
            del self._cache[entry_id]
            return None
        
        # Update access info
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        return entry.data
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all test data by category."""
        results = []
        for entry in self._cache.values():
            if entry.category == category and not self._is_expired(entry):
                results.append(entry.data)
        return results
    
    def get_by_tags(self, tags: List[str]) -> List[Dict[str, Any]]:
        """Get test data by tags."""
        results = []
        for entry in self._cache.values():
            if any(tag in entry.tags for tag in tags) and not self._is_expired(entry):
                results.append(entry.data)
        return results
    
    def delete(self, entry_id: str) -> bool:
        """Delete test data by ID."""
        if entry_id in self._cache:
            del self._cache[entry_id]
            return True
        return False
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        category_counts = {}
        tag_counts = {}
        
        for entry in self._cache.values():
            category_counts[entry.category] = category_counts.get(entry.category, 0) + 1
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "total_entries": len(self._cache),
            "total_size_bytes": total_size,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "access_counter": self._access_counter
        }
    
    def export_data(self, file_path: str):
        """Export test data to file."""
        export_data = {
            "entries": {entry_id: asdict(entry) for entry_id, entry in self._cache.items()},
            "stats": self.get_stats(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def import_data(self, file_path: str):
        """Import test data from file."""
        with open(file_path, 'r') as f:
            import_data = json.load(f)
        
        for entry_id, entry_data in import_data.get("entries", {}).items():
            # Convert datetime strings back to datetime objects
            entry_data["created_at"] = datetime.fromisoformat(entry_data["created_at"])
            entry_data["last_accessed"] = datetime.fromisoformat(entry_data["last_accessed"])
            
            entry = TestDataEntry(**entry_data)
            self._cache[entry_id] = entry
    
    def cleanup(self):
        """Cleanup resources."""
        if self._db_connection:
            self._db_connection.close()
        
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
    
    def __del__(self):
        """Destructor to cleanup resources."""
        self.cleanup()


class TestDataFactory:
    """Factory for creating test data."""
    
    def __init__(self, data_manager: TestDataManager):
        self.data_manager = data_manager
        self.config = test_config_manager.get_config()
    
    def create_request_data(self, **overrides) -> Dict[str, Any]:
        """Create copywriting request data."""
        base_data = {
            "product_description": test_data_config.get_random_product(),
            "target_platform": test_data_config.get_random_platform(),
            "tone": test_data_config.get_random_tone(),
            "target_audience": test_data_config.get_random_audience(),
            "key_points": test_data_config.get_random_key_points(),
            "instructions": test_data_config.get_random_instructions(),
            "restrictions": test_data_config.get_random_restrictions(),
            "creativity_level": 0.8,
            "language": "es"
        }
        
        base_data.update(overrides)
        return base_data
    
    def create_response_data(self, **overrides) -> Dict[str, Any]:
        """Create copywriting response data."""
        base_data = {
            "variants": [
                {
                    "headline": "¡Descubre la Innovación!",
                    "primary_text": "Producto revolucionario para tu vida",
                    "call_to_action": "Compra ahora",
                    "hashtags": ["#innovación", "#producto"]
                }
            ],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 2.5,
            "extra_metadata": {"tokens_used": 150}
        }
        
        base_data.update(overrides)
        return base_data
    
    def create_batch_data(self, count: int = 3) -> List[Dict[str, Any]]:
        """Create batch request data."""
        return [
            self.create_request_data(product_description=f"Producto {i}")
            for i in range(count)
        ]
    
    def create_performance_data(self, count: int = 20) -> List[Dict[str, Any]]:
        """Create performance test data."""
        return [
            self.create_request_data(product_description=f"Performance test product {i}")
            for i in range(count)
        ]
    
    def create_load_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """Create load test data."""
        return [
            self.create_request_data(product_description=f"Load test product {i}")
            for i in range(count)
        ]
    
    def create_security_data(self) -> Dict[str, List[str]]:
        """Create security test data."""
        return {
            "malicious_inputs": [
                "'; DROP TABLE users; --",
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "../../../etc/passwd",
                "{{7*7}}",
                "${7*7}",
                "`id`",
                "$(id)"
            ],
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'--",
                "'; INSERT INTO users VALUES ('hacker', 'password'); --"
            ],
            "xss_inputs": [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "';alert('XSS');//"
            ]
        }
    
    def store_request_data(self, data: Dict[str, Any], tags: List[str] = None) -> str:
        """Store request data in cache."""
        if tags is None:
            tags = ["request", "copywriting"]
        
        return self.data_manager.store(data, "request", tags)
    
    def store_response_data(self, data: Dict[str, Any], tags: List[str] = None) -> str:
        """Store response data in cache."""
        if tags is None:
            tags = ["response", "copywriting"]
        
        return self.data_manager.store(data, "response", tags)
    
    def store_performance_data(self, data: List[Dict[str, Any]], tags: List[str] = None) -> str:
        """Store performance test data in cache."""
        if tags is None:
            tags = ["performance", "load_test"]
        
        return self.data_manager.store(data, "performance", tags)
    
    def store_security_data(self, data: Dict[str, List[str]], tags: List[str] = None) -> str:
        """Store security test data in cache."""
        if tags is None:
            tags = ["security", "malicious"]
        
        return self.data_manager.store(data, "security", tags)


class TestDataCleanup:
    """Test data cleanup utilities."""
    
    def __init__(self, data_manager: TestDataManager):
        self.data_manager = data_manager
    
    def cleanup_expired(self):
        """Cleanup expired test data."""
        expired_ids = []
        for entry_id, entry in self.data_manager._cache.items():
            if entry.ttl_seconds and self._is_expired(entry):
                expired_ids.append(entry_id)
        
        for entry_id in expired_ids:
            self.data_manager.delete(entry_id)
        
        return len(expired_ids)
    
    def cleanup_by_category(self, category: str):
        """Cleanup test data by category."""
        entries_to_delete = []
        for entry_id, entry in self.data_manager._cache.items():
            if entry.category == category:
                entries_to_delete.append(entry_id)
        
        for entry_id in entries_to_delete:
            self.data_manager.delete(entry_id)
        
        return len(entries_to_delete)
    
    def cleanup_by_tags(self, tags: List[str]):
        """Cleanup test data by tags."""
        entries_to_delete = []
        for entry_id, entry in self.data_manager._cache.items():
            if any(tag in entry.tags for tag in tags):
                entries_to_delete.append(entry_id)
        
        for entry_id in entries_to_delete:
            self.data_manager.delete(entry_id)
        
        return len(entries_to_delete)
    
    def cleanup_old_entries(self, days: int = 7):
        """Cleanup entries older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        entries_to_delete = []
        
        for entry_id, entry in self.data_manager._cache.items():
            if entry.created_at < cutoff_date:
                entries_to_delete.append(entry_id)
        
        for entry_id in entries_to_delete:
            self.data_manager.delete(entry_id)
        
        return len(entries_to_delete)
    
    def cleanup_large_entries(self, size_threshold_mb: int = 1):
        """Cleanup entries larger than specified size."""
        size_threshold_bytes = size_threshold_mb * 1024 * 1024
        entries_to_delete = []
        
        for entry_id, entry in self.data_manager._cache.items():
            if entry.size_bytes > size_threshold_bytes:
                entries_to_delete.append(entry_id)
        
        for entry_id in entries_to_delete:
            self.data_manager.delete(entry_id)
        
        return len(entries_to_delete)
    
    def _is_expired(self, entry: TestDataEntry) -> bool:
        """Check if entry is expired."""
        if entry.ttl_seconds is None:
            return False
        
        expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.now() > expiry_time


# Global instances
test_data_manager = TestDataManager(
    cache_size=test_config_manager.get_config().test_data_cache_size,
    cleanup_interval=test_config_manager.get_config().test_data_cleanup_interval
)

test_data_factory = TestDataFactory(test_data_manager)
test_data_cleanup = TestDataCleanup(test_data_manager)
