"""
Tests for LLM Engine Cache functionality.

This module tests the caching mechanisms for LLM engines in the Blaze AI system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import tempfile
import shutil
from pathlib import Path

# Import the modules to test
from engines.cache import LLMEngineCache
from engines.base import EngineType, EnginePriority


class TestLLMEngineCache(unittest.TestCase):
    """Test cases for LLMEngineCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir()
        
        # Mock the logger to avoid logging issues during tests
        with patch('engines.cache.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            self.cache = LLMEngineCache(cache_directory=str(self.cache_dir))
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LLMEngineCache initialization."""
        self.assertEqual(self.cache.cache_directory, str(self.cache_dir))
        self.assertIsInstance(self.cache.cache, dict)
        self.assertIsInstance(self.cache.cache_stats, dict)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        # Test setting cache
        key = "test_key"
        value = {"data": "test_value"}
        
        self.cache.set(key, value)
        self.assertIn(key, self.cache.cache)
        self.assertEqual(self.cache.cache[key], value)
        
        # Test getting cache
        retrieved = self.cache.get(key)
        self.assertEqual(retrieved, value)
        
        # Test getting non-existent key
        not_found = self.cache.get("nonexistent")
        self.assertIsNone(not_found)
    
    def test_cache_expiration(self):
        """Test cache expiration functionality."""
        key = "expiring_key"
        value = {"data": "expiring_value"}
        
        # Set cache with short TTL
        self.cache.set(key, value, ttl=0.1)  # 100ms
        
        # Should be available immediately
        self.assertIsNotNone(self.cache.get(key))
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        self.assertIsNone(self.cache.get(key))
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        # Add some items to cache
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.assertEqual(len(self.cache.cache), 2)
        
        # Clear cache
        self.cache.clear()
        
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        # Perform some operations
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.get("key1")
        self.cache.get("nonexistent")
        
        stats = self.cache.get_stats()
        
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("sets", stats)
        self.assertIn("total_requests", stats)
        
        self.assertEqual(stats["sets"], 2)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["total_requests"], 2)
    
    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        key = "persistent_key"
        value = {"data": "persistent_value"}
        
        # Set cache
        self.cache.set(key, value)
        
        # Create new cache instance (simulating restart)
        with patch('engines.cache.get_logger') as mock_logger:
            mock_logger.return_value = Mock()
            new_cache = LLMEngineCache(cache_directory=str(self.cache_dir))
        
        # Check if data was persisted
        retrieved = new_cache.get(key)
        self.assertEqual(retrieved, value)
    
    def test_cache_cleanup(self):
        """Test cache cleanup of expired items."""
        # Add items with different TTLs
        self.cache.set("short", "short_value", ttl=0.1)
        self.cache.set("long", "long_value", ttl=10.0)
        
        # Wait for short item to expire
        time.sleep(0.2)
        
        # Trigger cleanup
        self.cache.cleanup()
        
        # Short item should be removed
        self.assertIsNone(self.cache.get("short"))
        
        # Long item should still be there
        self.assertIsNotNone(self.cache.get("long"))
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        # Set cache size limit
        self.cache.max_size = 2
        
        # Add items
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")  # This should trigger eviction
        
        # Check that only 2 items remain
        self.assertEqual(len(self.cache.cache), 2)
        
        # Check that oldest item was evicted
        self.assertNotIn("key1", self.cache.cache)


if __name__ == '__main__':
    unittest.main()
