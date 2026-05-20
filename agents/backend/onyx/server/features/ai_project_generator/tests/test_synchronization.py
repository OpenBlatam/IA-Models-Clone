"""
Synchronization and consistency tests
"""

import pytest
from pathlib import Path
import asyncio
from typing import Dict, Any


class TestSynchronization:
    """Tests for synchronization and consistency"""
    
    @pytest.mark.async
    async def test_concurrent_writes(self, temp_dir):
        """Test concurrent writes to same resource"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        # Concurrent writes
        async def write_operation(i):
            await cache.cache_project(
                f"Project {i}",
                {},
                {"id": f"proj-{i}", "data": f"value-{i}"}
            )
            return i
        
        tasks = [write_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)
    
    @pytest.mark.async
    async def test_read_write_consistency(self, temp_dir):
        """Test read-write consistency"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        # Write
        await cache.cache_project("Test", {}, {"id": "test-123", "value": 42})
        
        # Read immediately
        result = await cache.get_cached_project("Test", {})
        
        # Should be consistent
        assert result is not None
        assert result.get("id") == "test-123"
        assert result.get("value") == 42
    
    def test_file_system_sync(self, temp_dir):
        """Test file system synchronization"""
        test_file = temp_dir / "sync_test.txt"
        
        # Write
        test_file.write_text("content")
        
        # Read
        content = test_file.read_text(encoding="utf-8")
        
        # Should be synchronized
        assert content == "content"
    
    @pytest.mark.async
    async def test_state_consistency(self, temp_dir):
        """Test state consistency across operations"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        # Multiple operations
        await cache.cache_project("P1", {}, {"id": "1"})
        await cache.cache_project("P2", {}, {"id": "2"})
        await cache.cache_project("P3", {}, {"id": "3"})
        
        # Read all
        p1 = await cache.get_cached_project("P1", {})
        p2 = await cache.get_cached_project("P2", {})
        p3 = await cache.get_cached_project("P3", {})
        
        # Should be consistent
        assert p1 is not None
        assert p2 is not None
        assert p3 is not None
        assert p1.get("id") == "1"
        assert p2.get("id") == "2"
        assert p3.get("id") == "3"

