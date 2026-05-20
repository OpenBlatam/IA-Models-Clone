"""
Tests for CacheManager utility
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

from ..utils.cache_manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager"""

    def test_init(self, temp_dir):
        """Test CacheManager initialization"""
        cache_dir = temp_dir / "cache"
        manager = CacheManager(cache_dir=cache_dir)
        assert manager.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_init_default_dir(self):
        """Test CacheManager with default directory"""
        manager = CacheManager()
        assert manager.cache_dir.exists()

    def test_generate_cache_key(self, temp_dir):
        """Test cache key generation"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        description = "Test project"
        config = {"framework": "fastapi"}
        
        key1 = manager._generate_cache_key(description, config)
        key2 = manager._generate_cache_key(description, config)
        
        # Same inputs should generate same key
        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

    def test_generate_cache_key_different(self, temp_dir):
        """Test that different inputs generate different keys"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        key1 = manager._generate_cache_key("Project 1", {"framework": "fastapi"})
        key2 = manager._generate_cache_key("Project 2", {"framework": "fastapi"})
        
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_get_cached_project_not_found(self, temp_dir):
        """Test getting non-existent cached project"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        result = await manager.get_cached_project(
            "Non-existent project",
            {"framework": "fastapi"}
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_and_get_project(self, temp_dir):
        """Test caching and retrieving a project"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        description = "Test project"
        config = {"framework": "fastapi"}
        project_info = {
            "project_id": "test-123",
            "name": "test_project",
            "description": description
        }
        
        # Cache the project
        await manager.cache_project(description, config, project_info)
        
        # Retrieve it
        cached = await manager.get_cached_project(description, config)
        
        assert cached is not None
        assert cached["project_id"] == "test-123"
        assert cached["name"] == "test_project"

    @pytest.mark.asyncio
    async def test_cache_expiration(self, temp_dir):
        """Test cache expiration"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        description = "Expired project"
        config = {"framework": "fastapi"}
        project_info = {"project_id": "expired-123"}
        
        # Cache the project
        await manager.cache_project(description, config, project_info)
        
        # Manually set expiration to past
        cache_key = manager._generate_cache_key(description, config)
        cache_file = manager.cache_dir / f"{cache_key}.json"
        cache_data = json.loads(cache_file.read_text())
        cache_data["created_at"] = (datetime.now() - timedelta(days=8)).isoformat()
        cache_file.write_text(json.dumps(cache_data))
        
        # Should not find expired cache
        cached = await manager.get_cached_project(description, config)
        assert cached is None
        assert not cache_file.exists()  # Should be deleted

    @pytest.mark.asyncio
    async def test_clear_cache(self, temp_dir):
        """Test clearing cache"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Cache multiple projects
        await manager.cache_project("Project 1", {"f": "fastapi"}, {"id": "1"})
        await manager.cache_project("Project 2", {"f": "fastapi"}, {"id": "2"})
        
        # Verify cache files exist
        cache_files = list(manager.cache_dir.glob("*.json"))
        assert len(cache_files) == 2
        
        # Clear cache
        await manager.clear_cache()
        
        # Verify cache is cleared
        cache_files = list(manager.cache_dir.glob("*.json"))
        assert len(cache_files) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, temp_dir):
        """Test getting cache statistics"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Cache some projects
        await manager.cache_project("Project 1", {"f": "fastapi"}, {"id": "1"})
        await manager.cache_project("Project 2", {"f": "fastapi"}, {"id": "2"})
        
        stats = await manager.get_stats()
        assert stats["total_cached"] == 2
        assert stats["cache_dir"] == str(manager.cache_dir)

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, temp_dir):
        """Test error handling in cache operations"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Try to get from non-existent cache with invalid file
        cache_key = manager._generate_cache_key("Test", {"f": "fastapi"})
        cache_file = manager.cache_dir / f"{cache_key}.json"
        cache_file.write_text("invalid json")
        
        # Should handle error gracefully
        result = await manager.get_cached_project("Test", {"f": "fastapi"})
        assert result is None

