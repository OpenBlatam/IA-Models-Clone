"""
Tests for cache manager.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from piel_mejorador_ai_sam3.core.cache_manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
class TestCacheManager:
    """Tests for CacheManager."""
    
    async def test_set_and_get(self, temp_cache_dir):
        """Test setting and getting from cache."""
        cache = CacheManager(cache_dir=temp_cache_dir, default_ttl_hours=1)
        
        # Create a test file
        test_file = temp_cache_dir / "test.jpg"
        test_file.write_bytes(b"test data")
        
        result = {"enhanced": True, "tokens": 100}
        
        # Set cache
        await cache.set(
            file_path=str(test_file),
            enhancement_level="high",
            result=result
        )
        
        # Get from cache
        cached = await cache.get(
            file_path=str(test_file),
            enhancement_level="high"
        )
        
        assert cached is not None
        assert cached["enhanced"] == True
    
    async def test_cache_miss(self, temp_cache_dir):
        """Test cache miss."""
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        test_file = temp_cache_dir / "test.jpg"
        test_file.write_bytes(b"test data")
        
        # Try to get non-existent cache
        cached = await cache.get(
            file_path=str(test_file),
            enhancement_level="high"
        )
        
        assert cached is None
    
    async def test_cache_expiration(self, temp_cache_dir):
        """Test cache expiration."""
        from datetime import timedelta
        
        cache = CacheManager(cache_dir=temp_cache_dir, default_ttl_hours=0.0001)  # Very short TTL
        
        test_file = temp_cache_dir / "test.jpg"
        test_file.write_bytes(b"test data")
        
        # Set cache
        await cache.set(
            file_path=str(test_file),
            enhancement_level="high",
            result={"test": True},
            ttl=timedelta(seconds=0.1)  # 100ms TTL
        )
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        cached = await cache.get(
            file_path=str(test_file),
            enhancement_level="high"
        )
        
        assert cached is None
    
    async def test_get_stats(self, temp_cache_dir):
        """Test getting cache statistics."""
        cache = CacheManager(cache_dir=temp_cache_dir)
        
        stats = cache.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "cache_size" in stats




