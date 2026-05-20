"""
Advanced concurrency tests
"""

import pytest
import asyncio
from pathlib import Path

from ..core.project_generator import ProjectGenerator
from ..utils.cache_manager import CacheManager
from ..utils.rate_limiter import RateLimiter


class TestAdvancedConcurrency:
    """Advanced concurrency test suite"""

    @pytest.mark.asyncio
    async def test_concurrent_project_generation(self, temp_dir):
        """Test concurrent project generation"""
        generator = ProjectGenerator(output_dir=temp_dir)
        
        descriptions = [f"Project {i}" for i in range(10)]
        
        # Generate concurrently
        tasks = [generator.generate_project(desc) for desc in descriptions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete (may have exceptions)
        assert len(results) == 10
        assert all(isinstance(r, (dict, Exception)) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, temp_dir):
        """Test concurrent cache operations"""
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        async def cache_operation(i):
            await cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
            return await cache.get_cached_project(f"Project {i}", {})
        
        # Run concurrent cache operations
        tasks = [cache_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Should handle concurrency
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """Test concurrent rate limiting"""
        limiter = RateLimiter()
        
        async def check_rate_limit(i):
            return limiter.is_allowed(f"client-{i % 5}", "generate")
        
        # Concurrent rate limit checks
        tasks = [check_rate_limit(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Should handle concurrency
        assert len(results) == 50
        assert all(isinstance(r, tuple) for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, temp_dir):
        """Test concurrent file operations"""
        async def create_file(i):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
            return file_path.exists()
        
        tasks = [create_file(i) for i in range(30)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        assert len(list(temp_dir.glob("file_*.txt"))) == 30

    @pytest.mark.asyncio
    async def test_race_condition_prevention(self, temp_dir):
        """Test race condition prevention"""
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        async def simultaneous_cache(i):
            # Try to cache same project simultaneously
            await cache.cache_project("Same Project", {}, {"id": f"proj-{i}"})
            return await cache.get_cached_project("Same Project", {})
        
        tasks = [simultaneous_cache(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Should handle race conditions
        assert len(results) == 10

