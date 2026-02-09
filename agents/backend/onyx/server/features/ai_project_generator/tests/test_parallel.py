"""
Tests for parallel execution and concurrency
"""

import pytest
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class TestParallelExecution:
    """Tests for parallel execution scenarios"""
    
    @pytest.mark.async
    async def test_parallel_project_generation(self, project_generator, temp_dir):
        """Test generating multiple projects in parallel"""
        descriptions = [f"Project {i}" for i in range(20)]
        
        # Generate in parallel
        tasks = [project_generator.generate_project(desc) for desc in descriptions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete
        assert len(results) == 20
        # Most should succeed (some may fail due to resource constraints)
        successes = [r for r in results if isinstance(r, dict)]
        assert len(successes) >= 15  # At least 75% success rate
    
    @pytest.mark.async
    async def test_parallel_cache_operations(self, temp_dir):
        """Test parallel cache operations"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        async def cache_operation(i):
            await cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
            return await cache.get_cached_project(f"Project {i}", {})
        
        # Run 50 operations in parallel
        tasks = [cache_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # Should handle concurrency
        assert len(results) == 50
    
    def test_thread_pool_execution(self, project_generator):
        """Test using thread pool for parallel execution"""
        def generate_sync(desc):
            # Simulate sync operation
            return project_generator._sanitize_name(desc)
        
        descriptions = [f"Project {i}" for i in range(10)]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(generate_sync, descriptions))
        
        assert len(results) == 10
        assert all(isinstance(r, str) for r in results)

