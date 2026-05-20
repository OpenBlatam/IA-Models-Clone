"""
Test optimizations and performance improvements
"""

import pytest
from pathlib import Path
from typing import Dict, Any
import time


class TestOptimizations:
    """Tests for optimizations and performance"""
    
    @pytest.mark.performance
    def test_cache_effectiveness(self, temp_dir):
        """Test that caching improves performance"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        description = "A test project"
        config = {}
        project_info = {"project_id": "test-123"}
        
        # First call (no cache)
        start = time.time()
        import asyncio
        asyncio.run(cache.cache_project(description, config, project_info))
        first_time = time.time() - start
        
        # Second call (with cache)
        start = time.time()
        cached = asyncio.run(cache.get_cached_project(description, config))
        cached_time = time.time() - start
        
        # Cached should be faster
        assert cached_time < first_time
        assert cached is not None
    
    @pytest.mark.performance
    def test_batch_operations(self, temp_dir):
        """Test that batch operations are efficient"""
        from ..core.project_generator import ProjectGenerator
        
        generator = ProjectGenerator(base_output_dir=str(temp_dir / "projects"))
        
        descriptions = [f"Project {i}" for i in range(10)]
        
        # Measure batch time
        start = time.time()
        import asyncio
        tasks = [generator.generate_project(desc) for desc in descriptions]
        results = asyncio.run(asyncio.gather(*tasks))
        batch_time = time.time() - start
        
        # Should complete in reasonable time
        assert batch_time < 60.0  # 60 seconds for 10 projects
        assert len(results) == 10
    
    @pytest.mark.unit
    def test_memory_efficiency(self, temp_dir):
        """Test memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform operations
        from ..utils.cache_manager import CacheManager
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        for i in range(100):
            import asyncio
            asyncio.run(cache.cache_project(
                f"Project {i}",
                {},
                {"id": f"proj-{i}"}
            ))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
    
    @pytest.mark.unit
    def test_concurrent_efficiency(self, temp_dir):
        """Test that concurrent operations are efficient"""
        from ..utils.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        
        # Measure concurrent checks
        start = time.time()
        import asyncio
        
        async def check_rate_limit(i):
            return limiter.is_allowed(f"client-{i % 10}", "generate")
        
        tasks = [check_rate_limit(i) for i in range(100)]
        results = asyncio.run(asyncio.gather(*tasks))
        concurrent_time = time.time() - start
        
        # Should be fast even with many concurrent requests
        assert concurrent_time < 1.0  # Less than 1 second
        assert len(results) == 100

