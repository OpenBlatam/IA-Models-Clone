"""
Stress and load tests for AI Project Generator
"""

import pytest
import asyncio
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator
from ..utils.cache_manager import CacheManager
from ..utils.rate_limiter import RateLimiter


class TestStress:
    """Test suite for stress and load testing"""

    @pytest.mark.asyncio
    async def test_mass_project_generation(self, project_generator, temp_dir):
        """Test generating many projects under load"""
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi"}
            mock_frontend.return_value = {"framework": "react"}
            
            start_time = time.time()
            
            # Generate 100 projects concurrently
            tasks = [
                project_generator.generate_project(
                    description=f"Stress test project {i} with many features",
                    project_name=f"stress_{i}"
                )
                for i in range(100)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            
            # Should complete successfully
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) == 100
            # Should complete in reasonable time
            assert elapsed < 30.0

    @pytest.mark.asyncio
    async def test_cache_stress(self, temp_dir):
        """Test cache under heavy load"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        start_time = time.time()
        
        # Cache 500 projects
        tasks = []
        for i in range(500):
            description = f"Stress cache project {i}"
            config = {"framework": "fastapi", "id": i}
            project_info = {"project_id": f"stress-{i}"}
            tasks.append(manager.cache_project(description, config, project_info))
        
        await asyncio.gather(*tasks)
        cache_time = time.time() - start_time
        
        # Should cache quickly
        assert cache_time < 10.0
        
        # Retrieve all
        start_time = time.time()
        retrieve_tasks = []
        for i in range(500):
            description = f"Stress cache project {i}"
            config = {"framework": "fastapi", "id": i}
            retrieve_tasks.append(manager.get_cached_project(description, config))
        
        results = await asyncio.gather(*retrieve_tasks)
        retrieve_time = time.time() - start_time
        
        # Should retrieve quickly
        assert retrieve_time < 5.0
        assert all(r is not None for r in results)

    def test_rate_limiter_stress(self):
        """Test rate limiter under heavy load"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 10000, "window": 3600}
        
        start_time = time.time()
        
        # Make 5000 requests
        for i in range(5000):
            limiter.is_allowed(f"client_{i % 100}")
        
        elapsed = time.time() - start_time
        
        # Should handle quickly
        assert elapsed < 2.0

    def test_continuous_generator_queue_stress(self, continuous_generator):
        """Test continuous generator with very large queue"""
        start_time = time.time()
        
        # Add 5000 projects to queue
        for i in range(5000):
            continuous_generator.add_project(f"Stress project {i}", priority=i % 10)
        
        elapsed = time.time() - start_time
        
        # Should add quickly
        assert elapsed < 5.0
        assert len(continuous_generator.queue) == 5000
        
        # Test queue operations
        start_time = time.time()
        queue_info = continuous_generator.get_queue()
        elapsed = time.time() - start_time
        
        # Should retrieve quickly even with large queue
        assert elapsed < 1.0
        assert queue_info["total"] == 5000

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, temp_dir):
        """Test concurrent cache operations"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Concurrent cache and retrieve
        async def cache_and_retrieve(i):
            description = f"Concurrent project {i}"
            config = {"id": i}
            project_info = {"project_id": f"proj-{i}"}
            
            await manager.cache_project(description, config, project_info)
            cached = await manager.get_cached_project(description, config)
            return cached is not None
        
        tasks = [cache_and_retrieve(i) for i in range(200)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(results)

    def test_keyword_extraction_stress(self, project_generator):
        """Test keyword extraction under load"""
        descriptions = [
            f"A comprehensive AI project {i} with machine learning, deep learning, and advanced features"
            for i in range(1000)
        ]
        
        start_time = time.time()
        keywords_list = [project_generator._extract_keywords(desc) for desc in descriptions]
        elapsed = time.time() - start_time
        
        # Should process quickly
        assert elapsed < 3.0
        assert len(keywords_list) == 1000
        assert all("ai_type" in k for k in keywords_list)

    @pytest.mark.asyncio
    async def test_mixed_operations_stress(self, project_generator, temp_dir):
        """Test mixed operations under stress"""
        with patch.object(project_generator.backend_generator, 'generate', new_callable=AsyncMock) as mock_backend, \
             patch.object(project_generator.frontend_generator, 'generate', new_callable=AsyncMock) as mock_frontend, \
             patch.object(project_generator.test_generator, 'generate_backend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.test_generator, 'generate_frontend_tests', new_callable=AsyncMock), \
             patch.object(project_generator.cicd_generator, 'generate_github_actions', new_callable=AsyncMock), \
             patch.object(project_generator.cache_manager, 'get_cached_project', new_callable=AsyncMock, return_value=None), \
             patch.object(project_generator.cache_manager, 'cache_project', new_callable=AsyncMock), \
             patch.object(project_generator.validator, 'validate_project', new_callable=AsyncMock, return_value={"valid": True}):
            
            mock_backend.return_value = {"framework": "fastapi"}
            mock_frontend.return_value = {"framework": "react"}
            
            # Mix of operations
            tasks = []
            for i in range(50):
                tasks.append(project_generator.generate_project(
                    description=f"Mixed stress project {i}",
                    project_name=f"mixed_{i}"
                ))
            
            # Also add to continuous generator
            continuous_gen = ContinuousGenerator(base_output_dir=str(temp_dir / "projects2"))
            for i in range(50):
                continuous_gen.add_project(f"Queue project {i}")
            
            # Execute all
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) == 50

