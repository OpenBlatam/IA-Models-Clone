"""
Performance and stress tests
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


class TestPerformance:
    """Test suite for performance and stress testing"""

    @pytest.mark.asyncio
    async def test_project_generation_performance(self, project_generator, temp_dir):
        """Test project generation performance"""
        description = "A performance test project"
        
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
            result = await project_generator.generate_project(description=description)
            elapsed = time.time() - start_time
            
            assert "project_id" in result
            # Should complete in reasonable time (with mocks, should be very fast)
            assert elapsed < 5.0  # 5 seconds max with mocks

    @pytest.mark.asyncio
    async def test_cache_performance(self, temp_dir):
        """Test cache performance with many entries"""
        manager = CacheManager(cache_dir=temp_dir / "cache")
        
        # Cache many projects
        start_time = time.time()
        tasks = []
        for i in range(100):
            description = f"Project {i}"
            config = {"framework": "fastapi", "id": i}
            project_info = {"project_id": f"proj-{i}"}
            tasks.append(manager.cache_project(description, config, project_info))
        
        await asyncio.gather(*tasks)
        cache_time = time.time() - start_time
        
        # Should cache 100 projects in reasonable time
        assert cache_time < 10.0
        
        # Test retrieval performance
        start_time = time.time()
        tasks = []
        for i in range(100):
            description = f"Project {i}"
            config = {"framework": "fastapi", "id": i}
            tasks.append(manager.get_cached_project(description, config))
        
        results = await asyncio.gather(*tasks)
        retrieve_time = time.time() - start_time
        
        # Should retrieve 100 projects quickly
        assert retrieve_time < 5.0
        assert all(r is not None for r in results)

    def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        limiter = RateLimiter()
        limiter.limits["default"] = {"requests": 10000, "window": 3600}
        
        start_time = time.time()
        for i in range(1000):
            limiter.is_allowed(f"client_{i % 100}")
        elapsed = time.time() - start_time
        
        # Should handle 1000 requests quickly
        assert elapsed < 1.0

    def test_continuous_generator_queue_performance(self, continuous_generator):
        """Test continuous generator queue performance"""
        start_time = time.time()
        
        # Add many projects to queue
        for i in range(1000):
            continuous_generator.add_project(f"Project {i}", priority=i % 10)
        
        elapsed = time.time() - start_time
        
        # Should add 1000 projects quickly
        assert elapsed < 2.0
        assert len(continuous_generator.queue) == 1000
        
        # Test queue retrieval
        start_time = time.time()
        queue_info = continuous_generator.get_queue()
        elapsed = time.time() - start_time
        
        # Should retrieve queue info quickly
        assert elapsed < 0.1
        assert queue_info["total"] == 1000

    @pytest.mark.asyncio
    async def test_concurrent_project_generation(self, project_generator, temp_dir):
        """Test concurrent project generation performance"""
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
            
            # Generate 50 projects concurrently
            tasks = [
                project_generator.generate_project(
                    description=f"Concurrent project {i}",
                    project_name=f"concurrent_{i}"
                )
                for i in range(50)
            ]
            
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            
            assert len(results) == 50
            # Concurrent execution should be faster than sequential
            assert elapsed < 10.0

    def test_keyword_extraction_performance(self, project_generator):
        """Test keyword extraction performance"""
        descriptions = [
            "A chat AI system with authentication and database",
            "A vision AI system for object detection",
            "An audio AI system for speech recognition",
        ] * 100
        
        start_time = time.time()
        keywords_list = [project_generator._extract_keywords(desc) for desc in descriptions]
        elapsed = time.time() - start_time
        
        assert len(keywords_list) == 300
        # Should process 300 descriptions quickly
        assert elapsed < 2.0
        assert all("ai_type" in k for k in keywords_list)

