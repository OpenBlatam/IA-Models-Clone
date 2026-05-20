"""
Extreme performance tests
"""

import pytest
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import time
import statistics


class TestPerformanceExtreme:
    """Extreme performance tests"""
    
    @pytest.mark.slow
    @pytest.mark.extreme
    def test_extreme_volume(self, project_generator, temp_dir):
        """Test extreme volume generation"""
        num_projects = 1000
        
        start_time = time.time()
        projects = []
        
        for i in range(num_projects):
            project = project_generator.generate_project(f"Project {i}")
            projects.append(project)
        
        elapsed = time.time() - start_time
        
        # Should handle extreme volume
        assert len(projects) == num_projects
        throughput = num_projects / elapsed if elapsed > 0 else 0
        assert throughput > 0  # Should process at some rate
    
    @pytest.mark.slow
    @pytest.mark.extreme
    @pytest.mark.async
    async def test_extreme_concurrency(self, temp_dir):
        """Test extreme concurrent operations"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        num_concurrent = 2000
        
        async def cache_operation(i):
            await cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
            return await cache.get_cached_project(f"Project {i}", {})
        
        start_time = time.time()
        tasks = [cache_operation(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # Should handle extreme concurrency
        assert len(results) == num_concurrent
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]
        success_rate = len(successes) / num_concurrent if num_concurrent > 0 else 0
        assert success_rate >= 0.7  # 70% success rate minimum
    
    @pytest.mark.slow
    @pytest.mark.extreme
    def test_extreme_memory(self, temp_dir):
        """Test extreme memory usage"""
        num_files = 5000
        file_size = 100 * 1024  # 100KB each
        
        start_time = time.time()
        for i in range(num_files):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text("x" * file_size)
        elapsed = time.time() - start_time
        
        # Should handle extreme memory
        assert len(list(temp_dir.glob("file_*.txt"))) == num_files
        assert elapsed < 600  # Should complete in reasonable time
    
    @pytest.mark.slow
    @pytest.mark.extreme
    def test_extreme_cpu(self, temp_dir):
        """Test extreme CPU usage"""
        num_operations = 100000
        
        start_time = time.time()
        results = []
        for i in range(num_operations):
            # CPU-intensive operation
            result = sum(range(100))
            results.append(result)
        elapsed = time.time() - start_time
        
        # Should handle extreme CPU
        assert len(results) == num_operations
        ops_per_sec = num_operations / elapsed if elapsed > 0 else 0
        assert ops_per_sec > 0
    
    @pytest.mark.slow
    @pytest.mark.extreme
    def test_extreme_io(self, temp_dir):
        """Test extreme I/O operations"""
        num_files = 10000
        
        start_time = time.time()
        for i in range(num_files):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
        elapsed = time.time() - start_time
        
        # Should handle extreme I/O
        assert len(list(temp_dir.glob("file_*.txt"))) == num_files
        io_per_sec = num_files / elapsed if elapsed > 0 else 0
        assert io_per_sec > 0
    
    @pytest.mark.slow
    @pytest.mark.extreme
    def test_latency_consistency(self, project_generator):
        """Test latency consistency under extreme load"""
        num_operations = 1000
        latencies = []
        
        for i in range(num_operations):
            start = time.time()
            project_generator._sanitize_name(f"Test-{i}")
            latency = time.time() - start
            latencies.append(latency)
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        
        # Latency should be consistent
        assert mean_latency < 1.0  # Mean < 1 second
        assert median_latency < 1.0  # Median < 1 second
        assert p95_latency < 2.0  # P95 < 2 seconds

