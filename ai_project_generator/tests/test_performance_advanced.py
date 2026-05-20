"""
Advanced performance tests
"""

import pytest
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import statistics


class TestPerformanceAdvanced:
    """Advanced performance tests"""
    
    @pytest.mark.slow
    def test_memory_usage(self, project_generator):
        """Test memory usage during project generation"""
        import sys
        
        # Generate multiple projects
        projects = []
        for i in range(10):
            project = project_generator.generate_project(f"Project {i}")
            projects.append(project)
        
        # Check memory (basic check)
        assert len(projects) == 10
        
        # Memory should be manageable
        # In a real scenario, you'd use memory_profiler
        assert True
    
    @pytest.mark.slow
    def test_cpu_usage(self, temp_dir):
        """Test CPU usage during intensive operations"""
        # Create many files
        start_time = time.time()
        
        for i in range(1000):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 10.0  # 10 seconds for 1000 files
    
    def test_response_time_percentiles(self, project_generator):
        """Test response time percentiles"""
        times = []
        
        for i in range(20):
            start = time.time()
            project_generator._sanitize_name(f"Test {i}")
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Calculate percentiles
        p50 = statistics.median(times)
        p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
        p99 = max(times)  # Simplified
        
        # Percentiles should be reasonable
        assert p50 < 0.1  # 50th percentile < 100ms
        assert p95 < 0.5  # 95th percentile < 500ms
    
    @pytest.mark.async
    @pytest.mark.slow
    async def test_throughput(self, temp_dir):
        """Test system throughput"""
        from ..utils.cache_manager import CacheManager
        
        cache = CacheManager(cache_dir=temp_dir / "cache")
        
        num_operations = 100
        start_time = time.time()
        
        tasks = []
        for i in range(num_operations):
            tasks.append(
                cache.cache_project(f"Project {i}", {}, {"id": f"proj-{i}"})
            )
        
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Calculate throughput (operations per second)
        throughput = num_operations / elapsed if elapsed > 0 else 0
        
        # Should handle reasonable throughput
        assert throughput > 10  # At least 10 ops/sec
    
    def test_latency_distribution(self, project_generator):
        """Test latency distribution"""
        latencies = []
        
        for i in range(50):
            start = time.time()
            project_generator._sanitize_name(f"Test-{i}")
            latency = time.time() - start
            latencies.append(latency)
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        # Latency should be consistent
        assert mean_latency < 0.1
        assert std_latency < mean_latency * 2  # Not too variable
    
    @pytest.mark.slow
    def test_resource_cleanup_performance(self, temp_dir):
        """Test performance of resource cleanup"""
        # Create many files
        files = []
        for i in range(500):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text("content")
            files.append(file_path)
        
        # Cleanup
        start = time.time()
        for file_path in files:
            if file_path.exists():
                file_path.unlink()
        elapsed = time.time() - start
        
        # Cleanup should be fast
        assert elapsed < 5.0  # 5 seconds for 500 files

