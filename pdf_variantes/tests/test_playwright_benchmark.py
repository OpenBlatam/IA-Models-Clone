"""
Playwright Benchmark Tests
==========================
Performance benchmarking tests with Playwright.
"""

import pytest
from playwright.sync_api import Page, Response
import time
import statistics
from typing import List, Dict, Any


@pytest.fixture
def api_base_url():
    """API base URL."""
    return "http://localhost:8000"


class TestPlaywrightBenchmark:
    """Benchmark tests."""
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_response_time_benchmark(self, page, api_base_url):
        """Benchmark response times."""
        times = []
        iterations = 50
        
        for _ in range(iterations):
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        # Calculate statistics
        stats = {
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "p50": statistics.median(times),
            "p95": sorted(times)[int(len(times) * 0.95)],
            "p99": sorted(times)[int(len(times) * 0.99)]
        }
        
        # Benchmark thresholds
        assert stats["mean"] < 1.0, f"Mean response time too high: {stats['mean']:.3f}s"
        assert stats["p95"] < 2.0, f"P95 response time too high: {stats['p95']:.3f}s"
        assert stats["p99"] < 3.0, f"P99 response time too high: {stats['p99']:.3f}s"
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_throughput_benchmark(self, page, api_base_url):
        """Benchmark throughput."""
        start_time = time.time()
        request_count = 0
        duration = 10  # seconds
        
        while time.time() - start_time < duration:
            try:
                response = page.request.get(f"{api_base_url}/health", timeout=1000)
                if response.status == 200:
                    request_count += 1
            except Exception:
                pass
            time.sleep(0.1)
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed
        
        # Benchmark threshold
        assert throughput > 5, f"Throughput too low: {throughput:.2f} req/s"
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_concurrent_throughput_benchmark(self, browser, api_base_url):
        """Benchmark concurrent throughput."""
        import concurrent.futures
        
        def make_request():
            context = browser.new_context()
            page = context.new_page()
            try:
                return page.request.get(f"{api_base_url}/health")
            finally:
                page.close()
                context.close()
        
        start_time = time.time()
        request_count = 0
        duration = 10  # seconds
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            while time.time() - start_time < duration:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                request_count += sum(1 for r in results if r and r.status == 200)
                time.sleep(0.1)
        
        elapsed = time.time() - start_time
        throughput = request_count / elapsed
        
        # Benchmark threshold
        assert throughput > 20, f"Concurrent throughput too low: {throughput:.2f} req/s"
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_memory_usage_benchmark(self, page, api_base_url):
        """Benchmark memory usage."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for _ in range(100):
            page.request.get(f"{api_base_url}/health")
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Benchmark threshold
        assert memory_increase < 50, f"Memory usage too high: {memory_increase:.1f}MB increase"


class TestPlaywrightBenchmarkComparison:
    """Benchmark comparison tests."""
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_baseline_comparison(self, page, api_base_url):
        """Compare against baseline performance."""
        times = []
        
        for _ in range(20):
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        avg_time = statistics.mean(times)
        baseline = 0.5  # Baseline in seconds
        
        # Should not be more than 2x baseline
        assert avg_time < baseline * 2, f"Performance degraded: {avg_time:.3f}s vs baseline {baseline}s"
    
    @pytest.mark.playwright
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_performance_regression(self, page, api_base_url):
        """Test for performance regressions."""
        times = []
        
        for _ in range(30):
            start = time.time()
            response = page.request.get(f"{api_base_url}/health")
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        # Check for outliers (potential regressions)
        mean = statistics.mean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0
        
        # No more than 3 standard deviations
        outliers = [t for t in times if abs(t - mean) > 3 * stdev and stdev > 0]
        outlier_rate = len(outliers) / len(times)
        
        assert outlier_rate < 0.1, f"Too many performance outliers: {outlier_rate:.2%}"



