from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
from performance_metrics import (
from typing import Any, List, Dict, Optional
import logging
"""
Test Suite for API Performance Metrics System

This test suite validates:
- Performance metrics collection and storage
- Response time, latency, and throughput tracking
- System resource monitoring
- Performance alerts and thresholds
- Analytics and insights generation
- Integration with FastAPI
- Performance optimization features
"""


    APIPerformanceMetrics, PerformanceMonitor, PerformanceThreshold,
    PerformanceMetric, MetricSummary, MetricType, get_performance_metrics,
    set_performance_metrics, performance_tracking, track_performance,
    track_cache_performance, track_database_performance, track_external_api_performance
)


class TestAPIPerformanceMetrics:
    """Test cases for APIPerformanceMetrics class."""
    
    @pytest.fixture
    def metrics(self) -> Any:
        """Create a fresh metrics instance for each test."""
        return APIPerformanceMetrics(
            max_metrics=1000,
            window_size=60,
            enable_alerts=True,
            enable_persistence=False
        )
    
    @pytest.fixture
    def sample_metrics(self, metrics) -> Any:
        """Create sample metrics data."""
        # Record some sample requests
        for i in range(10):
            metrics.record_request(
                endpoint=f"/api/test/{i}",
                method="GET",
                status_code=200,
                response_time=100.0 + (i * 10),  # Varying response times
                request_id=str(uuid.uuid4()),
                user_id=f"user_{i}",
                metadata={"test": True, "iteration": i}
            )
        
        # Record some errors
        for i in range(3):
            metrics.record_request(
                endpoint="/api/test/error",
                method="POST",
                status_code=500,
                response_time=200.0,
                request_id=str(uuid.uuid4())
            )
        
        # Record cache hits/misses
        for _ in range(8):
            metrics.record_cache_hit()
        for _ in range(4):
            metrics.record_cache_miss()
        
        # Record database queries
        for i in range(5):
            metrics.record_database_query(50.0 + (i * 5))
        
        # Record external API calls
        for i in range(3):
            metrics.record_external_api_call(100.0 + (i * 10))
        
        return metrics
    
    def test_initialization(self, metrics) -> Any:
        """Test metrics system initialization."""
        assert metrics.max_metrics == 1000
        assert metrics.window_size == 60
        assert metrics.enable_alerts == True
        assert metrics.enable_persistence == False
        assert metrics.request_count == 0
        assert metrics.concurrent_requests == 0
        assert len(metrics.thresholds) > 0  # Default thresholds
    
    async def test_record_request(self, metrics) -> Any:
        """Test recording a single request."""
        request_id = str(uuid.uuid4())
        user_id = "test_user"
        
        metrics.record_request(
            endpoint="/api/test",
            method="GET",
            status_code=200,
            response_time=150.0,
            request_id=request_id,
            user_id=user_id,
            metadata={"test": True}
        )
        
        assert metrics.request_count == 1
        assert len(metrics.metrics["response_time"]) == 1
        assert len(metrics.endpoint_metrics["/api/test"]) == 1
        assert len(metrics.method_metrics["GET"]) == 1
        
        # Check metric data
        metric = metrics.metrics["response_time"][0]
        assert metric.endpoint == "/api/test"
        assert metric.method == "GET"
        assert metric.status_code == 200
        assert metric.response_time == 150.0
        assert metric.request_id == request_id
        assert metric.user_id == user_id
        assert metric.metadata["test"] == True
    
    async def test_record_request_without_optional_fields(self, metrics) -> Any:
        """Test recording a request without optional fields."""
        metrics.record_request(
            endpoint="/api/test",
            method="POST",
            status_code=201,
            response_time=200.0,
            request_id=str(uuid.uuid4())
        )
        
        assert metrics.request_count == 1
        assert len(metrics.metrics["response_time"]) == 1
        
        metric = metrics.metrics["response_time"][0]
        assert metric.user_id is None
        assert metric.metadata == {}
    
    def test_success_and_error_counting(self, metrics) -> Any:
        """Test success and error counting."""
        # Record successful requests
        for _ in range(5):
            metrics.record_request(
                endpoint="/api/test",
                method="GET",
                status_code=200,
                response_time=100.0,
                request_id=str(uuid.uuid4())
            )
        
        # Record error requests
        for _ in range(2):
            metrics.record_request(
                endpoint="/api/test",
                method="POST",
                status_code=500,
                response_time=200.0,
                request_id=str(uuid.uuid4())
            )
        
        assert metrics.success_counts["/api/test"] == 5
        assert metrics.error_counts["/api/test"] == 2
    
    async def test_concurrent_requests_tracking(self, metrics) -> Any:
        """Test concurrent requests tracking."""
        assert metrics.concurrent_requests == 0
        assert metrics.max_concurrent_requests == 0
        
        # Simulate concurrent requests
        for _ in range(3):
            metrics.record_request(
                endpoint="/api/test",
                method="GET",
                status_code=200,
                response_time=100.0,
                request_id=str(uuid.uuid4())
            )
        
        assert metrics.concurrent_requests == 3
        assert metrics.max_concurrent_requests == 3
        
        # Simulate request completion
        for _ in range(2):
            metrics.record_request_end()
        
        assert metrics.concurrent_requests == 1
        assert metrics.max_concurrent_requests == 3  # Should remain at max
    
    def test_cache_performance_tracking(self, metrics) -> Any:
        """Test cache performance tracking."""
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        
        # Record cache hits and misses
        for _ in range(6):
            metrics.record_cache_hit()
        
        for _ in range(4):
            metrics.record_cache_miss()
        
        assert metrics.cache_hits == 6
        assert metrics.cache_misses == 4
        
        hit_rate = metrics.get_cache_hit_rate()
        assert hit_rate == 0.6  # 6 hits / (6 hits + 4 misses)
    
    def test_database_performance_tracking(self, metrics) -> Any:
        """Test database performance tracking."""
        assert metrics.db_query_count == 0
        assert metrics.db_query_time == 0.0
        
        # Record database queries
        for i in range(3):
            query_time = 50.0 + (i * 10)
            metrics.record_database_query(query_time)
        
        assert metrics.db_query_count == 3
        assert metrics.db_query_time == 180.0  # 50 + 60 + 70
        
        stats = metrics.get_database_stats()
        assert stats["query_count"] == 3
        assert stats["total_time"] == 180.0
        assert stats["avg_query_time"] == 60.0
    
    async def test_external_api_performance_tracking(self, metrics) -> Any:
        """Test external API performance tracking."""
        assert metrics.external_api_calls == 0
        assert metrics.external_api_time == 0.0
        
        # Record external API calls
        for i in range(2):
            call_time = 100.0 + (i * 20)
            metrics.record_external_api_call(call_time)
        
        assert metrics.external_api_calls == 2
        assert metrics.external_api_time == 220.0  # 100 + 120
        
        stats = metrics.get_external_api_stats()
        assert stats["call_count"] == 2
        assert stats["total_time"] == 220.0
        assert stats["avg_call_time"] == 110.0
    
    def test_response_time_statistics(self, sample_metrics) -> Any:
        """Test response time statistics calculation."""
        stats = sample_metrics.get_response_time_stats()
        
        assert stats.count == 13  # 10 success + 3 error requests
        assert stats.mean > 0
        assert stats.median > 0
        assert stats.p95 > stats.median
        assert stats.p99 > stats.p95
        assert stats.min > 0
        assert stats.max > stats.min
        assert stats.sum > 0
        assert stats.std_dev >= 0
    
    def test_response_time_statistics_by_endpoint(self, sample_metrics) -> Any:
        """Test response time statistics for specific endpoints."""
        # Test overall stats
        overall_stats = sample_metrics.get_response_time_stats("*")
        assert overall_stats.count == 13
        
        # Test specific endpoint stats
        test_stats = sample_metrics.get_response_time_stats("/api/test/0")
        assert test_stats.count == 1
        
        # Test non-existent endpoint
        empty_stats = sample_metrics.get_response_time_stats("/api/nonexistent")
        assert empty_stats.count == 0
        assert empty_stats.mean == 0.0
    
    def test_error_rate_calculation(self, sample_metrics) -> Any:
        """Test error rate calculation."""
        # Overall error rate
        overall_error_rate = sample_metrics.get_error_rate()
        assert overall_error_rate == 3 / 13  # 3 errors / 13 total requests
        
        # Error rate for specific endpoint
        error_endpoint_rate = sample_metrics.get_error_rate("/api/test/error")
        assert error_endpoint_rate == 1.0  # All requests to error endpoint are errors
        
        # Error rate for success endpoint
        success_endpoint_rate = sample_metrics.get_error_rate("/api/test/0")
        assert success_endpoint_rate == 0.0  # No errors for success endpoint
    
    def test_throughput_calculation(self, sample_metrics) -> Any:
        """Test throughput calculation."""
        # Simulate time passing
        sample_metrics.start_time = time.time() - 10  # 10 seconds ago
        sample_metrics.request_count = 50
        
        throughput = sample_metrics.get_throughput()
        assert throughput > 0
        assert throughput == 50 / 10  # 5 requests per second
    
    def test_system_metrics_collection(self, metrics) -> Any:
        """Test system metrics collection."""
        # Mock psutil to return known values
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 75.0
            mock_cpu.return_value = 60.0
            
            # Trigger system metrics collection
            asyncio.run(metrics._collect_system_metrics())
            
            # Check that metrics were recorded
            assert len(metrics.memory_usage) > 0
            assert len(metrics.cpu_usage) > 0
    
    def test_memory_usage_calculation(self, sample_metrics) -> Any:
        """Test memory usage calculation."""
        # Add some memory usage data
        sample_metrics.memory_usage.extend([0.5, 0.6, 0.7, 0.8, 0.9])
        
        memory_usage = sample_metrics.get_memory_usage()
        assert memory_usage == 0.7  # Average of [0.5, 0.6, 0.7, 0.8, 0.9]
    
    def test_cpu_usage_calculation(self, sample_metrics) -> Any:
        """Test CPU usage calculation."""
        # Add some CPU usage data
        sample_metrics.cpu_usage.extend([0.3, 0.4, 0.5, 0.6, 0.7])
        
        cpu_usage = sample_metrics.get_cpu_usage()
        assert cpu_usage == 0.5  # Average of [0.3, 0.4, 0.5, 0.6, 0.7]
    
    def test_threshold_management(self, metrics) -> Any:
        """Test performance threshold management."""
        initial_threshold_count = len(metrics.thresholds)
        
        # Add custom threshold
        custom_threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            endpoint="/api/custom",
            threshold_value=2000.0,
            comparison="gt",
            alert_message="Custom endpoint too slow",
            severity="error"
        )
        
        metrics.add_threshold(custom_threshold)
        assert len(metrics.thresholds) == initial_threshold_count + 1
    
    def test_alert_generation(self, metrics) -> Any:
        """Test alert generation."""
        # Add threshold that will trigger
        threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            endpoint="*",
            threshold_value=100.0,  # Low threshold to trigger
            comparison="gt",
            alert_message="Response time too high",
            severity="warning"
        )
        metrics.add_threshold(threshold)
        
        # Record request that exceeds threshold
        metrics.record_request(
            endpoint="/api/test",
            method="GET",
            status_code=200,
            response_time=150.0,  # Exceeds 100ms threshold
            request_id=str(uuid.uuid4())
        )
        
        # Trigger threshold checking
        asyncio.run(metrics._check_thresholds())
        
        # Check that alert was generated
        alerts = metrics.get_alerts()
        assert len(alerts) > 0
        
        alert = alerts[0]
        assert alert.threshold == threshold
        assert alert.current_value > threshold.threshold_value
        assert alert.severity == "warning"
    
    def test_alert_filtering(self, metrics) -> Any:
        """Test alert filtering by severity."""
        # Add different severity alerts
        warning_threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            endpoint="*",
            threshold_value=100.0,
            comparison="gt",
            severity="warning"
        )
        
        error_threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            endpoint="*",
            threshold_value=50.0,
            comparison="gt",
            severity="error"
        )
        
        metrics.add_threshold(warning_threshold)
        metrics.add_threshold(error_threshold)
        
        # Record requests to trigger alerts
        for _ in range(2):
            metrics.record_request(
                endpoint="/api/test",
                method="GET",
                status_code=200,
                response_time=150.0,
                request_id=str(uuid.uuid4())
            )
        
        # Trigger threshold checking
        asyncio.run(metrics._check_thresholds())
        
        # Test filtering
        warning_alerts = metrics.get_alerts("warning")
        error_alerts = metrics.get_alerts("error")
        
        assert len(warning_alerts) > 0
        assert len(error_alerts) > 0
        
        # All warning alerts should have warning severity
        for alert in warning_alerts:
            assert alert.severity == "warning"
        
        # All error alerts should have error severity
        for alert in error_alerts:
            assert alert.severity == "error"
    
    def test_comprehensive_stats(self, sample_metrics) -> Any:
        """Test comprehensive statistics generation."""
        stats = sample_metrics.get_comprehensive_stats()
        
        # Check that all expected keys are present
        expected_keys = [
            "response_time", "throughput", "error_rate", "cache",
            "system", "database", "external_api", "alerts", "uptime"
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Check response time stats structure
        response_time_stats = stats["response_time"]
        assert "overall" in response_time_stats
        assert "by_endpoint" in response_time_stats
        
        # Check system stats
        system_stats = stats["system"]
        assert "memory_usage" in system_stats
        assert "cpu_usage" in system_stats
        assert "concurrent_requests" in system_stats
    
    def test_metrics_cleanup(self, metrics) -> Any:
        """Test old metrics cleanup."""
        # Add metrics with old timestamps
        old_time = time.time() - 120  # 2 minutes ago (outside 60s window)
        
        old_metric = PerformanceMetric(
            name="response_time",
            value=100.0,
            timestamp=old_time,
            endpoint="/api/old",
            method="GET",
            status_code=200,
            request_id=str(uuid.uuid4())
        )
        
        metrics.metrics["response_time"].append(old_metric)
        
        # Add metrics with recent timestamps
        recent_time = time.time() - 30  # 30 seconds ago (within 60s window)
        
        recent_metric = PerformanceMetric(
            name="response_time",
            value=100.0,
            timestamp=recent_time,
            endpoint="/api/recent",
            method="GET",
            status_code=200,
            request_id=str(uuid.uuid4())
        )
        
        metrics.metrics["response_time"].append(recent_metric)
        
        # Trigger cleanup
        asyncio.run(metrics._cleanup_old_metrics())
        
        # Check that old metrics were removed
        remaining_metrics = list(metrics.metrics["response_time"])
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].endpoint == "/api/recent"
    
    async def test_cleanup_task_management(self, metrics) -> Any:
        """Test cleanup task management."""
        # Start background tasks
        metrics._start_background_tasks()
        
        # Check that tasks are running
        assert metrics._cleanup_task is not None
        assert not metrics._cleanup_task.done()
        assert metrics._system_metrics_task is not None
        assert not metrics._system_metrics_task.done()
        
        # Close metrics
        await metrics.close()
        
        # Check that tasks are cancelled
        assert metrics._cleanup_task.cancelled()
        assert metrics._system_metrics_task.cancelled()


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor class."""
    
    @pytest.fixture
    def monitor(self) -> Any:
        """Create a performance monitor instance."""
        return PerformanceMonitor()
    
    @pytest.fixture
    def sample_monitor(self, monitor) -> Any:
        """Create a monitor with sample data."""
        # Add some sample metrics
        for i in range(10):
            monitor.metrics.record_request(
                endpoint=f"/api/test/{i}",
                method="GET",
                status_code=200,
                response_time=100.0 + (i * 20),
                request_id=str(uuid.uuid4())
            )
        
        return monitor
    
    def test_response_time_percentiles(self, sample_monitor) -> Any:
        """Test response time percentile calculation."""
        percentiles = sample_monitor.get_response_time_percentiles()
        
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert "p99.9" in percentiles
        
        # Check that percentiles are in ascending order
        assert percentiles["p50"] <= percentiles["p95"]
        assert percentiles["p95"] <= percentiles["p99"]
        assert percentiles["p99"] <= percentiles["p99.9"]
    
    def test_response_time_percentiles_by_endpoint(self, sample_monitor) -> Any:
        """Test response time percentiles for specific endpoints."""
        # Test specific endpoint
        percentiles = sample_monitor.get_response_time_percentiles("/api/test/0")
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
    
    def test_throughput_trend(self, sample_monitor) -> Any:
        """Test throughput trend calculation."""
        trend = sample_monitor.get_throughput_trend()
        assert isinstance(trend, list)
    
    def test_error_rate_trend(self, sample_monitor) -> Any:
        """Test error rate trend calculation."""
        # Add some errors
        for _ in range(3):
            sample_monitor.metrics.record_request(
                endpoint="/api/test/error",
                method="POST",
                status_code=500,
                response_time=200.0,
                request_id=str(uuid.uuid4())
            )
        
        error_rate = sample_monitor.get_error_rate_trend()
        assert error_rate > 0
        assert error_rate <= 1.0
    
    def test_system_health_score(self, sample_monitor) -> Any:
        """Test system health score calculation."""
        score = sample_monitor.get_system_health_score()
        
        assert 0.0 <= score <= 100.0
        assert isinstance(score, float)
    
    def test_system_health_score_with_poor_performance(self, monitor) -> Any:
        """Test system health score with poor performance indicators."""
        # Add slow response times
        for _ in range(5):
            monitor.metrics.record_request(
                endpoint="/api/slow",
                method="GET",
                status_code=200,
                response_time=2000.0,  # 2 seconds - should reduce score
                request_id=str(uuid.uuid4())
            )
        
        # Add errors
        for _ in range(3):
            monitor.metrics.record_request(
                endpoint="/api/error",
                method="POST",
                status_code=500,
                response_time=100.0,
                request_id=str(uuid.uuid4())
            )
        
        score = monitor.get_system_health_score()
        assert score < 100.0  # Should be reduced due to poor performance
    
    def test_performance_recommendations(self, sample_monitor) -> Any:
        """Test performance recommendations generation."""
        recommendations = sample_monitor.get_performance_recommendations()
        
        assert isinstance(recommendations, list)
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_performance_recommendations_with_issues(self, monitor) -> Any:
        """Test performance recommendations with performance issues."""
        # Add slow response times to trigger recommendations
        for _ in range(5):
            monitor.metrics.record_request(
                endpoint="/api/slow",
                method="GET",
                status_code=200,
                response_time=1500.0,  # 1.5 seconds
                request_id=str(uuid.uuid4())
            )
        
        # Add errors to trigger recommendations
        for _ in range(3):
            monitor.metrics.record_request(
                endpoint="/api/error",
                method="POST",
                status_code=500,
                response_time=100.0,
                request_id=str(uuid.uuid4())
            )
        
        recommendations = monitor.get_performance_recommendations()
        assert len(recommendations) > 0
        
        # Check for specific recommendation types
        recommendation_text = " ".join(recommendations).lower()
        assert "caching" in recommendation_text or "optimize" in recommendation_text or "error" in recommendation_text


class TestPerformanceDecorators:
    """Test cases for performance decorators."""
    
    @pytest.fixture
    def metrics(self) -> Any:
        """Create metrics instance for decorator tests."""
        return APIPerformanceMetrics()
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator(self, metrics) -> Any:
        """Test track_performance decorator."""
        @track_performance("test_function")
        async def test_function():
            
    """test_function function."""
await asyncio.sleep(0.1)
            return "success"
        
        # Set metrics globally
        set_performance_metrics(metrics)
        
        # Call decorated function
        result = await test_function()
        
        assert result == "success"
        
        # Check that metrics were recorded
        assert len(metrics.metrics["response_time"]) == 1
        
        metric = metrics.metrics["response_time"][0]
        assert metric.endpoint == "test_function"
        assert metric.method == "FUNCTION"
        assert metric.status_code == 200
        assert metric.response_time > 0
    
    @pytest.mark.asyncio
    async def test_track_cache_performance_decorator(self, metrics) -> Any:
        """Test track_cache_performance decorator."""
        @track_cache_performance
        async def test_cache_function():
            
    """test_cache_function function."""
return "cached_result"
        
        # Set metrics globally
        set_performance_metrics(metrics)
        
        # Call decorated function
        result = await test_cache_function()
        
        assert result == "cached_result"
        
        # Check that cache metrics were recorded
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 1  # Should record a miss since no cache
    
    @pytest.mark.asyncio
    async def test_track_database_performance_decorator(self, metrics) -> Any:
        """Test track_database_performance decorator."""
        @track_database_performance
        async def test_database_function():
            
    """test_database_function function."""
await asyncio.sleep(0.05)
            return "database_result"
        
        # Set metrics globally
        set_performance_metrics(metrics)
        
        # Call decorated function
        result = await test_database_function()
        
        assert result == "database_result"
        
        # Check that database metrics were recorded
        assert metrics.db_query_count == 1
        assert metrics.db_query_time > 0
    
    @pytest.mark.asyncio
    async async def test_track_external_api_performance_decorator(self, metrics) -> Any:
        """Test track_external_api_performance decorator."""
        @track_external_api_performance
        async def test_external_api_function():
            
    """test_external_api_function function."""
await asyncio.sleep(0.05)
            return "api_result"
        
        # Set metrics globally
        set_performance_metrics(metrics)
        
        # Call decorated function
        result = await test_external_api_function()
        
        assert result == "api_result"
        
        # Check that external API metrics were recorded
        assert metrics.external_api_calls == 1
        assert metrics.external_api_time > 0


class TestGlobalMetrics:
    """Test cases for global metrics management."""
    
    def test_get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Test getting global performance metrics."""
        # Clear any existing metrics
        set_performance_metrics(None)
        
        # Get metrics (should create new instance)
        metrics = get_performance_metrics()
        assert isinstance(metrics, APIPerformanceMetrics)
        
        # Get metrics again (should return same instance)
        metrics2 = get_performance_metrics()
        assert metrics is metrics2
    
    def test_set_performance_metrics(self) -> Any:
        """Test setting global performance metrics."""
        # Create custom metrics
        custom_metrics = APIPerformanceMetrics(max_metrics=500)
        
        # Set global metrics
        set_performance_metrics(custom_metrics)
        
        # Get metrics (should return custom instance)
        retrieved_metrics = get_performance_metrics()
        assert retrieved_metrics is custom_metrics
        assert retrieved_metrics.max_metrics == 500


class TestIntegration:
    """Integration tests for the performance metrics system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self) -> Any:
        """Test complete performance metrics workflow."""
        # Initialize metrics
        metrics = APIPerformanceMetrics(enable_alerts=True)
        monitor = PerformanceMonitor()
        
        try:
            # Simulate API requests
            for i in range(20):
                response_time = 100 + (i * 5)  # Varying response times
                status_code = 200 if i < 18 else 500  # Some errors
                
                metrics.record_request(
                    endpoint=f"/api/test/{i % 5}",
                    method="GET" if i % 2 == 0 else "POST",
                    status_code=status_code,
                    response_time=response_time,
                    request_id=str(uuid.uuid4())
                )
                
                # Simulate cache hits/misses
                if i % 3 == 0:
                    metrics.record_cache_hit()
                else:
                    metrics.record_cache_miss()
                
                # Simulate database queries
                if i % 2 == 0:
                    metrics.record_database_query(50.0 + (i * 2))
                
                # Simulate external API calls
                if i % 4 == 0:
                    metrics.record_external_api_call(100.0 + (i * 5))
            
            # Get comprehensive stats
            stats = metrics.get_comprehensive_stats()
            
            # Verify stats structure
            assert "response_time" in stats
            assert "throughput" in stats
            assert "error_rate" in stats
            assert "cache" in stats
            assert "system" in stats
            assert "database" in stats
            assert "external_api" in stats
            
            # Verify response time stats
            response_time_stats = stats["response_time"]["overall"]
            assert response_time_stats["count"] == 20
            assert response_time_stats["mean"] > 0
            
            # Verify error rate
            error_rate = stats["error_rate"]["overall"]
            assert error_rate == 2 / 20  # 2 errors out of 20 requests
            
            # Verify cache stats
            cache_stats = stats["cache"]
            assert cache_stats["hits"] == 7  # 7 cache hits (every 3rd request)
            assert cache_stats["misses"] == 13  # 13 cache misses
            
            # Verify database stats
            db_stats = stats["database"]
            assert db_stats["query_count"] == 10  # 10 database queries (every 2nd request)
            
            # Verify external API stats
            api_stats = stats["external_api"]
            assert api_stats["call_count"] == 5  # 5 API calls (every 4th request)
            
            # Test health score
            health_score = monitor.get_system_health_score()
            assert 0.0 <= health_score <= 100.0
            
            # Test recommendations
            recommendations = monitor.get_performance_recommendations()
            assert isinstance(recommendations, list)
            
            print(f"✅ Integration test completed successfully!")
            print(f"   - Total requests: {stats['response_time']['overall']['count']}")
            print(f"   - Average response time: {stats['response_time']['overall']['mean']:.2f}ms")
            print(f"   - Error rate: {stats['error_rate']['overall']:.2%}")
            print(f"   - Cache hit rate: {cache_stats['hit_rate']:.2%}")
            print(f"   - Health score: {health_score:.1f}/100")
            print(f"   - Recommendations: {len(recommendations)}")
            
        finally:
            await metrics.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 