"""
Tests for MetricsCollector utility
"""

import pytest
import time
from unittest.mock import patch

from ..utils.metrics_collector import MetricsCollector


class TestMetricsCollector:
    """Test suite for MetricsCollector"""

    def test_init(self):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector()
        assert collector.metrics["requests_total"] == 0
        assert collector.metrics["projects_generated"] == 0
        assert collector.metrics["projects_failed"] == 0
        assert collector.start_time > 0

    def test_record_request(self):
        """Test recording a request"""
        collector = MetricsCollector()
        
        collector.record_request("/api/v1/generate", 200, 0.5)
        
        assert collector.metrics["requests_total"] == 1
        assert collector.metrics["requests_by_endpoint"]["/api/v1/generate"] == 1
        assert collector.metrics["requests_by_status"][200] == 1
        assert len(collector.metrics["response_times"]) == 1

    def test_record_multiple_requests(self):
        """Test recording multiple requests"""
        collector = MetricsCollector()
        
        collector.record_request("/api/v1/generate", 200, 0.5)
        collector.record_request("/api/v1/status", 200, 0.2)
        collector.record_request("/api/v1/generate", 500, 1.0)
        
        assert collector.metrics["requests_total"] == 3
        assert collector.metrics["requests_by_endpoint"]["/api/v1/generate"] == 2
        assert collector.metrics["requests_by_endpoint"]["/api/v1/status"] == 1
        assert collector.metrics["requests_by_status"][200] == 2
        assert collector.metrics["requests_by_status"][500] == 1

    def test_record_project_generated_success(self):
        """Test recording successful project generation"""
        collector = MetricsCollector()
        
        collector.record_project_generated(success=True)
        collector.record_project_generated(success=True)
        
        assert collector.metrics["projects_generated"] == 2
        assert collector.metrics["projects_failed"] == 0

    def test_record_project_generated_failure(self):
        """Test recording failed project generation"""
        collector = MetricsCollector()
        
        collector.record_project_generated(success=False)
        collector.record_project_generated(success=True)
        collector.record_project_generated(success=False)
        
        assert collector.metrics["projects_generated"] == 1
        assert collector.metrics["projects_failed"] == 2

    def test_record_cache_hit(self):
        """Test recording cache hit"""
        collector = MetricsCollector()
        
        collector.record_cache_hit(hit=True)
        collector.record_cache_hit(hit=True)
        collector.record_cache_hit(hit=False)
        
        assert collector.metrics["cache_hits"] == 2
        assert collector.metrics["cache_misses"] == 1

    def test_record_rate_limit_hit(self):
        """Test recording rate limit hit"""
        collector = MetricsCollector()
        
        collector.record_rate_limit_hit()
        collector.record_rate_limit_hit()
        
        assert collector.metrics["rate_limit_hits"] == 2

    def test_get_metrics(self):
        """Test getting all metrics"""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_request("/api/v1/generate", 200, 0.5)
        collector.record_request("/api/v1/status", 200, 0.3)
        collector.record_project_generated(success=True)
        collector.record_cache_hit(hit=True)
        
        metrics = collector.get_metrics()
        
        assert "requests" in metrics
        assert "projects" in metrics
        assert "cache" in metrics
        assert metrics["requests"]["total"] == 2
        assert metrics["projects"]["generated"] == 1
        assert metrics["cache"]["hits"] == 1

    def test_get_metrics_average_response_time(self):
        """Test average response time calculation"""
        collector = MetricsCollector()
        
        collector.record_request("/api/v1/test", 200, 0.5)
        collector.record_request("/api/v1/test", 200, 0.3)
        collector.record_request("/api/v1/test", 200, 0.7)
        
        metrics = collector.get_metrics()
        
        avg_time = metrics["requests"]["average_response_time_seconds"]
        assert avg_time == pytest.approx(0.5, abs=0.1)

    def test_get_metrics_success_rate(self):
        """Test success rate calculation"""
        collector = MetricsCollector()
        
        collector.record_project_generated(success=True)
        collector.record_project_generated(success=True)
        collector.record_project_generated(success=False)
        
        metrics = collector.get_metrics()
        
        success_rate = metrics["projects"]["success_rate"]
        assert success_rate == pytest.approx(66.67, abs=0.1)

    def test_get_metrics_uptime(self):
        """Test uptime calculation"""
        collector = MetricsCollector()
        time.sleep(0.1)  # Small delay
        
        metrics = collector.get_metrics()
        
        assert "uptime_seconds" in metrics
        assert metrics["uptime_seconds"] > 0

    def test_response_times_deque_limit(self):
        """Test that response times deque has a limit"""
        collector = MetricsCollector()
        
        # Record more than maxlen
        for i in range(1500):
            collector.record_request("/test", 200, 0.1)
        
        # Should be limited to maxlen (1000)
        assert len(collector.metrics["response_times"]) == 1000

