"""
Tests for PerformanceAnalyzer utility
"""

import pytest
from datetime import datetime, timedelta

from ..utils.performance_analyzer import PerformanceAnalyzer


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer"""

    def test_init(self):
        """Test PerformanceAnalyzer initialization"""
        analyzer = PerformanceAnalyzer()
        assert len(analyzer.metrics_history) == 0
        assert analyzer.endpoint_performance == {}
        assert analyzer.resource_usage == []

    def test_record_metric(self):
        """Test recording a metric"""
        analyzer = PerformanceAnalyzer()
        
        analyzer.record_metric("generation_time", 5.5, tags={"project_type": "chat"})
        
        assert len(analyzer.metrics_history) == 1
        assert analyzer.metrics_history[0]["name"] == "generation_time"
        assert analyzer.metrics_history[0]["value"] == 5.5

    def test_record_metric_with_tags(self):
        """Test recording metric with tags"""
        analyzer = PerformanceAnalyzer()
        
        analyzer.record_metric("api_latency", 100.0, tags={"endpoint": "/generate", "method": "POST"})
        
        metric = analyzer.metrics_history[0]
        assert metric["tags"]["endpoint"] == "/generate"
        assert metric["tags"]["method"] == "POST"

    def test_analyze_performance(self):
        """Test analyzing performance"""
        analyzer = PerformanceAnalyzer()
        
        # Record multiple metrics
        for i in range(10):
            analyzer.record_metric("generation_time", 5.0 + i)
        
        analysis = analyzer.analyze_performance(time_window_minutes=60)
        
        assert "metrics" in analysis
        assert "generation_time" in analysis["metrics"]
        assert analysis["metrics"]["generation_time"]["count"] == 10
        assert "avg" in analysis["metrics"]["generation_time"]
        assert "min" in analysis["metrics"]["generation_time"]
        assert "max" in analysis["metrics"]["generation_time"]

    def test_analyze_performance_no_metrics(self):
        """Test analyzing with no metrics"""
        analyzer = PerformanceAnalyzer()
        
        analysis = analyzer.analyze_performance(time_window_minutes=60)
        
        assert "error" in analysis

    def test_percentile_calculation(self):
        """Test percentile calculation"""
        analyzer = PerformanceAnalyzer()
        
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p95 = analyzer._percentile(data, 95)
        p99 = analyzer._percentile(data, 99)
        
        assert p95 >= 9
        assert p99 >= 9

    def test_get_performance_report(self):
        """Test getting performance report"""
        analyzer = PerformanceAnalyzer()
        
        # Record metrics
        for i in range(20):
            analyzer.record_metric("generation_time", 5.0 + i * 0.5)
        
        report = analyzer.get_performance_report()
        
        assert "summary" in report
        assert "metrics" in report

