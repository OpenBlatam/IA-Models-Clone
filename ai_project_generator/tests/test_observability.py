"""
Observability and monitoring tests
"""

import pytest
from typing import Dict, Any, List
import time
from datetime import datetime


class TestObservability:
    """Tests for observability features"""
    
    def test_metrics_collection(self, temp_dir):
        """Test that metrics are collected correctly"""
        from ..utils.metrics_collector import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record metrics
        collector.record_metric("generation_time", 5.5)
        collector.record_metric("generation_time", 6.0)
        collector.record_metric("generation_time", 4.5)
        
        # Get statistics
        stats = collector.get_statistics("generation_time")
        
        assert stats is not None
        assert "count" in stats
        assert stats["count"] == 3
        assert "avg" in stats
    
    def test_logging_integration(self, temp_dir):
        """Test logging integration"""
        import logging
        
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        # Should be able to log
        logger.info("Test log message")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Logging should not raise exceptions
        assert True
    
    def test_trace_information(self, project_generator):
        """Test that trace information is available"""
        description = "A test project"
        
        # Generate project
        project = project_generator.generate_project(description)
        
        # Should have traceable information
        assert "project_id" in project
        assert "project_path" in project
        
        # Project ID should be traceable
        project_id = project["project_id"]
        assert len(project_id) > 0
    
    def test_performance_tracking(self, temp_dir):
        """Test performance tracking"""
        from .test_utils_helpers import PerformanceTestHelpers
        
        def operation():
            time.sleep(0.01)
            return "result"
        
        # Track performance
        with PerformanceTestHelpers.measure_time() as elapsed:
            result = operation()
        
        # Should have tracked time
        assert elapsed is not None
        assert elapsed > 0
        assert result == "result"

