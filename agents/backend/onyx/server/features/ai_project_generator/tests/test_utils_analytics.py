"""
Tests for AnalyticsEngine utility
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta

from ..utils.analytics_engine import AnalyticsEngine


class TestAnalyticsEngine:
    """Test suite for AnalyticsEngine"""

    def test_init(self, temp_dir):
        """Test AnalyticsEngine initialization"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        assert analytics.data_dir == temp_dir / "analytics"
        assert analytics.data_dir.exists()
        assert "projects_by_type" in analytics.metrics

    def test_init_default_dir(self):
        """Test AnalyticsEngine with default directory"""
        analytics = AnalyticsEngine()
        assert analytics.data_dir.exists()

    def test_record_project(self, temp_dir):
        """Test recording a project"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        project_info = {
            "ai_type": "chat",
            "backend_framework": "fastapi",
            "frontend_framework": "react",
            "author": "Test Author"
        }
        
        analytics.record_project(project_info, generation_time=5.5, success=True)
        
        assert analytics.metrics["projects_by_type"]["chat"] == 1
        assert analytics.metrics["projects_by_framework"]["fastapi+react"] == 1
        assert analytics.metrics["projects_by_author"]["Test Author"] == 1
        assert len(analytics.metrics["generation_times"]) == 1
        assert analytics.metrics["generation_times"][0] == 5.5

    def test_record_project_failure(self, temp_dir):
        """Test recording failed project"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        project_info = {"ai_type": "chat", "backend_framework": "fastapi"}
        analytics.record_project(project_info, generation_time=10.0, success=False)
        
        assert len(analytics.metrics["success_rates"]) == 1
        assert analytics.metrics["success_rates"][0] == 0

    def test_get_trends(self, temp_dir):
        """Test getting trends"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        # Record projects for different dates
        for i in range(5):
            project_info = {"ai_type": "chat", "backend_framework": "fastapi"}
            analytics.record_project(project_info, generation_time=5.0, success=True)
        
        trends = analytics.get_trends(days=30)
        
        assert "daily_data" in trends
        assert len(trends["daily_data"]) == 30

    def test_get_top_ai_types(self, temp_dir):
        """Test getting top AI types"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        # Record different AI types
        for ai_type in ["chat", "vision", "chat", "audio", "chat"]:
            project_info = {"ai_type": ai_type, "backend_framework": "fastapi"}
            analytics.record_project(project_info, generation_time=5.0, success=True)
        
        top_types = analytics.get_top_ai_types(limit=3)
        
        assert len(top_types) <= 3
        assert top_types[0]["ai_type"] == "chat"  # Most common

    def test_get_performance_report(self, temp_dir):
        """Test getting performance report"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        # Record projects with different generation times
        for i in range(10):
            project_info = {"ai_type": "chat", "backend_framework": "fastapi"}
            analytics.record_project(project_info, generation_time=5.0 + i, success=True)
        
        report = analytics.get_performance_report()
        
        assert "avg_generation_time" in report
        assert "total_projects" in report
        assert report["total_projects"] == 10

    def test_get_framework_usage(self, temp_dir):
        """Test getting framework usage statistics"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        frameworks = [
            ("fastapi", "react"),
            ("fastapi", "react"),
            ("flask", "vue"),
        ]
        
        for backend, frontend in frameworks:
            project_info = {
                "backend_framework": backend,
                "frontend_framework": frontend,
                "ai_type": "chat"
            }
            analytics.record_project(project_info, generation_time=5.0, success=True)
        
        usage = analytics.get_framework_usage()
        
        assert "fastapi+react" in usage
        assert usage["fastapi+react"] == 2

    def test_metrics_limit(self, temp_dir):
        """Test that metrics lists are limited"""
        analytics = AnalyticsEngine(data_dir=temp_dir / "analytics")
        
        # Add more than limit
        for i in range(10050):
            project_info = {"ai_type": "chat", "backend_framework": "fastapi"}
            analytics.record_project(project_info, generation_time=5.0, success=True)
        
        # Should be limited to 10000
        assert len(analytics.metrics["generation_times"]) == 10000
        assert len(analytics.metrics["success_rates"]) == 10000

