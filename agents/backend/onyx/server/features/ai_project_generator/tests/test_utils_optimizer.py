"""
Tests for AutoOptimizer utility
"""

import pytest
from ..utils.auto_optimizer import AutoOptimizer


class TestAutoOptimizer:
    """Test suite for AutoOptimizer"""

    def test_init(self):
        """Test AutoOptimizer initialization"""
        optimizer = AutoOptimizer()
        assert optimizer.optimization_rules == []
        assert optimizer.optimization_history == []

    def test_analyze_project_fast(self):
        """Test analyzing fast project"""
        optimizer = AutoOptimizer()
        
        project_info = {
            "description": "A simple project",
            "features": ["auth"],
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        
        result = optimizer.analyze_project(project_info, generation_time=5.0)
        
        assert "analysis_date" in result
        assert "generation_time" in result
        assert result["generation_time"] == 5.0
        assert "suggestions" in result
        assert "optimization_score" in result
        assert 0 <= result["optimization_score"] <= 100

    def test_analyze_project_slow(self):
        """Test analyzing slow project"""
        optimizer = AutoOptimizer()
        
        project_info = {
            "description": "A complex project",
            "features": [],
            "backend_framework": "fastapi"
        }
        
        result = optimizer.analyze_project(project_info, generation_time=150.0)
        
        # Should suggest optimization for slow generation
        performance_suggestions = [s for s in result["suggestions"] if s.get("type") == "performance"]
        assert len(performance_suggestions) > 0

    def test_analyze_project_many_features(self):
        """Test analyzing project with many features"""
        optimizer = AutoOptimizer()
        
        project_info = {
            "description": "A project with many features",
            "features": [f"feature_{i}" for i in range(15)],
            "backend_framework": "fastapi"
        }
        
        result = optimizer.analyze_project(project_info, generation_time=10.0)
        
        # Should suggest modularization
        complexity_suggestions = [s for s in result["suggestions"] if s.get("type") == "complexity"]
        assert len(complexity_suggestions) > 0

    def test_analyze_project_django_simple(self):
        """Test analyzing simple project with Django"""
        optimizer = AutoOptimizer()
        
        project_info = {
            "description": "A simple project",
            "features": ["auth"],
            "backend_framework": "django",
            "frontend_framework": "react"
        }
        
        result = optimizer.analyze_project(project_info, generation_time=10.0)
        
        # Should suggest lighter framework
        framework_suggestions = [s for s in result["suggestions"] if s.get("type") == "framework"]
        assert len(framework_suggestions) > 0

    def test_calculate_score(self):
        """Test calculating optimization score"""
        optimizer = AutoOptimizer()
        
        # Fast project with few features should have high score
        high_score = optimizer._calculate_score(5.0, 2)
        
        # Slow project with many features should have lower score
        low_score = optimizer._calculate_score(100.0, 20)
        
        assert high_score > low_score
        assert 0 <= high_score <= 100
        assert 0 <= low_score <= 100

    def test_optimize_project_config(self):
        """Test optimizing project configuration"""
        optimizer = AutoOptimizer()
        
        project_info = {
            "description": "A test project",
            "features": ["auth", "database"],
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        
        optimized = optimizer.optimize_project_config(project_info)
        
        assert "original" in optimized
        assert "optimized" in optimized
        assert optimized["original"] == project_info

    def test_get_optimization_history(self):
        """Test getting optimization history"""
        optimizer = AutoOptimizer()
        
        project_info = {"description": "Test"}
        optimizer.analyze_project(project_info, generation_time=10.0)
        
        history = optimizer.get_optimization_history()
        
        assert len(history) > 0

