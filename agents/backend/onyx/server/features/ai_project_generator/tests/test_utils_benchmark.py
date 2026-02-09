"""
Tests for BenchmarkSystem utility
"""

import pytest
from ..utils.benchmark_system import BenchmarkSystem


class TestBenchmarkSystem:
    """Test suite for BenchmarkSystem"""

    def test_init(self):
        """Test BenchmarkSystem initialization"""
        benchmark = BenchmarkSystem()
        assert benchmark.benchmark_results == {}

    def test_benchmark_project_generation(self):
        """Test benchmarking project generation"""
        benchmark = BenchmarkSystem()
        
        project_info = {
            "description": "A test project",
            "features": ["auth", "database"],
            "backend_framework": "fastapi",
            "frontend_framework": "react"
        }
        
        result = benchmark.benchmark_project_generation(
            project_id="test-123",
            generation_time=5.5,
            project_info=project_info
        )
        
        assert result["project_id"] == "test-123"
        assert result["generation_time"] == 5.5
        assert "timestamp" in result
        assert "metrics" in result
        assert "performance_score" in result
        assert result["performance_score"] >= 0
        assert result["performance_score"] <= 100

    def test_benchmark_multiple_projects(self):
        """Test benchmarking multiple projects"""
        benchmark = BenchmarkSystem()
        
        projects = [
            ("project-1", 5.0, {"description": "Simple project"}),
            ("project-2", 10.0, {"description": "Complex project"}),
            ("project-3", 7.5, {"description": "Medium project"}),
        ]
        
        for project_id, time, info in projects:
            benchmark.benchmark_project_generation(project_id, time, info)
        
        assert len(benchmark.benchmark_results) == 3

    def test_calculate_performance_score(self):
        """Test calculating performance score"""
        benchmark = BenchmarkSystem()
        
        project_info = {"description": "Test project"}
        
        # Fast generation should have high score
        fast_score = benchmark._calculate_performance_score(5.0, project_info)
        
        # Slow generation should have lower score
        slow_score = benchmark._calculate_performance_score(50.0, project_info)
        
        assert fast_score > slow_score
        assert 0 <= fast_score <= 100
        assert 0 <= slow_score <= 100

    def test_compare_projects(self):
        """Test comparing multiple projects"""
        benchmark = BenchmarkSystem()
        
        # Create benchmarks
        benchmark.benchmark_project_generation("proj-1", 5.0, {"description": "Fast"})
        benchmark.benchmark_project_generation("proj-2", 15.0, {"description": "Slow"})
        benchmark.benchmark_project_generation("proj-3", 10.0, {"description": "Medium"})
        
        comparison = benchmark.compare_projects(["proj-1", "proj-2", "proj-3"])
        
        assert "projects" in comparison
        assert len(comparison["projects"]) == 3
        assert "average_generation_time" in comparison
        assert "fastest_project" in comparison
        assert "slowest_project" in comparison
        assert comparison["fastest_project"] == "proj-1"
        assert comparison["slowest_project"] == "proj-2"

    def test_compare_projects_not_found(self):
        """Test comparing non-existent projects"""
        benchmark = BenchmarkSystem()
        
        comparison = benchmark.compare_projects(["nonexistent-1", "nonexistent-2"])
        
        assert "error" in comparison

    def test_get_benchmark(self):
        """Test getting specific benchmark"""
        benchmark = BenchmarkSystem()
        
        project_info = {"description": "Test"}
        benchmark.benchmark_project_generation("test-456", 8.0, project_info)
        
        result = benchmark.get_benchmark("test-456")
        
        assert result is not None
        assert result["project_id"] == "test-456"
        assert result["generation_time"] == 8.0

    def test_get_benchmark_not_found(self):
        """Test getting non-existent benchmark"""
        benchmark = BenchmarkSystem()
        
        result = benchmark.get_benchmark("nonexistent")
        
        assert result is None

    def test_list_benchmarks(self):
        """Test listing all benchmarks"""
        benchmark = BenchmarkSystem()
        
        for i in range(5):
            benchmark.benchmark_project_generation(f"proj-{i}", 5.0 + i, {"description": f"Project {i}"})
        
        benchmarks = benchmark.list_benchmarks()
        
        assert len(benchmarks) == 5

