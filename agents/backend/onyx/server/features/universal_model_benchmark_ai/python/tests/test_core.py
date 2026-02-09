"""
Core Module Tests - Comprehensive test suite.

Tests:
- Configuration
- Model loading
- Benchmark execution
- Results management
- Analytics
- Monitoring
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from core.config import ModelConfig, BenchmarkConfig, load_config, save_config
from core.results import ResultsManager, BenchmarkResult, ResultStatus
from core.experiments import ExperimentManager, ExperimentConfig, ExperimentStatus
from core.model_registry import ModelRegistry, ModelMetadata, ModelStatus
from core.analytics import AnalyticsEngine
from core.monitoring import HealthMonitor, AlertLevel
from core.cost_tracking import CostTracker, ResourceType
from core.auth import AuthManager, UserRole


class TestResultsManager:
    """Test ResultsManager."""
    
    def setup_method(self):
        """Setup test."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ResultsManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Teardown test."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_get_result(self):
        """Test saving and getting results."""
        result = BenchmarkResult(
            benchmark_name="mmlu",
            model_name="test-model",
            accuracy=0.85,
            latency_p50=0.1,
            latency_p95=0.2,
            latency_p99=0.3,
            throughput=100.0,
            memory_usage={"gpu": 8.0},
            total_samples=100,
            correct_samples=85,
        )
        
        self.manager.save_result(result)
        results = self.manager.get_results(model_name="test-model")
        
        assert len(results) == 1
        assert results[0].model_name == "test-model"
        assert results[0].accuracy == 0.85
    
    def test_get_comparison(self):
        """Test getting comparison."""
        # Add multiple results
        for i, model in enumerate(["model1", "model2", "model3"]):
            result = BenchmarkResult(
                benchmark_name="mmlu",
                model_name=model,
                accuracy=0.7 + i * 0.1,
                latency_p50=0.1,
                latency_p95=0.2,
                latency_p99=0.3,
                throughput=100.0,
                memory_usage={},
                total_samples=100,
                correct_samples=70 + i * 10,
            )
            self.manager.save_result(result)
        
        comparison = self.manager.get_comparison("mmlu")
        assert comparison.benchmark_name == "mmlu"
        assert len(comparison.model_results) == 3


class TestExperimentManager:
    """Test ExperimentManager."""
    
    def setup_method(self):
        """Setup test."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ExperimentManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Teardown test."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_experiment(self):
        """Test creating experiment."""
        config = ExperimentConfig(
            name="test-exp",
            model_name="test-model",
            benchmark_name="mmlu",
        )
        
        exp = self.manager.create_experiment(config)
        
        assert exp.config.name == "test-exp"
        assert exp.status == ExperimentStatus.DRAFT
        assert exp.id is not None
    
    def test_start_and_complete_experiment(self):
        """Test starting and completing experiment."""
        config = ExperimentConfig(
            name="test-exp",
            model_name="test-model",
            benchmark_name="mmlu",
        )
        
        exp = self.manager.create_experiment(config)
        exp = self.manager.start_experiment(exp.id)
        
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None
        
        results = {"accuracy": 0.85}
        exp = self.manager.complete_experiment(exp.id, results)
        
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.results == results


class TestModelRegistry:
    """Test ModelRegistry."""
    
    def setup_method(self):
        """Setup test."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Teardown test."""
        shutil.rmtree(self.temp_dir)
    
    def test_register_model(self):
        """Test registering model."""
        metadata = ModelMetadata(
            name="test-model",
            version="1.0.0",
            architecture="llama",
            parameters=7_000_000_000,
        )
        
        version = self.registry.register_model(metadata, "/path/to/model")
        
        assert version.model_name == "test-model"
        assert version.version == "1.0.0"
        assert version.status == ModelStatus.DRAFT
    
    def test_get_best_models(self):
        """Test getting best models."""
        # Register multiple models with results
        for i, model in enumerate(["model1", "model2", "model3"]):
            metadata = ModelMetadata(
                name=model,
                version="1.0.0",
            )
            version = self.registry.register_model(metadata, f"/path/{model}")
            self.registry.add_benchmark_results(
                model, "1.0.0", "mmlu", {"accuracy": 0.7 + i * 0.1}
            )
        
        best = self.registry.get_best_models("mmlu", top_k=2)
        
        assert len(best) == 2
        assert best[0].model_name == "model3"  # Highest accuracy


class TestAnalyticsEngine:
    """Test AnalyticsEngine."""
    
    def test_analyze_trends(self):
        """Test trend analysis."""
        engine = AnalyticsEngine()
        
        # Create mock results with trend
        results = []
        for i in range(10):
            result = type('Result', (), {
                'accuracy': 0.7 + i * 0.02,
                'timestamp': datetime.now().isoformat(),
            })()
            results.append(result)
        
        trend = engine.analyze_trends(results, metric="accuracy")
        
        assert trend.trend == "increasing"
        assert trend.change_percentage > 0


class TestCostTracker:
    """Test CostTracker."""
    
    def test_record_usage(self):
        """Test recording usage."""
        tracker = CostTracker()
        
        record = tracker.record_usage(
            "test-model",
            "mmlu",
            ResourceType.GPU,
            amount=1,
            duration_seconds=3600,
        )
        
        assert record.total_cost > 0
        assert len(tracker.records) == 1
    
    def test_budget_management(self):
        """Test budget management."""
        tracker = CostTracker()
        tracker.set_budget(100.0)
        
        # Record usage
        tracker.record_usage(
            "test-model",
            "mmlu",
            ResourceType.GPU,
            amount=1,
            duration_seconds=3600,
        )
        
        status = tracker.get_budget_status()
        
        assert status["budget"] == 100.0
        assert status["spent"] > 0
        assert "remaining" in status


class TestAuthManager:
    """Test AuthManager."""
    
    def test_create_user(self):
        """Test creating user."""
        manager = AuthManager()
        
        user = manager.create_user(
            username="testuser",
            email="test@example.com",
            role=UserRole.USER,
        )
        
        assert user.username == "testuser"
        assert user.role == UserRole.USER
    
    def test_generate_token(self):
        """Test token generation."""
        manager = AuthManager()
        user = manager.create_user("testuser", "test@example.com")
        
        token = manager.generate_token(user)
        
        assert token.access_token is not None
        assert token.expires_in == 3600
        
        # Verify token
        payload = manager.verify_token(token.access_token)
        assert payload is not None
        assert payload["username"] == "testuser"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])












