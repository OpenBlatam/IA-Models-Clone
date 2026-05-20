from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import json
from pathlib import Path
from functional_training import (
from functional_config_loader import (
from functional_evaluation_metrics import (
from functional_fastapi_app import (
        import yaml
        from hypothesis import given, strategies as st
        from hypothesis import given, strategies as st
        import time
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🧪 Functional Approach Tests
============================

Comprehensive tests for the functional, declarative programming approach.
Tests pure functions, immutable data structures, and functional patterns.
"""


# Import functional modules
    TrainingConfig, TrainingMode, ModelType, TrainingState,
    create_default_config, update_config, validate_config,
    create_model, create_optimizer, create_scheduler,
    calculate_metrics, train_epoch, validate_epoch,
    get_device_info, setup_device_optimization
)

    ExperimentConfig, ExperimentMetadata, ConfigValidationResult,
    load_config_from_yaml, save_config_to_yaml,
    create_experiment_config, validate_config_parameters,
    merge_configs, create_config_from_template
)

    TaskType, MetricType, MetricConfig, EvaluationResult,
    calculate_classification_metrics, calculate_regression_metrics,
    evaluate_model, compare_models, rank_models,
    quick_evaluate_classification, quick_evaluate_regression
)

    TrainingRequest, InferenceRequest, ExperimentStatus, TrainingResponse,
    create_app_config, create_cors_config, setup_logging,
    create_experiment_id, create_experiment_state, update_experiment_state,
    create_training_response, create_experiment_status_response,
    calculate_estimated_duration, handle_training_error
)

# ============================================================================
# Test Data
# ============================================================================

@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name="distilbert-base-uncased",
        dataset_path="data/test.csv",
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5
    )

@pytest.fixture
def sample_data():
    """Sample training data."""
    np.random.seed(42)
    return {
        'y_true': np.random.randint(0, 3, 100),
        'y_pred': np.random.randint(0, 3, 100),
        'y_prob': np.random.rand(100, 3),
        'y_true_reg': np.random.randn(100),
        'y_pred_reg': np.random.randn(100) + 0.1
    }

@pytest.fixture
def sample_training_request():
    """Sample training request."""
    return TrainingRequest(
        model_name="distilbert-base-uncased",
        dataset_path="data/test.csv",
        num_epochs=5,
        batch_size=16,
        learning_rate=2e-5
    )

# ============================================================================
# Tests for Functional Training
# ============================================================================

class TestFunctionalTraining:
    """Test functional training functions."""
    
    def test_create_default_config(self) -> Any:
        """Test creating default configuration."""
        config = create_default_config("test-model", "data/test.csv")
        
        assert config.model_name == "test-model"
        assert config.dataset_path == "data/test.csv"
        assert config.model_type == ModelType.TRANSFORMER
        assert config.training_mode == TrainingMode.FINE_TUNE
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
    
    def test_update_config_immutable(self, sample_config) -> Any:
        """Test that config updates are immutable."""
        original_batch_size = sample_config.batch_size
        updated_config = update_config(sample_config, batch_size=32)
        
        # Original config should be unchanged
        assert sample_config.batch_size == original_batch_size
        # New config should have updated value
        assert updated_config.batch_size == 32
        # Should be different objects
        assert sample_config is not updated_config
    
    def test_validate_config_valid(self, sample_config) -> bool:
        """Test validation of valid configuration."""
        is_valid, errors = validate_config(sample_config)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_config_invalid(self) -> bool:
        """Test validation of invalid configuration."""
        invalid_config = TrainingConfig(
            model_type=ModelType.TRANSFORMER,
            training_mode=TrainingMode.FINE_TUNE,
            model_name="test",
            dataset_path="data/test.csv",
            batch_size=-1,  # Invalid
            learning_rate=0,  # Invalid
            num_epochs=0  # Invalid
        )
        
        is_valid, errors = validate_config(invalid_config)
        assert not is_valid
        assert len(errors) > 0
        assert "batch_size must be positive" in errors
    
    def test_calculate_metrics_pure(self, sample_data) -> Any:
        """Test that calculate_metrics is pure."""
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        
        # First call
        metrics1 = calculate_metrics(y_true, y_pred)
        # Second call with same inputs
        metrics2 = calculate_metrics(y_true, y_pred)
        
        # Should return same result
        assert metrics1 == metrics2
    
    def test_get_device_info_pure(self) -> Optional[Dict[str, Any]]:
        """Test that get_device_info is pure."""
        info1 = get_device_info()
        info2 = get_device_info()
        
        # Should return same result (device info doesn't change during test)
        assert info1.device == info2.device
        assert info1.gpu_available == info2.gpu_available
        assert info1.num_gpus == info2.num_gpus

# ============================================================================
# Tests for Functional Config Loader
# ============================================================================

class TestFunctionalConfigLoader:
    """Test functional configuration loader."""
    
    def test_load_and_save_yaml(self, sample_config, tmp_path) -> Any:
        """Test loading and saving YAML configuration."""
        config_path = tmp_path / "test_config.yaml"
        
        # Save config
        save_config_to_yaml(sample_config, str(config_path))
        assert config_path.exists()
        
        # Load config
        loaded_config = load_config_from_yaml(str(config_path))
        
        # Should be equivalent
        assert loaded_config.model_name == sample_config.model_name
        assert loaded_config.batch_size == sample_config.batch_size
        assert loaded_config.learning_rate == sample_config.learning_rate
    
    def test_create_experiment_config(self, sample_config) -> Any:
        """Test creating experiment configuration."""
        exp_config = create_experiment_config("test_exp", "Test experiment", sample_config)
        
        assert exp_config.experiment_id == "test_exp"
        assert exp_config.description == "Test experiment"
        assert exp_config.config == sample_config
        assert exp_config.metadata.experiment_id == "test_exp"
        assert exp_config.metadata.timestamp != ""
    
    def test_validate_config_parameters(self, sample_config) -> bool:
        """Test configuration parameter validation."""
        result = validate_config_parameters(sample_config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_merge_configs(self, sample_config) -> Any:
        """Test merging configurations."""
        overrides = {"batch_size": 32, "learning_rate": 1e-5}
        merged_config = merge_configs(sample_config, overrides)
        
        assert merged_config.batch_size == 32
        assert merged_config.learning_rate == 1e-5
        assert merged_config.model_name == sample_config.model_name  # Unchanged
    
    def test_create_config_from_template(self) -> Any:
        """Test creating config from template."""
        config = create_config_from_template(
            "quick", "test-model", "data/test.csv", num_epochs=10
        )
        
        assert config.model_name == "test-model"
        assert config.dataset_path == "data/test.csv"
        assert config.num_epochs == 10
        assert config.model_type == ModelType.TRANSFORMER

# ============================================================================
# Tests for Functional Evaluation Metrics
# ============================================================================

class TestFunctionalEvaluationMetrics:
    """Test functional evaluation metrics."""
    
    def test_calculate_classification_metrics_pure(self, sample_data) -> Any:
        """Test that classification metrics calculation is pure."""
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        y_prob = sample_data['y_prob']
        
        # First call
        metrics1 = calculate_classification_metrics(y_true, y_pred, y_prob)
        # Second call with same inputs
        metrics2 = calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # Should return same result
        assert metrics1 == metrics2
    
    def test_calculate_regression_metrics_pure(self, sample_data) -> Any:
        """Test that regression metrics calculation is pure."""
        y_true = sample_data['y_true_reg']
        y_pred = sample_data['y_pred_reg']
        
        # First call
        metrics1 = calculate_regression_metrics(y_true, y_pred)
        # Second call with same inputs
        metrics2 = calculate_regression_metrics(y_true, y_pred)
        
        # Should return same result
        assert metrics1 == metrics2
    
    def test_evaluate_model_classification(self, sample_data) -> Any:
        """Test model evaluation for classification."""
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        y_prob = sample_data['y_prob']
        
        result = evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION)
        
        assert result.task_type == TaskType.CLASSIFICATION
        assert 'accuracy' in result.metrics
        assert 'f1' in result.metrics
        assert 'precision' in result.metrics
        assert 'recall' in result.metrics
        assert result.confusion_matrix is not None
        assert result.inference_time_ms >= 0
    
    def test_evaluate_model_regression(self, sample_data) -> Any:
        """Test model evaluation for regression."""
        y_true = sample_data['y_true_reg']
        y_pred = sample_data['y_pred_reg']
        
        result = evaluate_model(y_true, y_pred, None, TaskType.REGRESSION)
        
        assert result.task_type == TaskType.REGRESSION
        assert 'mse' in result.metrics
        assert 'mae' in result.metrics
        assert 'r2' in result.metrics
        assert result.confusion_matrix is None
    
    def test_compare_models(self, sample_data) -> Any:
        """Test model comparison."""
        y_true = sample_data['y_true']
        y_pred1 = sample_data['y_pred']
        y_pred2 = np.random.randint(0, 3, 100)
        y_prob = sample_data['y_prob']
        
        result1 = evaluate_model(y_true, y_pred1, y_prob, TaskType.CLASSIFICATION)
        result2 = evaluate_model(y_true, y_pred2, y_prob, TaskType.CLASSIFICATION)
        
        comparison = compare_models({
            'Model A': result1,
            'Model B': result2
        }, metric_name='f1')
        
        assert len(comparison.model_names) == 2
        assert 'Model A' in comparison.model_names
        assert 'Model B' in comparison.model_names
        assert comparison.best_model in comparison.model_names
        assert len(comparison.improvement_scores) == 2
    
    def test_rank_models(self, sample_data) -> Any:
        """Test model ranking."""
        y_true = sample_data['y_true']
        y_pred1 = sample_data['y_pred']
        y_pred2 = np.random.randint(0, 3, 100)
        y_prob = sample_data['y_prob']
        
        result1 = evaluate_model(y_true, y_pred1, y_prob, TaskType.CLASSIFICATION)
        result2 = evaluate_model(y_true, y_pred2, y_prob, TaskType.CLASSIFICATION)
        
        rankings = rank_models({
            'Model A': result1,
            'Model B': result2
        }, metric_name='f1')
        
        assert len(rankings) == 2
        assert all(isinstance(rank, tuple) for rank in rankings)
        assert all(len(rank) == 2 for rank in rankings)
        # Should be sorted by score (descending)
        scores = [rank[1] for rank in rankings]
        assert scores == sorted(scores, reverse=True)
    
    def test_quick_evaluate_classification(self, sample_data) -> Any:
        """Test quick classification evaluation."""
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        y_prob = sample_data['y_prob']
        
        metrics = quick_evaluate_classification(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_quick_evaluate_regression(self, sample_data) -> Any:
        """Test quick regression evaluation."""
        y_true = sample_data['y_true_reg']
        y_pred = sample_data['y_pred_reg']
        
        metrics = quick_evaluate_regression(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

# ============================================================================
# Tests for Functional FastAPI App
# ============================================================================

class TestFunctionalFastAPIApp:
    """Test functional FastAPI app functions."""
    
    def test_create_app_config(self) -> Any:
        """Test creating app configuration."""
        config = create_app_config()
        
        assert config['title'] == "Blatam Academy Functional NLP API"
        assert config['description'] == "Production-ready functional NLP training and inference API"
        assert config['version'] == "1.0.0"
    
    def test_create_cors_config(self) -> Any:
        """Test creating CORS configuration."""
        cors_config = create_cors_config()
        
        assert cors_config['allow_origins'] == ["*"]
        assert cors_config['allow_credentials'] is True
        assert cors_config['allow_methods'] == ["*"]
        assert cors_config['allow_headers'] == ["*"]
    
    def test_create_experiment_id(self, sample_training_request) -> Any:
        """Test creating experiment ID."""
        # With custom ID
        request_with_id = TrainingRequest(
            **sample_training_request.dict(),
            experiment_id="custom_id"
        )
        exp_id = create_experiment_id(request_with_id)
        assert exp_id == "custom_id"
        
        # Without custom ID (should generate one)
        exp_id = create_experiment_id(sample_training_request)
        assert exp_id.startswith("exp_")
        assert "distilbert-base-uncased" in exp_id
    
    def test_create_experiment_state(self, sample_training_request) -> Any:
        """Test creating experiment state."""
        exp_id = "test_exp"
        state = create_experiment_state(exp_id, sample_training_request)
        
        assert state['experiment_id'] == exp_id
        assert state['status'] == "starting"
        assert state['progress'] == 0.0
        assert state['current_epoch'] == 0
        assert state['total_epochs'] == sample_training_request.num_epochs
        assert 'start_time' in state
        assert 'request' in state
    
    def test_update_experiment_state_immutable(self, sample_training_request) -> Any:
        """Test that experiment state updates are immutable."""
        exp_id = "test_exp"
        original_state = create_experiment_state(exp_id, sample_training_request)
        
        updated_state = update_experiment_state(original_state, status="running", progress=0.5)
        
        # Original state should be unchanged
        assert original_state['status'] == "starting"
        assert original_state['progress'] == 0.0
        
        # Updated state should have new values
        assert updated_state['status'] == "running"
        assert updated_state['progress'] == 0.5
        
        # Should be different objects
        assert original_state is not updated_state
    
    def test_create_training_response(self) -> Any:
        """Test creating training response."""
        response = create_training_response(
            experiment_id="test_exp",
            status="started",
            message="Success",
            estimated_duration=300
        )
        
        assert response.experiment_id == "test_exp"
        assert response.status == "started"
        assert response.message == "Success"
        assert response.estimated_duration == 300
    
    def test_calculate_estimated_duration(self) -> Any:
        """Test calculating estimated duration."""
        # Quick training
        duration = calculate_estimated_duration(5, "quick")
        assert duration == 5 * 300  # 5 epochs * 300 seconds
        
        # Advanced training
        duration = calculate_estimated_duration(10, "advanced")
        assert duration == 10 * 400  # 10 epochs * 400 seconds
        
        # Config training
        duration = calculate_estimated_duration(3, "config")
        assert duration == 3 * 350  # 3 epochs * 350 seconds
    
    def test_handle_training_error(self) -> Any:
        """Test handling training errors."""
        error = ValueError("Test error")
        experiment_id = "test_exp"
        
        with pytest.raises(Exception) as exc_info:
            handle_training_error(error, experiment_id)
        
        assert "Training failed" in str(exc_info.value)

# ============================================================================
# Integration Tests
# ============================================================================

class TestFunctionalIntegration:
    """Test integration between functional modules."""
    
    def test_end_to_end_training_flow(self, sample_config) -> Any:
        """Test end-to-end training flow using functional approach."""
        # Create config
        config = create_default_config("test-model", "data/test.csv")
        config = update_config(config, num_epochs=2, batch_size=8)
        
        # Validate config
        is_valid, errors = validate_config(config)
        assert is_valid
        
        # Create experiment config
        exp_config = create_experiment_config("test_exp", "Test experiment", config)
        assert exp_config.config == config
        
        # Simulate training metrics
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.randint(0, 2, 50)
        y_prob = np.random.rand(50, 2)
        
        # Evaluate model
        result = evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION)
        assert result.task_type == TaskType.CLASSIFICATION
        assert 'accuracy' in result.metrics
    
    def test_config_loading_and_evaluation_pipeline(self, tmp_path) -> Any:
        """Test config loading and evaluation pipeline."""
        # Create sample config file
        config_data = {
            'model_type': 'transformer',
            'training_mode': 'fine_tune',
            'model_name': 'test-model',
            'dataset_path': 'data/test.csv',
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 5
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            yaml.dump(config_data, f)
        
        # Load config
        config = load_config_from_yaml(str(config_path))
        assert config.model_name == "test-model"
        
        # Validate config
        is_valid, errors = validate_config(config)
        assert is_valid
        
        # Create experiment config
        exp_config = create_experiment_config("test_exp", "Test", config)
        assert exp_config.config == config
    
    def test_functional_data_transformation_pipeline(self, sample_data) -> Any:
        """Test functional data transformation pipeline."""
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        y_prob = sample_data['y_prob']
        
        # Transform 1: Calculate metrics
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # Transform 2: Create evaluation result
        result = evaluate_model(y_true, y_pred, y_prob, TaskType.CLASSIFICATION)
        
        # Transform 3: Compare with another model
        y_pred2 = np.random.randint(0, 3, 100)
        result2 = evaluate_model(y_true, y_pred2, y_prob, TaskType.CLASSIFICATION)
        
        comparison = compare_models({
            'Model A': result,
            'Model B': result2
        })
        
        # Each transformation should be pure and immutable
        assert metrics != result.metrics  # Different data structures
        assert result != result2  # Different results
        assert comparison.model_names == ['Model A', 'Model B']

# ============================================================================
# Property-Based Tests
# ============================================================================

class TestFunctionalProperties:
    """Test functional properties using hypothesis."""
    
    @pytest.mark.skipif(True, reason="Requires hypothesis library")
    def test_config_immutability_property(self) -> Any:
        """Test that config updates are always immutable."""
        
        @given(
            model_name=st.text(min_size=1),
            dataset_path=st.text(min_size=1),
            batch_size=st.integers(min_value=1, max_value=128)
        )
        def config_immutability_property(model_name, dataset_path, batch_size) -> Any:
            config = create_default_config(model_name, dataset_path)
            original_batch_size = config.batch_size
            
            updated_config = update_config(config, batch_size=batch_size)
            
            # Original should be unchanged
            assert config.batch_size == original_batch_size
            # New should have updated value
            assert updated_config.batch_size == batch_size
            # Should be different objects
            assert config is not updated_config
        
        config_immutability_property()
    
    @pytest.mark.skipif(True, reason="Requires hypothesis library")
    def test_metrics_purity_property(self) -> Any:
        """Test that metric calculations are always pure."""
        
        @given(
            y_true=st.lists(st.integers(min_value=0, max_value=2), min_size=10),
            y_pred=st.lists(st.integers(min_value=0, max_value=2), min_size=10)
        )
        def metrics_purity_property(y_true, y_pred) -> Any:
            if len(y_true) == len(y_pred):
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                
                metrics1 = calculate_classification_metrics(y_true_arr, y_pred_arr)
                metrics2 = calculate_classification_metrics(y_true_arr, y_pred_arr)
                
                assert metrics1 == metrics2
        
        metrics_purity_property()

# ============================================================================
# Performance Tests
# ============================================================================

class TestFunctionalPerformance:
    """Test performance characteristics of functional approach."""
    
    def test_config_update_performance(self, sample_config) -> Any:
        """Test performance of config updates."""
        
        start_time = time.time()
        for _ in range(1000):
            update_config(sample_config, batch_size=np.random.randint(1, 128))
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second
    
    def test_metrics_calculation_performance(self, sample_data) -> Any:
        """Test performance of metrics calculation."""
        
        y_true = sample_data['y_true']
        y_pred = sample_data['y_pred']
        y_prob = sample_data['y_prob']
        
        start_time = time.time()
        for _ in range(100):
            calculate_classification_metrics(y_true, y_pred, y_prob)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second

# ============================================================================
# Demo Functions
# ============================================================================

def demo_functional_tests():
    """Demo the functional tests."""
    print("🧪 Functional Tests Demo")
    print("=" * 50)
    
    # Test config creation
    config = create_default_config("test-model", "data/test.csv")
    print(f"✅ Created config: {config.model_name}")
    
    # Test config update
    updated_config = update_config(config, batch_size=32)
    print(f"✅ Updated config: batch_size {updated_config.batch_size}")
    print(f"✅ Original config unchanged: batch_size {config.batch_size}")
    
    # Test validation
    is_valid, errors = validate_config(updated_config)
    print(f"✅ Config validation: {is_valid}")
    
    # Test metrics calculation
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    metrics = calculate_classification_metrics(y_true, y_pred)
    print(f"✅ Calculated metrics: {list(metrics.keys())}")
    
    # Test experiment config
    exp_config = create_experiment_config("test_exp", "Test experiment", config)
    print(f"✅ Created experiment: {exp_config.experiment_id}")
    
    print("\n🎉 All functional tests passed!")

match __name__:
    case "__main__":
    demo_functional_tests() 