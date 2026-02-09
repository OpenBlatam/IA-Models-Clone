"""
ML Testing Helpers
Specialized helpers for ML/AI component testing
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock
import pytest


class MLTestHelpers:
    """Helpers for ML model testing"""
    
    @staticmethod
    def create_mock_model(
        input_shape: Tuple[int, ...] = (224, 224, 3),
        output_shape: Tuple[int, ...] = (10,),
        prediction_value: float = 0.75
    ) -> Mock:
        """Create mock ML model"""
        model = Mock()
        model.predict = Mock(return_value=np.array([prediction_value] * output_shape[0]))
        model.predict_batch = Mock(return_value=np.array([[prediction_value] * output_shape[0]]))
        model.input_shape = input_shape
        model.output_shape = output_shape
        return model
    
    @staticmethod
    def create_mock_model_manager(
        models: Optional[Dict[str, Mock]] = None,
        default_model: Optional[str] = None
    ) -> Mock:
        """Create mock model manager"""
        manager = Mock()
        manager.models = models or {}
        manager.default_model = default_model
        manager.load_model = AsyncMock(return_value=Mock())
        manager.get_model = Mock(return_value=manager.models.get(default_model or "default"))
        manager.predict = AsyncMock(return_value={"prediction": 0.75, "confidence": 0.9})
        return manager
    
    @staticmethod
    def assert_prediction_valid(prediction: Dict[str, Any], expected_keys: List[str] = None):
        """Assert prediction result is valid"""
        expected_keys = expected_keys or ["prediction", "confidence"]
        for key in expected_keys:
            assert key in prediction, f"Prediction missing key: {key}"
        
        if "prediction" in prediction:
            assert isinstance(prediction["prediction"], (int, float, np.number))
        
        if "confidence" in prediction:
            assert 0 <= prediction["confidence"] <= 1
    
    @staticmethod
    def create_test_image_array(
        shape: Tuple[int, ...] = (224, 224, 3),
        dtype: type = np.uint8,
        value_range: Tuple[int, int] = (0, 255)
    ) -> np.ndarray:
        """Create test image array"""
        return np.random.randint(
            value_range[0],
            value_range[1],
            size=shape,
            dtype=dtype
        )
    
    @staticmethod
    def assert_image_array_valid(image: np.ndarray, expected_shape: Tuple[int, ...] = None):
        """Assert image array is valid"""
        assert isinstance(image, np.ndarray), "Image must be numpy array"
        assert len(image.shape) >= 2, "Image must have at least 2 dimensions"
        if expected_shape:
            assert image.shape == expected_shape, \
                f"Image shape {image.shape} does not match expected {expected_shape}"


class ImageProcessingHelpers:
    """Helpers for image processing testing"""
    
    @staticmethod
    def create_mock_image_processor(
        return_metrics: Optional[Dict[str, float]] = None,
        return_conditions: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock image processor"""
        processor = Mock()
        processor.process = AsyncMock(return_value={
            "metrics": return_metrics or {
                "overall_score": 75.0,
                "texture_score": 80.0,
                "hydration_score": 70.0
            },
            "conditions": return_conditions or []
        })
        processor.validate = AsyncMock(return_value=True)
        processor.enhance = AsyncMock(return_value=np.array([[[255, 255, 255]]]))
        return processor
    
    @staticmethod
    def assert_metrics_valid(metrics: Dict[str, float], min_score: float = 0.0, max_score: float = 100.0):
        """Assert metrics are valid"""
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"Metric {key} must be numeric"
            assert min_score <= value <= max_score, \
                f"Metric {key} ({value}) not in range [{min_score}, {max_score}]"
    
    @staticmethod
    def assert_conditions_valid(conditions: List[Dict[str, Any]]):
        """Assert conditions are valid"""
        for condition in conditions:
            assert "name" in condition, "Condition must have name"
            assert "confidence" in condition, "Condition must have confidence"
            assert 0 <= condition["confidence"] <= 1, \
                f"Confidence {condition['confidence']} not in range [0, 1]"


class ModelTrainingHelpers:
    """Helpers for model training testing"""
    
    @staticmethod
    def create_mock_training_config(
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Create mock training configuration"""
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "adam",
            "loss": "categorical_crossentropy"
        }
    
    @staticmethod
    def create_mock_training_history(
        epochs: int = 10,
        metrics: List[str] = None
    ) -> Dict[str, List[float]]:
        """Create mock training history"""
        metrics = metrics or ["loss", "accuracy", "val_loss", "val_accuracy"]
        history = {}
        for metric in metrics:
            history[metric] = [np.random.random() for _ in range(epochs)]
        return history
    
    @staticmethod
    def assert_training_history_valid(history: Dict[str, List[float]]):
        """Assert training history is valid"""
        assert isinstance(history, dict), "History must be dictionary"
        assert len(history) > 0, "History must contain metrics"
        for metric, values in history.items():
            assert isinstance(values, list), f"Metric {metric} must be list"
            assert len(values) > 0, f"Metric {metric} must have values"
            assert all(isinstance(v, (int, float)) for v in values), \
                f"Metric {metric} values must be numeric"


class ExperimentHelpers:
    """Helpers for experiment tracking testing"""
    
    @staticmethod
    def create_mock_experiment(
        name: str = "test_experiment",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> Mock:
        """Create mock experiment"""
        experiment = Mock()
        experiment.name = name
        experiment.parameters = parameters or {}
        experiment.metrics = metrics or {}
        experiment.status = "running"
        experiment.start = Mock()
        experiment.end = Mock()
        experiment.log_metric = Mock()
        experiment.log_parameter = Mock()
        return experiment
    
    @staticmethod
    def assert_experiment_valid(experiment: Mock):
        """Assert experiment is valid"""
        assert hasattr(experiment, "name"), "Experiment must have name"
        assert hasattr(experiment, "parameters"), "Experiment must have parameters"
        assert hasattr(experiment, "metrics"), "Experiment must have metrics"
        assert experiment.status in ["running", "completed", "failed"], \
            f"Invalid experiment status: {experiment.status}"


# Convenience exports
create_mock_model = MLTestHelpers.create_mock_model
create_mock_model_manager = MLTestHelpers.create_mock_model_manager
assert_prediction_valid = MLTestHelpers.assert_prediction_valid
create_test_image_array = MLTestHelpers.create_test_image_array
assert_image_array_valid = MLTestHelpers.assert_image_array_valid

create_mock_image_processor = ImageProcessingHelpers.create_mock_image_processor
assert_metrics_valid = ImageProcessingHelpers.assert_metrics_valid
assert_conditions_valid = ImageProcessingHelpers.assert_conditions_valid

create_mock_training_config = ModelTrainingHelpers.create_mock_training_config
create_mock_training_history = ModelTrainingHelpers.create_mock_training_history
assert_training_history_valid = ModelTrainingHelpers.assert_training_history_valid

create_mock_experiment = ExperimentHelpers.create_mock_experiment
assert_experiment_valid = ExperimentHelpers.assert_experiment_valid



