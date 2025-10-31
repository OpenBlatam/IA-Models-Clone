from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from model_training_evaluation import (
        import shutil
        from transformers import AutoTokenizer
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
            import shutil
            import shutil
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Model Training and Evaluation System

This test suite covers:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance tests for training and inference
- Edge case handling and error scenarios
- Model validation and quality checks
- Security and robustness testing
"""



    ModelTrainer, ModelEvaluator, HyperparameterOptimizer,
    ModelVersionManager, ModelDeploymentManager, ModelType,
    TrainingConfig, EvaluationMetrics, create_model_trainer,
    calculate_model_complexity, validate_model_performance,
    run_ab_test, ThreatDetectionDataset, AnomalyDetectionDataset,
    BaseDataset
)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_training_config_creation(self) -> Any:
        """Test creating TrainingConfig with default values."""
        config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="test-model",
            dataset_path="test.csv"
        )
        
        assert config.model_type == ModelType.THREAT_DETECTION
        assert config.model_name == "test-model"
        assert config.dataset_path == "test.csv"
        assert config.validation_split == 0.2
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
    
    def test_training_config_custom_values(self) -> Any:
        """Test creating TrainingConfig with custom values."""
        config = TrainingConfig(
            model_type=ModelType.ANOMALY_DETECTION,
            model_name="custom-model",
            dataset_path="custom.csv",
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=20
        )
        
        assert config.model_type == ModelType.ANOMALY_DETECTION
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 20


class TestEvaluationMetrics:
    """Test EvaluationMetrics dataclass."""
    
    def test_evaluation_metrics_creation(self) -> Any:
        """Test creating EvaluationMetrics."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92,
            confusion_matrix=np.array([[100, 10], [15, 75]]),
            false_positive_rate=0.1,
            false_negative_rate=0.12,
            true_positive_rate=0.88,
            true_negative_rate=0.9,
            inference_time=0.05,
            training_time=120.5,
            model_size_mb=45.2
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.85
        assert metrics.inference_time == 0.05
        assert metrics.model_size_mb == 45.2
    
    def test_evaluation_metrics_with_regression(self) -> Any:
        """Test EvaluationMetrics with regression metrics."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92,
            confusion_matrix=np.array([[100, 10], [15, 75]]),
            mse=0.15,
            mae=0.12,
            r2_score=0.78,
            false_positive_rate=0.1,
            false_negative_rate=0.12,
            true_positive_rate=0.88,
            true_negative_rate=0.9,
            inference_time=0.05,
            training_time=120.5,
            model_size_mb=45.2
        )
        
        assert metrics.mse == 0.15
        assert metrics.mae == 0.12
        assert metrics.r2_score == 0.78


class TestBaseDataset:
    """Test BaseDataset abstract class."""
    
    def test_base_dataset_abstract_methods(self) -> Any:
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset("test.csv")


class TestThreatDetectionDataset:
    """Test ThreatDetectionDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "threat_data.csv")
        
        # Create test data
        data = {
            'text': [
                "Normal network traffic",
                "Suspicious activity detected",
                "Malicious payload identified",
                "Legitimate user request"
            ],
            'label': [0, 1, 1, 0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_threat_detection_dataset_creation(self) -> Any:
        """Test creating ThreatDetectionDataset."""
        dataset = ThreatDetectionDataset(self.dataset_path)
        
        assert len(dataset) == 4
        assert len(dataset.data) == 4
        assert len(dataset.labels) == 4
        assert dataset.labels == [0, 1, 1, 0]
    
    def test_threat_detection_dataset_with_tokenizer(self) -> Any:
        """Test ThreatDetectionDataset with tokenizer."""
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        dataset = ThreatDetectionDataset(self.dataset_path, tokenizer, max_length=128)
        
        assert len(dataset) == 4
        assert isinstance(dataset.data[0], dict)
        assert 'input_ids' in dataset.data[0]
        assert 'attention_mask' in dataset.data[0]
    
    def test_threat_detection_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test ThreatDetectionDataset __getitem__ method."""
        dataset = ThreatDetectionDataset(self.dataset_path)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert item[1] == 0  # label
    
    def test_threat_detection_dataset_invalid_path(self) -> Any:
        """Test ThreatDetectionDataset with invalid path."""
        with pytest.raises(Exception):
            ThreatDetectionDataset("nonexistent.csv")


class TestAnomalyDetectionDataset:
    """Test AnomalyDetectionDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "anomaly_data.csv")
        
        # Create test data
        features = [
            json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            json.dumps([0.2, 0.3, 0.4, 0.5, 0.6]),
            json.dumps([0.3, 0.4, 0.5, 0.6, 0.7]),
            json.dumps([0.4, 0.5, 0.6, 0.7, 0.8])
        ]
        
        data = {
            'features': features,
            'label': [0, 0, 1, 1]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_anomaly_detection_dataset_creation(self) -> Any:
        """Test creating AnomalyDetectionDataset."""
        dataset = AnomalyDetectionDataset(self.dataset_path)
        
        assert len(dataset) == 4
        assert isinstance(dataset.data, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
        assert dataset.data.shape[1] == 5  # 5 features
        assert dataset.labels.shape[0] == 4
    
    def test_anomaly_detection_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test AnomalyDetectionDataset __getitem__ method."""
        dataset = AnomalyDetectionDataset(self.dataset_path)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)


class TestModelTrainer:
    """Test ModelTrainer class."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path="test.csv",
            output_dir=os.path.join(self.temp_dir, "models"),
            logging_dir=os.path.join(self.temp_dir, "logs"),
            num_epochs=1,
            batch_size=4
        )
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_trainer_initialization(self) -> Any:
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(self.config)
        
        assert trainer.config == self.config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None
    
    def test_setup_directories(self) -> Any:
        """Test directory setup."""
        trainer = ModelTrainer(self.config)
        
        assert os.path.exists(self.config.output_dir)
        assert os.path.exists(self.config.cache_dir)
        assert os.path.exists(self.config.logging_dir)
    
    @patch('model_training_evaluation.mlflow.set_tracking_uri')
    @patch('model_training_evaluation.mlflow.set_experiment')
    def test_setup_mlflow(self, mock_set_experiment, mock_set_tracking_uri) -> Any:
        """Test MLflow setup."""
        trainer = ModelTrainer(self.config)
        
        mock_set_tracking_uri.assert_called_once_with("sqlite:///mlflow.db")
        mock_set_experiment.assert_called_once_with("cybersecurity_threat_detection")
    
    def test_setup_model_threat_detection(self) -> Any:
        """Test model setup for threat detection."""
        trainer = ModelTrainer(self.config)
        trainer._setup_model()
        
        assert trainer.model is not None
        assert trainer.tokenizer is not None
    
    def test_setup_model_anomaly_detection(self) -> Any:
        """Test model setup for anomaly detection."""
        config = TrainingConfig(
            model_type=ModelType.ANOMALY_DETECTION,
            model_name="autoencoder",
            dataset_path="test.csv",
            output_dir=os.path.join(self.temp_dir, "models"),
            logging_dir=os.path.join(self.temp_dir, "logs")
        )
        
        trainer = ModelTrainer(config)
        trainer._setup_model()
        
        assert trainer.model is not None
        assert trainer.tokenizer is None
    
    def test_setup_model_unsupported_type(self) -> Any:
        """Test model setup with unsupported type."""
        config = TrainingConfig(
            model_type=ModelType.LOG_ANALYSIS,  # Unsupported
            model_name="test",
            dataset_path="test.csv"
        )
        
        trainer = ModelTrainer(config)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._setup_model()
    
    def test_setup_training_arguments(self) -> Any:
        """Test training arguments setup."""
        trainer = ModelTrainer(self.config)
        training_args = trainer._setup_training_arguments()
        
        assert training_args.output_dir == self.config.output_dir
        assert training_args.num_train_epochs == self.config.num_epochs
        assert training_args.per_device_train_batch_size == self.config.batch_size
        assert training_args.learning_rate == self.config.learning_rate
    
    def test_calculate_model_size(self) -> Any:
        """Test model size calculation."""
        trainer = ModelTrainer(self.config)
        
        # Create a dummy model file
        model_path = os.path.join(self.temp_dir, "dummy_model")
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, "model.pt"), "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("dummy model content")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        size = trainer._calculate_model_size(model_path)
        assert size > 0
    
    def test_get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Test dataset info extraction."""
        trainer = ModelTrainer(self.config)
        
        # Create dummy dataset
        data = {'text': ['a', 'b', 'c'], 'label': [0, 1, 0]}
        df = pd.DataFrame(data)
        df.to_csv(self.config.dataset_path, index=False)
        
        info = trainer._get_dataset_info()
        assert info["total_samples"] == 3
        assert "text" in info["columns"]
        assert "label" in info["columns"]
    
    def test_get_dependencies(self) -> Optional[Dict[str, Any]]:
        """Test dependency extraction."""
        trainer = ModelTrainer(self.config)
        deps = trainer._get_dependencies()
        
        assert "torch" in deps
        assert "numpy" in deps
        assert "pandas" in deps


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        os.makedirs(self.model_path, exist_ok=True)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('model_training_evaluation.AutoModelForSequenceClassification.from_pretrained')
    @patch('model_training_evaluation.AutoTokenizer.from_pretrained')
    def test_model_evaluator_initialization(self, mock_tokenizer, mock_model) -> Any:
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(self.model_path, ModelType.THREAT_DETECTION)
        
        assert evaluator.model_path == self.model_path
        assert evaluator.model_type == ModelType.THREAT_DETECTION
        mock_model.assert_called_once_with(self.model_path)
        mock_tokenizer.assert_called_once_with(self.model_path)
    
    def test_load_model_invalid_path(self) -> Any:
        """Test loading model from invalid path."""
        with pytest.raises(Exception):
            ModelEvaluator("nonexistent_path", ModelType.THREAT_DETECTION)
    
    def test_calculate_metrics_threat_detection(self) -> Any:
        """Test metrics calculation for threat detection."""
        evaluator = ModelEvaluator(self.model_path, ModelType.THREAT_DETECTION)
        
        # Mock predictions and labels
        predictions = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        true_labels = np.array([0, 1, 0, 1])
        
        metrics = evaluator._calculate_metrics(predictions, true_labels, 0.05)
        
        assert metrics.accuracy > 0
        assert metrics.precision > 0
        assert metrics.recall > 0
        assert metrics.f1_score > 0
        assert metrics.roc_auc > 0
        assert metrics.inference_time == 0.05
    
    def test_calculate_metrics_anomaly_detection(self) -> Any:
        """Test metrics calculation for anomaly detection."""
        evaluator = ModelEvaluator(self.model_path, ModelType.ANOMALY_DETECTION)
        
        # Mock predictions and labels
        predictions = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        true_labels = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        
        metrics = evaluator._calculate_metrics(predictions, true_labels, 0.03)
        
        assert metrics.accuracy > 0
        assert metrics.inference_time == 0.03


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create dummy dataset
        data = {'text': ['a', 'b', 'c'], 'label': [0, 1, 0]}
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_hyperparameter_optimizer_initialization(self) -> Any:
        """Test HyperparameterOptimizer initialization."""
        optimizer = HyperparameterOptimizer(
            ModelType.THREAT_DETECTION,
            self.dataset_path
        )
        
        assert optimizer.model_type == ModelType.THREAT_DETECTION
        assert optimizer.dataset_path == self.dataset_path
        assert optimizer.study is None
    
    @patch('model_training_evaluation.ModelTrainer')
    @patch('model_training_evaluation.optuna.create_study')
    def test_optimize(self, mock_create_study, mock_trainer) -> Any:
        """Test hyperparameter optimization."""
        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 1e-4, "batch_size": 32}
        mock_study.best_value = 0.85
        mock_create_study.return_value = mock_study
        
        # Mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train = AsyncMock()
        mock_trainer_instance.train.return_value.evaluation_metrics.f1_score = 0.85
        mock_trainer.return_value = mock_trainer_instance
        
        optimizer = HyperparameterOptimizer(
            ModelType.THREAT_DETECTION,
            self.dataset_path
        )
        
        best_params = optimizer.optimize(n_trials=2)
        
        assert best_params == {"learning_rate": 1e-4, "batch_size": 32}
        assert optimizer.study == mock_study


class TestModelVersionManager:
    """Test ModelVersionManager class."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_version_manager_initialization(self) -> Any:
        """Test ModelVersionManager initialization."""
        manager = ModelVersionManager(self.models_dir)
        
        assert manager.models_dir == Path(self.models_dir)
        assert manager.metadata_file == Path(self.models_dir) / "metadata.json"
        assert manager.metadata == {}
    
    def test_register_model(self) -> Any:
        """Test model registration."""
        manager = ModelVersionManager(self.models_dir)
        
        # Create mock metadata
        metadata = MagicMock()
        metadata.model_id = "test-id"
        metadata.model_type = ModelType.THREAT_DETECTION
        metadata.model_name = "test-model"
        metadata.version = "1.0.0"
        metadata.created_at = "2023-01-01T00:00:00"
        metadata.is_production = False
        metadata.model_path = "/path/to/model"
        metadata.artifacts_path = "/path/to/artifacts"
        metadata.evaluation_metrics.accuracy = 0.85
        metadata.evaluation_metrics.f1_score = 0.82
        metadata.evaluation_metrics.precision = 0.80
        metadata.evaluation_metrics.recall = 0.85
        metadata.tags = ["test"]
        
        manager.register_model(metadata)
        
        assert "test-id" in manager.metadata
        assert manager.metadata["test-id"]["model_name"] == "test-model"
        assert manager.metadata["test-id"]["model_type"] == "threat_detection"
    
    def test_get_model(self) -> Optional[Dict[str, Any]]:
        """Test getting model by ID."""
        manager = ModelVersionManager(self.models_dir)
        
        # Register a model first
        metadata = MagicMock()
        metadata.model_id = "test-id"
        metadata.model_type = ModelType.THREAT_DETECTION
        metadata.model_name = "test-model"
        metadata.version = "1.0.0"
        metadata.created_at = "2023-01-01T00:00:00"
        metadata.is_production = False
        metadata.model_path = "/path/to/model"
        metadata.artifacts_path = "/path/to/artifacts"
        metadata.evaluation_metrics.accuracy = 0.85
        metadata.evaluation_metrics.f1_score = 0.82
        metadata.evaluation_metrics.precision = 0.80
        metadata.evaluation_metrics.recall = 0.85
        metadata.tags = ["test"]
        
        manager.register_model(metadata)
        
        # Get the model
        model_data = manager.get_model("test-id")
        assert model_data is not None
        assert model_data["model_name"] == "test-model"
        
        # Get non-existent model
        assert manager.get_model("nonexistent") is None
    
    def test_list_models(self) -> List[Any]:
        """Test listing models."""
        manager = ModelVersionManager(self.models_dir)
        
        # Register multiple models
        for i in range(3):
            metadata = MagicMock()
            metadata.model_id = f"test-id-{i}"
            metadata.model_type = ModelType.THREAT_DETECTION if i % 2 == 0 else ModelType.ANOMALY_DETECTION
            metadata.model_name = f"test-model-{i}"
            metadata.version = f"1.0.{i}"
            metadata.created_at = f"2023-01-0{i+1}T00:00:00"
            metadata.is_production = False
            metadata.model_path = f"/path/to/model-{i}"
            metadata.artifacts_path = f"/path/to/artifacts-{i}"
            metadata.evaluation_metrics.accuracy = 0.85
            metadata.evaluation_metrics.f1_score = 0.82
            metadata.evaluation_metrics.precision = 0.80
            metadata.evaluation_metrics.recall = 0.85
            metadata.tags = ["test"]
            
            manager.register_model(metadata)
        
        # List all models
        all_models = manager.list_models()
        assert len(all_models) == 3
        
        # List models by type
        threat_models = manager.list_models(ModelType.THREAT_DETECTION)
        assert len(threat_models) == 2
        
        anomaly_models = manager.list_models(ModelType.ANOMALY_DETECTION)
        assert len(anomaly_models) == 1
    
    def test_set_production_model(self) -> Any:
        """Test setting production model."""
        manager = ModelVersionManager(self.models_dir)
        
        # Register models
        for i in range(2):
            metadata = MagicMock()
            metadata.model_id = f"test-id-{i}"
            metadata.model_type = ModelType.THREAT_DETECTION
            metadata.model_name = f"test-model-{i}"
            metadata.version = f"1.0.{i}"
            metadata.created_at = f"2023-01-0{i+1}T00:00:00"
            metadata.is_production = False
            metadata.model_path = f"/path/to/model-{i}"
            metadata.artifacts_path = f"/path/to/artifacts-{i}"
            metadata.evaluation_metrics.accuracy = 0.85
            metadata.evaluation_metrics.f1_score = 0.82
            metadata.evaluation_metrics.precision = 0.80
            metadata.evaluation_metrics.recall = 0.85
            metadata.tags = ["test"]
            
            manager.register_model(metadata)
        
        # Set production model
        manager.set_production_model("test-id-1", ModelType.THREAT_DETECTION)
        
        # Check that only one model is production
        production_models = [m for m in manager.metadata.values() if m["is_production"]]
        assert len(production_models) == 1
        assert production_models[0]["model_id"] == "test-id-1"
    
    def test_get_production_model(self) -> Optional[Dict[str, Any]]:
        """Test getting production model."""
        manager = ModelVersionManager(self.models_dir)
        
        # Register a model and set it as production
        metadata = MagicMock()
        metadata.model_id = "test-id"
        metadata.model_type = ModelType.THREAT_DETECTION
        metadata.model_name = "test-model"
        metadata.version = "1.0.0"
        metadata.created_at = "2023-01-01T00:00:00"
        metadata.is_production = False
        metadata.model_path = "/path/to/model"
        metadata.artifacts_path = "/path/to/artifacts"
        metadata.evaluation_metrics.accuracy = 0.85
        metadata.evaluation_metrics.f1_score = 0.82
        metadata.evaluation_metrics.precision = 0.80
        metadata.evaluation_metrics.recall = 0.85
        metadata.tags = ["test"]
        
        manager.register_model(metadata)
        manager.set_production_model("test-id", ModelType.THREAT_DETECTION)
        
        # Get production model
        production_model = manager.get_production_model(ModelType.THREAT_DETECTION)
        assert production_model is not None
        assert production_model["model_id"] == "test-id"
        
        # Get production model for non-existent type
        assert manager.get_production_model(ModelType.ANOMALY_DETECTION) is None


class TestModelDeploymentManager:
    """Test ModelDeploymentManager class."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_deployment_manager_initialization(self) -> Any:
        """Test ModelDeploymentManager initialization."""
        manager = ModelDeploymentManager(self.models_dir)
        
        assert manager.models_dir == Path(self.models_dir)
        assert isinstance(manager.version_manager, ModelVersionManager)
        assert manager.deployed_models == {}
    
    @patch('model_training_evaluation.AutoModelForSequenceClassification.from_pretrained')
    @patch('model_training_evaluation.AutoTokenizer.from_pretrained')
    async def test_deploy_model(self, mock_tokenizer, mock_model) -> Any:
        """Test model deployment."""
        manager = ModelDeploymentManager(self.models_dir)
        
        # Mock version manager
        mock_model_data = {
            "model_id": "test-id",
            "model_path": "/path/to/model",
            "model_type": "threat_detection"
        }
        manager.version_manager.get_production_model = MagicMock(return_value=mock_model_data)
        manager.version_manager.get_model = MagicMock(return_value=mock_model_data)
        
        deployment_id = await manager.deploy_model(ModelType.THREAT_DETECTION)
        
        assert deployment_id == "threat_detection_test-id"
        assert deployment_id in manager.deployed_models
        mock_model.assert_called_once_with("/path/to/model")
        mock_tokenizer.assert_called_once_with("/path/to/model")
    
    async def test_deploy_model_no_production(self) -> Any:
        """Test deploying model when no production model exists."""
        manager = ModelDeploymentManager(self.models_dir)
        manager.version_manager.get_production_model = MagicMock(return_value=None)
        
        with pytest.raises(ValueError, match="No production model found"):
            await manager.deploy_model(ModelType.THREAT_DETECTION)
    
    async def test_predict(self) -> Any:
        """Test model prediction."""
        manager = ModelDeploymentManager(self.models_dir)
        
        # Mock deployed model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        
        manager.deployed_models["test-deployment"] = {
            "model": mock_model,
            "tokenizer": MagicMock(),
            "model_type": ModelType.THREAT_DETECTION,
            "model_id": "test-id"
        }
        
        # Mock prediction
        with patch('torch.device') as mock_device:
            mock_device.return_value = "cpu"
            
            with patch('torch.no_grad'):
                mock_model.return_value.logits = torch.tensor([[0.8, 0.2]])
                mock_model.return_value.logits.softmax.return_value = torch.tensor([[0.8, 0.2]])
                mock_model.return_value.logits.softmax.return_value.argmax.return_value = torch.tensor([0])
                mock_model.return_value.logits.softmax.return_value.argmax.return_value.item.return_value = 0
                mock_model.return_value.logits.softmax.return_value.__getitem__.return_value.__getitem__.return_value.item.return_value = 0.8
                
                result = await manager.predict("test-deployment", "test input")
                
                assert "prediction" in result
                assert "confidence" in result
                assert "inference_time" in result
                assert result["model_id"] == "test-id"
    
    async def test_predict_undeployed_model(self) -> Any:
        """Test prediction with undeployed model."""
        manager = ModelDeploymentManager(self.models_dir)
        
        with pytest.raises(ValueError, match="Model test-deployment not deployed"):
            await manager.predict("test-deployment", "test input")
    
    def test_list_deployed_models(self) -> List[Any]:
        """Test listing deployed models."""
        manager = ModelDeploymentManager(self.models_dir)
        
        # Add some deployed models
        manager.deployed_models["deployment-1"] = {}
        manager.deployed_models["deployment-2"] = {}
        
        deployed_models = manager.list_deployed_models()
        assert len(deployed_models) == 2
        assert "deployment-1" in deployed_models
        assert "deployment-2" in deployed_models
    
    def test_undeploy_model(self) -> Any:
        """Test model undeployment."""
        manager = ModelDeploymentManager(self.models_dir)
        
        # Add a deployed model
        manager.deployed_models["test-deployment"] = {}
        
        # Undeploy it
        manager.undeploy_model("test-deployment")
        
        assert "test-deployment" not in manager.deployed_models
    
    def test_undeploy_nonexistent_model(self) -> Any:
        """Test undeploying non-existent model."""
        manager = ModelDeploymentManager(self.models_dir)
        
        with pytest.raises(ValueError, match="Model test-deployment not deployed"):
            manager.undeploy_model("test-deployment")


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_model_complexity(self) -> Any:
        """Test model complexity calculation."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        complexity = calculate_model_complexity(model)
        
        assert "total_parameters" in complexity
        assert "trainable_parameters" in complexity
        assert "non_trainable_parameters" in complexity
        assert complexity["total_parameters"] > 0
        assert complexity["trainable_parameters"] > 0
    
    def test_validate_model_performance(self) -> bool:
        """Test model performance validation."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92,
            confusion_matrix=np.array([[100, 10], [15, 75]]),
            false_positive_rate=0.1,
            false_negative_rate=0.12,
            true_positive_rate=0.88,
            true_negative_rate=0.9,
            inference_time=0.05,
            training_time=120.5,
            model_size_mb=45.2
        )
        
        thresholds = {
            "min_accuracy": 0.8,
            "min_f1": 0.8,
            "max_fpr": 0.15,
            "max_inference_time": 1.0
        }
        
        is_valid = validate_model_performance(metrics, thresholds)
        assert is_valid is True
        
        # Test with stricter thresholds
        strict_thresholds = {
            "min_accuracy": 0.9,
            "min_f1": 0.9,
            "max_fpr": 0.05,
            "max_inference_time": 0.01
        }
        
        is_valid = validate_model_performance(metrics, strict_thresholds)
        assert is_valid is False
    
    def test_create_model_trainer(self) -> Any:
        """Test model trainer factory function."""
        trainer = create_model_trainer(
            ModelType.THREAT_DETECTION,
            model_name="test-model",
            dataset_path="test.csv",
            batch_size=64
        )
        
        assert isinstance(trainer, ModelTrainer)
        assert trainer.config.model_type == ModelType.THREAT_DETECTION
        assert trainer.config.model_name == "test-model"
        assert trainer.config.batch_size == 64


class TestIntegrationTests:
    """Integration tests for complete workflows."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Create test dataset
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        texts = [f"Sample text {i} with features: {', '.join([f'f{j}={v:.3f}' for j, v in enumerate(features[:5])])}" 
                for i, features in enumerate(X)]
        
        df = pd.DataFrame({'text': texts, 'label': y})
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_training_evaluation_workflow(self) -> Any:
        """Test complete training and evaluation workflow."""
        # Create config
        config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="distilbert-base-uncased",
            dataset_path=self.dataset_path,
            output_dir=os.path.join(self.temp_dir, "models"),
            logging_dir=os.path.join(self.temp_dir, "logs"),
            num_epochs=1,
            batch_size=4
        )
        
        # Train model
        trainer = ModelTrainer(config)
        metadata = await trainer.train()
        
        assert metadata is not None
        assert metadata.model_id is not None
        assert metadata.model_path is not None
        
        # Evaluate model
        evaluator = ModelEvaluator(metadata.model_path, config.model_type)
        metrics = await evaluator.evaluate(self.dataset_path)
        
        assert metrics is not None
        assert metrics.accuracy > 0
        assert metrics.f1_score > 0
    
    @pytest.mark.asyncio
    async def test_model_versioning_workflow(self) -> Any:
        """Test complete model versioning workflow."""
        # Create version manager
        version_manager = ModelVersionManager(self.temp_dir)
        
        # Create mock metadata
        metadata = MagicMock()
        metadata.model_id = "test-id"
        metadata.model_type = ModelType.THREAT_DETECTION
        metadata.model_name = "test-model"
        metadata.version = "1.0.0"
        metadata.created_at = "2023-01-01T00:00:00"
        metadata.is_production = False
        metadata.model_path = "/path/to/model"
        metadata.artifacts_path = "/path/to/artifacts"
        metadata.evaluation_metrics.accuracy = 0.85
        metadata.evaluation_metrics.f1_score = 0.82
        metadata.evaluation_metrics.precision = 0.80
        metadata.evaluation_metrics.recall = 0.85
        metadata.tags = ["test"]
        
        # Register model
        version_manager.register_model(metadata)
        
        # Set as production
        version_manager.set_production_model("test-id", ModelType.THREAT_DETECTION)
        
        # Get production model
        production_model = version_manager.get_production_model(ModelType.THREAT_DETECTION)
        assert production_model is not None
        assert production_model["model_id"] == "test-id"
    
    @pytest.mark.asyncio
    async def test_deployment_workflow(self) -> Any:
        """Test complete deployment workflow."""
        # Create deployment manager
        deployment_manager = ModelDeploymentManager(self.temp_dir)
        
        # Mock version manager
        mock_model_data = {
            "model_id": "test-id",
            "model_path": "/path/to/model",
            "model_type": "threat_detection"
        }
        deployment_manager.version_manager.get_production_model = MagicMock(return_value=mock_model_data)
        deployment_manager.version_manager.get_model = MagicMock(return_value=mock_model_data)
        
        # Deploy model
        with patch('model_training_evaluation.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
             patch('model_training_evaluation.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            deployment_id = await deployment_manager.deploy_model(ModelType.THREAT_DETECTION)
            
            assert deployment_id in deployment_manager.deployed_models
            
            # List deployed models
            deployed_models = deployment_manager.list_deployed_models()
            assert deployment_id in deployed_models
            
            # Undeploy model
            deployment_manager.undeploy_model(deployment_id)
            assert deployment_id not in deployment_manager.deployed_models


class TestPerformanceTests:
    """Performance tests."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_complexity_performance(self) -> Any:
        """Test model complexity calculation performance."""
        # Create a large model
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
        
        start_time = time.time()
        complexity = calculate_model_complexity(model)
        calculation_time = time.time() - start_time
        
        assert calculation_time < 1.0  # Should complete within 1 second
        assert complexity["total_parameters"] > 1000000  # Large model
    
    def test_metrics_validation_performance(self) -> Any:
        """Test metrics validation performance."""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            roc_auc=0.92,
            confusion_matrix=np.array([[100, 10], [15, 75]]),
            false_positive_rate=0.1,
            false_negative_rate=0.12,
            true_positive_rate=0.88,
            true_negative_rate=0.9,
            inference_time=0.05,
            training_time=120.5,
            model_size_mb=45.2
        )
        
        thresholds = {
            "min_accuracy": 0.8,
            "min_f1": 0.8,
            "max_fpr": 0.15,
            "max_inference_time": 1.0
        }
        
        start_time = time.time()
        for _ in range(1000):  # Test many validations
            validate_model_performance(metrics, thresholds)
        validation_time = time.time() - start_time
        
        assert validation_time < 1.0  # Should complete within 1 second


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_type(self) -> Any:
        """Test handling of invalid model types."""
        with pytest.raises(ValueError):
            TrainingConfig(
                model_type="invalid_type",  # Invalid
                model_name="test",
                dataset_path="test.csv"
            )
    
    def test_invalid_dataset_path(self) -> Any:
        """Test handling of invalid dataset paths."""
        config = TrainingConfig(
            model_type=ModelType.THREAT_DETECTION,
            model_name="test",
            dataset_path="nonexistent.csv"
        )
        
        trainer = ModelTrainer(config)
        
        with pytest.raises(Exception):
            trainer._get_dataset_info()
    
    def test_empty_dataset(self) -> Any:
        """Test handling of empty datasets."""
        temp_dir = tempfile.mkdtemp()
        try:
            empty_dataset_path = os.path.join(temp_dir, "empty.csv")
            
            # Create empty dataset
            df = pd.DataFrame(columns=['text', 'label'])
            df.to_csv(empty_dataset_path, index=False)
            
            with pytest.raises(Exception):
                ThreatDetectionDataset(empty_dataset_path)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_malformed_dataset(self) -> Any:
        """Test handling of malformed datasets."""
        temp_dir = tempfile.mkdtemp()
        try:
            malformed_dataset_path = os.path.join(temp_dir, "malformed.csv")
            
            # Create malformed dataset
            with open(malformed_dataset_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("invalid,csv,format\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("no,proper,columns\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            with pytest.raises(Exception):
                ThreatDetectionDataset(malformed_dataset_path)
        finally:
            shutil.rmtree(temp_dir)


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 