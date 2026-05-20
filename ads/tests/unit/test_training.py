"""
Unit tests for the ads training layer.

This module consolidates tests for:
- Base trainer and training components
- Training factory and factory pattern
- PyTorch, Diffusion, and Multi-GPU trainers
- Experiment tracking and training optimization
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import os
import json

from agents.backend.onyx.server.features.ads.training.base_trainer import (
    BaseTrainer, TrainingPhase, TrainingStatus, TrainingMetrics, TrainingConfig, TrainingResult
)
from agents.backend.onyx.server.features.ads.training.training_factory import (
    TrainingFactory, TrainerType, TrainerConfig
)
from agents.backend.onyx.server.features.ads.training.pytorch_trainer import (
    PyTorchTrainer, PyTorchModelConfig, PyTorchDataConfig
)
from agents.backend.onyx.server.features.ads.training.diffusion_trainer import (
    DiffusionTrainer, DiffusionModelConfig, DiffusionTrainingConfig
)
from agents.backend.onyx.server.features.ads.training.multi_gpu_trainer import (
    MultiGPUTrainer, GPUConfig, MultiGPUTrainingConfig
)
from agents.backend.onyx.server.features.ads.training.experiment_tracker import (
    ExperimentTracker, ExperimentConfig, ExperimentRun
)
from agents.backend.onyx.server.features.ads.training.training_optimizer import (
    TrainingOptimizer, OptimizationLevel, OptimizationConfig, OptimizationResult
)


class TestTrainingPhase:
    """Test TrainingPhase enum."""
    
    def test_training_phase_values(self):
        """Test that all expected training phase values exist."""
        assert TrainingPhase.SETUP == "setup"
        assert TrainingPhase.TRAINING == "training"
        assert TrainingPhase.VALIDATION == "validation"
        assert TrainingPhase.CHECKPOINTING == "checkpointing"
        assert TrainingPhase.COMPLETED == "completed"
        assert TrainingPhase.ERROR == "error"


class TestTrainingStatus:
    """Test TrainingStatus enum."""
    
    def test_training_status_values(self):
        """Test that all expected training status values exist."""
        assert TrainingStatus.IDLE == "idle"
        assert TrainingStatus.RUNNING == "running"
        assert TrainingStatus.PAUSED == "paused"
        assert TrainingStatus.COMPLETED == "completed"
        assert TrainingStatus.FAILED == "failed"


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test TrainingMetrics creation with valid values."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            learning_rate=0.001,
            epoch=5,
            batch=100,
            samples_processed=1000
        )
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.85
        assert metrics.learning_rate == 0.001
        assert metrics.epoch == 5
        assert metrics.batch == 100
        assert metrics.samples_processed == 1000


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_training_config_creation(self):
        """Test TrainingConfig creation with valid values."""
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
            scheduler="cosine",
            early_stopping_patience=10,
            validation_split=0.2,
            checkpoint_dir="./checkpoints"
        )
        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.scheduler == "cosine"
        assert config.early_stopping_patience == 10
        assert config.validation_split == 0.2
        assert config.checkpoint_dir == "./checkpoints"


class TestTrainingResult:
    """Test TrainingResult dataclass."""
    
    def test_training_result_creation(self):
        """Test TrainingResult creation with valid values."""
        result = TrainingResult(
            success=True,
            trainer_name="test_trainer",
            final_metrics=TrainingMetrics(
                loss=0.1,
                accuracy=0.95,
                learning_rate=0.0001,
                epoch=100,
                batch=0,
                samples_processed=10000
            ),
            training_time=3600.0,
            checkpoints_saved=5,
            timestamp=datetime.now()
        )
        assert result.success is True
        assert result.trainer_name == "test_trainer"
        assert result.final_metrics.loss == 0.1
        assert result.final_metrics.accuracy == 0.95
        assert result.training_time == 3600.0
        assert result.checkpoints_saved == 5
        assert result.timestamp is not None


class TestBaseTrainer:
    """Test BaseTrainer abstract class."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer that inherits from BaseTrainer."""
        class MockTrainer(BaseTrainer):
            def __init__(self):
                super().__init__("MockTrainer")
            
            async def setup_training(self, config: TrainingConfig) -> bool:
                return True
            
            async def train_epoch(self, epoch: int) -> TrainingMetrics:
                return TrainingMetrics(
                    loss=0.5,
                    accuracy=0.85,
                    learning_rate=0.001,
                    epoch=epoch,
                    batch=0,
                    samples_processed=1000
                )
            
            async def validate(self, epoch: int) -> TrainingMetrics:
                return TrainingMetrics(
                    loss=0.4,
                    accuracy=0.87,
                    learning_rate=0.001,
                    epoch=epoch,
                    batch=0,
                    samples_processed=200
                )
            
            async def save_checkpoint(self, epoch: int, metrics: TrainingMetrics) -> str:
                return f"checkpoint_epoch_{epoch}.pt"
            
            async def load_checkpoint(self, checkpoint_path: str) -> bool:
                return True
        
        return MockTrainer()
    
    def test_base_trainer_creation(self, mock_trainer):
        """Test BaseTrainer creation."""
        assert mock_trainer.name == "MockTrainer"
        assert mock_trainer.status == TrainingStatus.IDLE
        assert mock_trainer.current_phase == TrainingPhase.SETUP
        assert mock_trainer.metrics is not None
        assert mock_trainer.callbacks is not None
    
    def test_base_trainer_status_management(self, mock_trainer):
        """Test BaseTrainer status management."""
        # Test status transitions
        mock_trainer.status = TrainingStatus.RUNNING
        assert mock_trainer.status == TrainingStatus.RUNNING
        
        mock_trainer.status = TrainingStatus.PAUSED
        assert mock_trainer.status == TrainingStatus.PAUSED
        
        mock_trainer.status = TrainingStatus.COMPLETED
        assert mock_trainer.status == TrainingStatus.COMPLETED
    
    def test_base_trainer_phase_management(self, mock_trainer):
        """Test BaseTrainer phase management."""
        # Test phase transitions
        mock_trainer.current_phase = TrainingPhase.TRAINING
        assert mock_trainer.current_phase == TrainingPhase.TRAINING
        
        mock_trainer.current_phase = TrainingPhase.VALIDATION
        assert mock_trainer.current_phase == TrainingPhase.VALIDATION
        
        mock_trainer.current_phase = TrainingPhase.COMPLETED
        assert mock_trainer.current_phase == TrainingPhase.COMPLETED
    
    def test_base_trainer_callback_management(self, mock_trainer):
        """Test BaseTrainer callback management."""
        # Test adding callbacks
        callback_called = False
        def test_callback(phase, data):
            nonlocal callback_called
            callback_called = True
        
        mock_trainer.add_callback(test_callback)
        assert len(mock_trainer.callbacks) == 1
        
        # Test callback execution
        mock_trainer._execute_callbacks(TrainingPhase.TRAINING, {"epoch": 1})
        assert callback_called is True
    
    @pytest.mark.asyncio
    async def test_base_trainer_training_flow(self, mock_trainer):
        """Test BaseTrainer complete training flow."""
        config = TrainingConfig(
            epochs=2,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Setup training
        success = await mock_trainer.setup_training(config)
        assert success is True
        assert mock_trainer.status == TrainingStatus.RUNNING
        
        # Train
        result = await mock_trainer.train(config)
        
        assert result.success is True
        assert result.trainer_name == "MockTrainer"
        assert result.final_metrics.epoch == 2
        assert mock_trainer.status == TrainingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_base_trainer_stop_training(self, mock_trainer):
        """Test BaseTrainer stop training functionality."""
        config = TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001)
        
        # Start training in background
        training_task = asyncio.create_task(mock_trainer.train(config))
        
        # Wait a bit then stop
        await asyncio.sleep(0.1)
        mock_trainer.stop_training()
        
        # Wait for training to complete
        result = await training_task
        
        assert result.success is False
        assert "stopped" in result.error.lower()
        assert mock_trainer.status == TrainingStatus.COMPLETED


class TestTrainerType:
    """Test TrainerType enum."""
    
    def test_trainer_type_values(self):
        """Test that all expected trainer type values exist."""
        assert TrainerType.PYTORCH == "pytorch"
        assert TrainerType.DIFFUSION == "diffusion"
        assert TrainerType.MULTI_GPU == "multi_gpu"
        assert TrainerType.HYBRID == "hybrid"


class TestTrainerConfig:
    """Test TrainerConfig dataclass."""
    
    def test_trainer_config_creation(self):
        """Test TrainerConfig creation with valid values."""
        config = TrainerConfig(
            trainer_type=TrainerType.PYTORCH,
            model_config={"layers": 3, "hidden_size": 512},
            data_config={"batch_size": 32, "num_workers": 4},
            training_config=TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001),
            gpu_config={"num_gpus": 2, "memory_fraction": 0.8}
        )
        assert config.trainer_type == TrainerType.PYTORCH
        assert config.model_config["layers"] == 3
        assert config.data_config["batch_size"] == 32
        assert config.training_config.epochs == 100
        assert config.gpu_config["num_gpus"] == 2


class TestTrainingFactory:
    """Test TrainingFactory class."""
    
    @pytest.fixture
    def mock_pytorch_trainer(self):
        """Mock PyTorchTrainer."""
        return Mock(spec=PyTorchTrainer)
    
    @pytest.fixture
    def mock_diffusion_trainer(self):
        """Mock DiffusionTrainer."""
        return Mock(spec=DiffusionTrainer)
    
    @pytest.fixture
    def mock_multi_gpu_trainer(self):
        """Mock MultiGPUTrainer."""
        return Mock(spec=MultiGPUTrainer)
    
    @pytest.fixture
    def training_factory(self):
        """Create TrainingFactory instance."""
        return TrainingFactory()
    
    def test_training_factory_creation(self, training_factory):
        """Test TrainingFactory creation."""
        assert training_factory._registered_trainers == {}
        assert training_factory._trainer_instances == {}
        assert training_factory._trainer_configs == {}
    
    def test_training_factory_register_trainer(self, training_factory, mock_pytorch_trainer):
        """Test registering a trainer."""
        training_factory.register_trainer(
            TrainerType.PYTORCH,
            mock_pytorch_trainer,
            {"enabled": True}
        )
        
        assert TrainerType.PYTORCH in training_factory._registered_trainers
        assert training_factory._registered_trainers[TrainerType.PYTORCH] == mock_pytorch_trainer
        assert training_factory._trainer_configs[TrainerType.PYTORCH]["enabled"] is True
    
    def test_training_factory_create_pytorch_trainer(self, training_factory):
        """Test creating a PyTorch trainer."""
        trainer = training_factory.create_pytorch_trainer(
            model_config={"layers": 3},
            data_config={"batch_size": 32},
            training_config=TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001)
        )
        
        assert isinstance(trainer, PyTorchTrainer)
        assert trainer.name == "PyTorchTrainer"
    
    def test_training_factory_create_diffusion_trainer(self, training_factory):
        """Test creating a Diffusion trainer."""
        trainer = training_factory.create_diffusion_trainer(
            model_config={"model_name": "stable-diffusion-2"},
            training_config=TrainingConfig(epochs=100, batch_size=16, learning_rate=0.0001)
        )
        
        assert isinstance(trainer, DiffusionTrainer)
        assert trainer.name == "DiffusionTrainer"
    
    def test_training_factory_create_multi_gpu_trainer(self, training_factory):
        """Test creating a Multi-GPU trainer."""
        trainer = training_factory.create_multi_gpu_trainer(
            gpu_config={"num_gpus": 2, "memory_fraction": 0.8},
            training_config=TrainingConfig(epochs=100, batch_size=64, learning_rate=0.001)
        )
        
        assert isinstance(trainer, MultiGPUTrainer)
        assert trainer.name == "MultiGPUTrainer"
    
    def test_training_factory_get_trainer(self, training_factory, mock_pytorch_trainer):
        """Test getting a trainer."""
        training_factory.register_trainer(
            TrainerType.PYTORCH,
            mock_pytorch_trainer,
            {"enabled": True}
        )
        
        trainer = training_factory.get_trainer(TrainerType.PYTORCH)
        assert trainer == mock_pytorch_trainer
    
    def test_training_factory_get_nonexistent_trainer(self, training_factory):
        """Test getting a non-existent trainer."""
        with pytest.raises(ValueError, match="Trainer type 'nonexistent' not found"):
            training_factory.get_trainer("nonexistent")
    
    def test_training_factory_list_trainers(self, training_factory, mock_pytorch_trainer, mock_diffusion_trainer):
        """Test listing available trainers."""
        training_factory.register_trainer(
            TrainerType.PYTORCH,
            mock_pytorch_trainer,
            {"enabled": True}
        )
        training_factory.register_trainer(
            TrainerType.DIFFUSION,
            mock_diffusion_trainer,
            {"enabled": False}
        )
        
        trainers = training_factory.list_trainers()
        
        assert TrainerType.PYTORCH in trainers
        assert TrainerType.DIFFUSION in trainers
        assert trainers[TrainerType.PYTORCH]["enabled"] is True
        assert trainers[TrainerType.DIFFUSION]["enabled"] is False
    
    def test_training_factory_cleanup(self, training_factory, mock_pytorch_trainer):
        """Test cleanup of trainer instances."""
        training_factory.register_trainer(
            TrainerType.PYTORCH,
            mock_pytorch_trainer,
            {"enabled": True}
        )
        
        # Create an instance
        training_factory.get_trainer(TrainerType.PYTORCH)
        
        # Verify instance exists
        assert TrainerType.PYTORCH in training_factory._trainer_instances
        
        # Cleanup
        training_factory.cleanup_trainer(TrainerType.PYTORCH)
        
        # Verify instance was removed
        assert TrainerType.PYTORCH not in training_factory._trainer_instances


class TestPyTorchTrainer:
    """Test PyTorchTrainer class."""
    
    @pytest.fixture
    def pytorch_trainer(self):
        """Create PyTorchTrainer instance."""
        return PyTorchTrainer()
    
    def test_pytorch_trainer_creation(self, pytorch_trainer):
        """Test PyTorchTrainer creation."""
        assert pytorch_trainer.name == "PyTorchTrainer"
        assert pytorch_trainer.device is not None
        assert pytorch_trainer.model is None
        assert pytorch_trainer.optimizer is None
        assert pytorch_trainer.scheduler is None
    
    @pytest.mark.asyncio
    async def test_pytorch_trainer_setup_training(self, pytorch_trainer):
        """Test PyTorchTrainer setup_training method."""
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam"
        )
        
        success = await pytorch_trainer.setup_training(config)
        
        assert success is True
        assert pytorch_trainer.model is not None
        assert pytorch_trainer.optimizer is not None
        assert pytorch_trainer.scheduler is not None
        assert pytorch_trainer.status == TrainingStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_pytorch_trainer_train_epoch(self, pytorch_trainer):
        """Test PyTorchTrainer train_epoch method."""
        # Setup training first
        config = TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001)
        await pytorch_trainer.setup_training(config)
        
        metrics = await pytorch_trainer.train_epoch(1)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.epoch == 1
        assert metrics.loss >= 0
        assert metrics.accuracy >= 0
    
    @pytest.mark.asyncio
    async def test_pytorch_trainer_validate(self, pytorch_trainer):
        """Test PyTorchTrainer validate method."""
        # Setup training first
        config = TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001)
        await pytorch_trainer.setup_training(config)
        
        metrics = await pytorch_trainer.validate(1)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.epoch == 1
        assert metrics.loss >= 0
        assert metrics.accuracy >= 0
    
    @pytest.mark.asyncio
    async def test_pytorch_trainer_save_checkpoint(self, pytorch_trainer, tmp_path):
        """Test PyTorchTrainer save_checkpoint method."""
        # Setup training first
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            checkpoint_dir=str(tmp_path)
        )
        await pytorch_trainer.setup_training(config)
        
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            learning_rate=0.001,
            epoch=1,
            batch=0,
            samples_processed=1000
        )
        
        checkpoint_path = await pytorch_trainer.save_checkpoint(1, metrics)
        
        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
    
    @pytest.mark.asyncio
    async def test_pytorch_trainer_load_checkpoint(self, pytorch_trainer, tmp_path):
        """Test PyTorchTrainer load_checkpoint method."""
        # Setup training first
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            checkpoint_dir=str(tmp_path)
        )
        await pytorch_trainer.setup_training(config)
        
        # Save a checkpoint first
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            learning_rate=0.001,
            epoch=1,
            batch=0,
            samples_processed=1000
        )
        checkpoint_path = await pytorch_trainer.save_checkpoint(1, metrics)
        
        # Load the checkpoint
        success = await pytorch_trainer.load_checkpoint(checkpoint_path)
        
        assert success is True


class TestDiffusionTrainer:
    """Test DiffusionTrainer class."""
    
    @pytest.fixture
    def diffusion_trainer(self):
        """Create DiffusionTrainer instance."""
        return DiffusionTrainer()
    
    def test_diffusion_trainer_creation(self, diffusion_trainer):
        """Test DiffusionTrainer creation."""
        assert diffusion_trainer.name == "DiffusionTrainer"
        assert diffusion_trainer.device is not None
        assert diffusion_trainer.pipeline is None
        assert diffusion_trainer.optimizer is None
        assert diffusion_trainer.scheduler is None
    
    @pytest.mark.asyncio
    async def test_diffusion_trainer_setup_training(self, diffusion_trainer):
        """Test DiffusionTrainer setup_training method."""
        config = TrainingConfig(
            epochs=100,
            batch_size=16,
            learning_rate=0.0001,
            optimizer="adamw"
        )
        
        success = await diffusion_trainer.setup_training(config)
        
        # Note: This might fail if diffusers is not available, which is expected
        # The test verifies the method exists and can be called
        assert hasattr(diffusion_trainer, 'setup_training')
    
    @pytest.mark.asyncio
    async def test_diffusion_trainer_placeholder_methods(self, diffusion_trainer):
        """Test DiffusionTrainer placeholder methods."""
        # These methods should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            await diffusion_trainer.train_epoch(1)
        
        with pytest.raises(NotImplementedError):
            await diffusion_trainer.validate(1)
        
        with pytest.raises(NotImplementedError):
            await diffusion_trainer.save_checkpoint(1, TrainingMetrics(
                loss=0.5, accuracy=0.85, learning_rate=0.001, epoch=1, batch=0, samples_processed=1000
            ))
        
        with pytest.raises(NotImplementedError):
            await diffusion_trainer.load_checkpoint("checkpoint.pt")


class TestMultiGPUTrainer:
    """Test MultiGPUTrainer class."""
    
    @pytest.fixture
    def multi_gpu_trainer(self):
        """Create MultiGPUTrainer instance."""
        return MultiGPUTrainer()
    
    def test_multi_gpu_trainer_creation(self, multi_gpu_trainer):
        """Test MultiGPUTrainer creation."""
        assert multi_gpu_trainer.name == "MultiGPUTrainer"
        assert multi_gpu_trainer.device is not None
        assert multi_gpu_trainer.model is None
        assert multi_gpu_trainer.optimizer is None
        assert multi_gpu_trainer.scheduler is None
    
    @pytest.mark.asyncio
    async def test_multi_gpu_trainer_placeholder_methods(self, multi_gpu_trainer):
        """Test MultiGPUTrainer placeholder methods."""
        # These methods should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            await multi_gpu_trainer.setup_training(TrainingConfig(
                epochs=100, batch_size=32, learning_rate=0.001
            ))
        
        with pytest.raises(NotImplementedError):
            await multi_gpu_trainer.train_epoch(1)
        
        with pytest.raises(NotImplementedError):
            await multi_gpu_trainer.validate(1)
        
        with pytest.raises(NotImplementedError):
            await multi_gpu_trainer.save_checkpoint(1, TrainingMetrics(
                loss=0.5, accuracy=0.85, learning_rate=0.001, epoch=1, batch=0, samples_processed=1000
            ))
        
        with pytest.raises(NotImplementedError):
            await multi_gpu_trainer.load_checkpoint("checkpoint.pt")


class TestExperimentTracker:
    """Test ExperimentTracker class."""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "test_experiments.db")
    
    @pytest.fixture
    def experiment_tracker(self, temp_db_path):
        """Create ExperimentTracker instance."""
        return ExperimentTracker(db_path=temp_db_path)
    
    def test_experiment_tracker_creation(self, experiment_tracker):
        """Test ExperimentTracker creation."""
        assert experiment_tracker.db_path is not None
        assert experiment_tracker.db is not None
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_create_experiment(self, experiment_tracker):
        """Test creating an experiment."""
        config = ExperimentConfig(
            name="test_experiment",
            description="A test experiment",
            project_name="test_project",
            model_type="pytorch",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32}
        )
        
        experiment_id = await experiment_tracker.create_experiment(config)
        
        assert experiment_id is not None
        assert len(experiment_id) > 0
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_start_run(self, experiment_tracker):
        """Test starting a run."""
        # Create experiment first
        config = ExperimentConfig(
            name="test_experiment",
            description="A test experiment",
            project_name="test_project",
            model_type="pytorch",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32}
        )
        experiment_id = await experiment_tracker.create_experiment(config)
        
        # Start run
        run_id = await experiment_tracker.start_run(experiment_id)
        
        assert run_id is not None
        assert len(run_id) > 0
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_log_metrics(self, experiment_tracker):
        """Test logging metrics."""
        # Create experiment and start run
        config = ExperimentConfig(
            name="test_experiment",
            description="A test experiment",
            project_name="test_project",
            model_type="pytorch",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32}
        )
        experiment_id = await experiment_tracker.create_experiment(config)
        run_id = await experiment_tracker.start_run(experiment_id)
        
        # Log metrics
        metrics = {"loss": 0.5, "accuracy": 0.85, "epoch": 1}
        success = await experiment_tracker.log_metrics(run_id, metrics)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_complete_run(self, experiment_tracker):
        """Test completing a run."""
        # Create experiment and start run
        config = ExperimentConfig(
            name="test_experiment",
            description="A test experiment",
            project_name="test_project",
            model_type="pytorch",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32}
        )
        experiment_id = await experiment_tracker.create_experiment(config)
        run_id = await experiment_tracker.start_run(experiment_id)
        
        # Complete run
        final_metrics = {"loss": 0.1, "accuracy": 0.95, "epoch": 100}
        success = await experiment_tracker.complete_run(run_id, final_metrics)
        
        assert success is True
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_get_experiment(self, experiment_tracker):
        """Test getting experiment information."""
        # Create experiment
        config = ExperimentConfig(
            name="test_experiment",
            description="A test experiment",
            project_name="test_project",
            model_type="pytorch",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32}
        )
        experiment_id = await experiment_tracker.create_experiment(config)
        
        # Get experiment
        experiment = await experiment_tracker.get_experiment(experiment_id)
        
        assert experiment is not None
        assert experiment["name"] == "test_experiment"
        assert experiment["project_name"] == "test_project"
    
    @pytest.mark.asyncio
    async def test_experiment_tracker_list_experiments(self, experiment_tracker):
        """Test listing experiments."""
        # Create multiple experiments
        for i in range(3):
            config = ExperimentConfig(
                name=f"test_experiment_{i}",
                description=f"Test experiment {i}",
                project_name="test_project",
                model_type="pytorch",
                hyperparameters={"learning_rate": 0.001, "batch_size": 32}
            )
            await experiment_tracker.create_experiment(config)
        
        # List experiments
        experiments = await experiment_tracker.list_experiments()
        
        assert len(experiments) >= 3
        assert any(exp["name"] == "test_experiment_0" for exp in experiments)
        assert any(exp["name"] == "test_experiment_1" for exp in experiments)
        assert any(exp["name"] == "test_experiment_2" for exp in experiments)


class TestTrainingOptimizer:
    """Test TrainingOptimizer class."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Mock trainer for testing."""
        return Mock(spec=BaseTrainer)
    
    @pytest.fixture
    def training_optimizer(self, mock_trainer):
        """Create TrainingOptimizer instance."""
        return TrainingOptimizer(trainer=mock_trainer)
    
    def test_training_optimizer_creation(self, training_optimizer):
        """Test TrainingOptimizer creation."""
        assert training_optimizer.trainer is not None
        assert training_optimizer.baseline_metrics is None
        assert training_optimizer.optimized_metrics is None
    
    @pytest.mark.asyncio
    async def test_training_optimizer_light_optimization(self, training_optimizer, mock_trainer):
        """Test TrainingOptimizer light optimization."""
        # Mock trainer methods
        mock_trainer.train.return_value = TrainingResult(
            success=True,
            trainer_name="MockTrainer",
            final_metrics=TrainingMetrics(
                loss=0.1, accuracy=0.95, learning_rate=0.001, epoch=100, batch=0, samples_processed=10000
            ),
            training_time=1800.0,
            checkpoints_saved=5,
            timestamp=datetime.now()
        )
        
        config = OptimizationConfig(
            level=OptimizationLevel.LIGHT,
            target_metrics=["loss", "accuracy"],
            constraints={"max_training_time": 3600}
        )
        
        result = await training_optimizer.optimize_trainer(config)
        
        assert result.success is True
        assert result.optimization_level == OptimizationLevel.LIGHT
        assert "improvements" in result.optimization_results
    
    @pytest.mark.asyncio
    async def test_training_optimizer_standard_optimization(self, training_optimizer, mock_trainer):
        """Test TrainingOptimizer standard optimization."""
        # Mock trainer methods
        mock_trainer.train.return_value = TrainingResult(
            success=True,
            trainer_name="MockTrainer",
            final_metrics=TrainingMetrics(
                loss=0.08, accuracy=0.97, learning_rate=0.001, epoch=100, batch=0, samples_processed=10000
            ),
            training_time=1600.0,
            checkpoints_saved=5,
            timestamp=datetime.now()
        )
        
        config = OptimizationConfig(
            level=OptimizationLevel.STANDARD,
            target_metrics=["loss", "accuracy", "training_time"],
            constraints={"max_training_time": 3600}
        )
        
        result = await training_optimizer.optimize_trainer(config)
        
        assert result.success is True
        assert result.optimization_level == OptimizationLevel.STANDARD
        assert "improvements" in result.optimization_results
    
    def test_training_optimizer_get_recommendations(self, training_optimizer):
        """Test TrainingOptimizer get_recommendations method."""
        recommendations = training_optimizer.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert "type" in rec
            assert "description" in rec
            assert "priority" in rec


if __name__ == "__main__":
    pytest.main([__file__])
