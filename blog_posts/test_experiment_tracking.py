from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from experiment_tracking import (
            import time
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Suite for Experiment Tracking and Model Checkpointing System
================================================================

This module provides comprehensive tests for the experiment tracking and
model checkpointing functionality, including unit tests, integration tests,
and performance tests.
"""


# Import the modules to test
    ExperimentMetadata,
    CheckpointMetadata,
    BaseTracker,
    WandBTracker,
    TensorBoardTracker,
    MLflowTracker,
    ModelCheckpointer,
    ExperimentTracker,
    experiment_tracking
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "description": "Test experiment",
        "tags": ["test", "unit"],
        "tracking": {
            "use_wandb": False,
            "use_tensorboard": True,
            "use_mlflow": False
        },
        "checkpoint_dir": "test_checkpoints",
        "max_checkpoints": 3,
        "model": {
            "type": "transformer",
            "name": "test-model"
        },
        "training": {
            "epochs": 5,
            "learning_rate": 1e-4
        }
    }


@pytest.fixture
def sample_model():
    """Sample PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )


@pytest.fixture
def sample_optimizer(sample_model) -> Any:
    """Sample optimizer for testing."""
    return torch.optim.Adam(sample_model.parameters())


@pytest.fixture
def sample_scheduler(sample_optimizer) -> Any:
    """Sample scheduler for testing."""
    return torch.optim.lr_scheduler.StepLR(sample_optimizer, step_size=1)


# =============================================================================
# EXPERIMENT METADATA TESTS
# =============================================================================

class TestExperimentMetadata:
    """Test ExperimentMetadata class."""
    
    def test_initialization(self) -> Any:
        """Test metadata initialization."""
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            experiment_id="test_123",
            description="Test description",
            tags=["test", "unit"]
        )
        
        assert metadata.experiment_name == "test_experiment"
        assert metadata.experiment_id == "test_123"
        assert metadata.description == "Test description"
        assert metadata.tags == ["test", "unit"]
        assert metadata.status == "running"
        assert metadata.start_time is not None
        assert metadata.end_time is None
    
    def test_generate_id(self) -> Any:
        """Test experiment ID generation."""
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            experiment_id=""
        )
        
        generated_id = metadata.generate_id()
        assert isinstance(generated_id, str)
        assert len(generated_id) > 0
        assert "_" in generated_id
    
    def test_to_dict(self) -> Any:
        """Test conversion to dictionary."""
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            experiment_id="test_123"
        )
        
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["experiment_name"] == "test_experiment"
        assert metadata_dict["experiment_id"] == "test_123"
    
    def test_mark_completed(self) -> Any:
        """Test marking experiment as completed."""
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            experiment_id="test_123"
        )
        
        metadata.mark_completed()
        assert metadata.status == "completed"
        assert metadata.end_time is not None
    
    def test_mark_failed(self) -> Any:
        """Test marking experiment as failed."""
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            experiment_id="test_123"
        )
        
        metadata.mark_failed("Test error")
        assert metadata.status == "failed"
        assert metadata.end_time is not None


# =============================================================================
# CHECKPOINT METADATA TESTS
# =============================================================================

class TestCheckpointMetadata:
    """Test CheckpointMetadata class."""
    
    def test_initialization(self, sample_model) -> Any:
        """Test checkpoint metadata initialization."""
        metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            experiment_id="test_123",
            epoch=1,
            step=100,
            timestamp=datetime.now(),
            model_state_dict=sample_model.state_dict()
        )
        
        assert metadata.checkpoint_id == "test_checkpoint"
        assert metadata.experiment_id == "test_123"
        assert metadata.epoch == 1
        assert metadata.step == 100
        assert metadata.model_state_dict is not None
    
    def test_generate_id(self) -> Any:
        """Test checkpoint ID generation."""
        metadata = CheckpointMetadata(
            checkpoint_id="",
            experiment_id="test_123",
            epoch=1,
            step=100,
            timestamp=datetime.now(),
            model_state_dict={}
        )
        
        generated_id = metadata.generate_id()
        assert isinstance(generated_id, str)
        assert "checkpoint" in generated_id
        assert "test_123" in generated_id
        assert "epoch_1" in generated_id
        assert "step_100" in generated_id
    
    def test_to_dict(self, sample_model) -> Any:
        """Test conversion to dictionary."""
        metadata = CheckpointMetadata(
            checkpoint_id="test_checkpoint",
            experiment_id="test_123",
            epoch=1,
            step=100,
            timestamp=datetime.now(),
            model_state_dict=sample_model.state_dict()
        )
        
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["checkpoint_id"] == "test_checkpoint"
        assert metadata_dict["experiment_id"] == "test_123"


# =============================================================================
# BASE TRACKER TESTS
# =============================================================================

class TestBaseTracker:
    """Test BaseTracker class."""
    
    def test_initialization(self) -> Any:
        """Test base tracker initialization."""
        config = {"test": "config"}
        tracker = BaseTracker("test_experiment", config)
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.config == config
    
    def test_abstract_methods(self) -> Any:
        """Test that abstract methods raise NotImplementedError."""
        tracker = BaseTracker("test_experiment", {})
        
        with pytest.raises(NotImplementedError):
            tracker.log_metrics({"test": 1.0})
        
        with pytest.raises(NotImplementedError):
            tracker.log_hyperparameters({"test": "value"})
        
        with pytest.raises(NotImplementedError):
            tracker.log_model("test_path")
        
        with pytest.raises(NotImplementedError):
            tracker.log_artifact("test_path")
        
        with pytest.raises(NotImplementedError):
            tracker.log_image("test_path", "test_image")
        
        with pytest.raises(NotImplementedError):
            tracker.log_text("test_text", "test_name")
        
        with pytest.raises(NotImplementedError):
            tracker.finish()


# =============================================================================
# TENSORBOARD TRACKER TESTS
# =============================================================================

class TestTensorBoardTracker:
    """Test TensorBoardTracker class."""
    
    def test_initialization(self, temp_dir, sample_config) -> Any:
        """Test TensorBoard tracker initialization."""
        sample_config["tensorboard_log_dir"] = temp_dir
        
        tracker = TensorBoardTracker("test_experiment", sample_config)
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.log_dir.exists()
        assert tracker.writer is not None
    
    def test_log_metrics(self, temp_dir, sample_config) -> Any:
        """Test logging metrics to TensorBoard."""
        sample_config["tensorboard_log_dir"] = temp_dir
        tracker = TensorBoardTracker("test_experiment", sample_config)
        
        metrics = {"loss": 0.5, "accuracy": 0.8}
        tracker.log_metrics(metrics, step=1)
        
        # Check that log directory contains files
        log_files = list(tracker.log_dir.glob("*"))
        assert len(log_files) > 0
    
    def test_log_hyperparameters(self, temp_dir, sample_config) -> Any:
        """Test logging hyperparameters to TensorBoard."""
        sample_config["tensorboard_log_dir"] = temp_dir
        tracker = TensorBoardTracker("test_experiment", sample_config)
        
        hyperparameters = {"learning_rate": 1e-4, "batch_size": 32}
        tracker.log_hyperparameters(hyperparameters)
        
        # Check that log directory contains files
        log_files = list(tracker.log_dir.glob("*"))
        assert len(log_files) > 0
    
    def test_log_text(self, temp_dir, sample_config) -> Any:
        """Test logging text to TensorBoard."""
        sample_config["tensorboard_log_dir"] = temp_dir
        tracker = TensorBoardTracker("test_experiment", sample_config)
        
        tracker.log_text("Test text", "test_name")
        
        # Check that log directory contains files
        log_files = list(tracker.log_dir.glob("*"))
        assert len(log_files) > 0
    
    def test_finish(self, temp_dir, sample_config) -> Any:
        """Test finishing TensorBoard experiment."""
        sample_config["tensorboard_log_dir"] = temp_dir
        tracker = TensorBoardTracker("test_experiment", sample_config)
        
        tracker.finish()
        # Should not raise any exceptions


# =============================================================================
# MODEL CHECKPOINTER TESTS
# =============================================================================

class TestModelCheckpointer:
    """Test ModelCheckpointer class."""
    
    def test_initialization(self, temp_dir) -> Any:
        """Test checkpointer initialization."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir), max_checkpoints=5)
        
        assert checkpointer.checkpoint_dir == checkpoint_dir
        assert checkpointer.max_checkpoints == 5
        assert checkpoint_dir.exists()
        assert checkpointer.registry_file.exists()
    
    def test_save_checkpoint(self, temp_dir, sample_model, sample_optimizer) -> Any:
        """Test saving model checkpoint."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir))
        
        checkpoint_path = checkpointer.save_checkpoint(
            experiment_id="test_123",
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            train_loss=0.5,
            val_loss=0.6,
            train_accuracy=0.8,
            val_accuracy=0.75,
            is_best=False
        )
        
        assert Path(checkpoint_path).exists()
        assert checkpointer.registry_file.exists()
        
        # Check registry
        with open(checkpointer.registry_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            registry = json.load(f)
        
        assert "test_123" in registry["experiments"]
        assert len(registry["experiments"]["test_123"]) == 1
    
    def test_load_checkpoint(self, temp_dir, sample_model, sample_optimizer) -> Any:
        """Test loading model checkpoint."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir))
        
        # Save checkpoint first
        checkpoint_path = checkpointer.save_checkpoint(
            experiment_id="test_123",
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            train_loss=0.5,
            val_loss=0.6
        )
        
        # Create new model and optimizer
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        metadata, epoch, step = checkpointer.load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        assert epoch == 1
        assert step == 100
        assert metadata.experiment_id == "test_123"
    
    def test_get_best_checkpoint(self, temp_dir, sample_model, sample_optimizer) -> Optional[Dict[str, Any]]:
        """Test getting best checkpoint."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir))
        
        # Save multiple checkpoints with different validation losses
        checkpointer.save_checkpoint(
            experiment_id="test_123",
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            val_loss=0.8
        )
        
        checkpointer.save_checkpoint(
            experiment_id="test_123",
            model=sample_model,
            epoch=2,
            step=200,
            optimizer=sample_optimizer,
            val_loss=0.5  # Best loss
        )
        
        checkpointer.save_checkpoint(
            experiment_id="test_123",
            model=sample_model,
            epoch=3,
            step=300,
            optimizer=sample_optimizer,
            val_loss=0.9
        )
        
        best_checkpoint = checkpointer.get_best_checkpoint("test_123")
        assert best_checkpoint is not None
        
        # Load best checkpoint and verify it has the lowest validation loss
        metadata, _, _ = checkpointer.load_checkpoint(best_checkpoint, sample_model)
        assert metadata.val_loss == 0.5
    
    def test_cleanup_old_checkpoints(self, temp_dir, sample_model, sample_optimizer) -> Any:
        """Test cleanup of old checkpoints."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir), max_checkpoints=2)
        
        # Save more checkpoints than allowed
        for i in range(4):
            checkpointer.save_checkpoint(
                experiment_id="test_123",
                model=sample_model,
                epoch=i,
                step=i * 100,
                optimizer=sample_optimizer,
                val_loss=0.5 + i * 0.1
            )
        
        # Check that only max_checkpoints remain
        checkpoints = checkpointer.list_checkpoints("test_123")
        assert len(checkpoints) <= 2
    
    def test_list_checkpoints(self, temp_dir, sample_model, sample_optimizer) -> List[Any]:
        """Test listing checkpoints."""
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        checkpointer = ModelCheckpointer(str(checkpoint_dir))
        
        # Save multiple checkpoints
        for i in range(3):
            checkpointer.save_checkpoint(
                experiment_id="test_123",
                model=sample_model,
                epoch=i,
                step=i * 100,
                optimizer=sample_optimizer,
                val_loss=0.5 + i * 0.1
            )
        
        checkpoints = checkpointer.list_checkpoints("test_123")
        assert len(checkpoints) == 3
        
        # Check that checkpoints are sorted by timestamp
        timestamps = [cp["timestamp"] for cp in checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)


# =============================================================================
# EXPERIMENT TRACKER TESTS
# =============================================================================

class TestExperimentTracker:
    """Test ExperimentTracker class."""
    
    def test_initialization(self, sample_config) -> Any:
        """Test experiment tracker initialization."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.metadata.experiment_name == "test_experiment"
        assert tracker.metadata.status == "running"
        assert tracker.checkpointer is not None
        assert len(tracker.trackers) > 0
    
    def test_log_metrics(self, sample_config) -> Any:
        """Test logging metrics."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        metrics = {"loss": 0.5, "accuracy": 0.8}
        tracker.log_metrics(metrics, step=1)
        
        # Check that metrics are stored in history
        assert len(tracker.metrics_history["train_loss"]) == 0  # No train_loss in metrics
        assert len(tracker.metrics_history["val_loss"]) == 0    # No val_loss in metrics
    
    def test_save_checkpoint(self, sample_config, sample_model, sample_optimizer) -> Any:
        """Test saving checkpoint through tracker."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        checkpoint_path = tracker.save_checkpoint(
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            train_loss=0.5,
            val_loss=0.6,
            is_best=False
        )
        
        assert Path(checkpoint_path).exists()
    
    def test_load_checkpoint(self, sample_config, sample_model, sample_optimizer) -> Any:
        """Test loading checkpoint through tracker."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        # Save checkpoint first
        checkpoint_path = tracker.save_checkpoint(
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            train_loss=0.5,
            val_loss=0.6
        )
        
        # Create new model and optimizer
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        epoch, step = tracker.load_checkpoint(checkpoint_path, new_model, new_optimizer)
        
        assert epoch == 1
        assert step == 100
    
    def test_get_best_checkpoint(self, sample_config, sample_model, sample_optimizer) -> Optional[Dict[str, Any]]:
        """Test getting best checkpoint through tracker."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        # Save multiple checkpoints
        tracker.save_checkpoint(
            model=sample_model,
            epoch=1,
            step=100,
            optimizer=sample_optimizer,
            val_loss=0.8
        )
        
        tracker.save_checkpoint(
            model=sample_model,
            epoch=2,
            step=200,
            optimizer=sample_optimizer,
            val_loss=0.5  # Best loss
        )
        
        best_checkpoint = tracker.get_best_checkpoint()
        assert best_checkpoint is not None
    
    def test_create_performance_plots(self, sample_config) -> Any:
        """Test creating performance plots."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        # Add some metrics to history
        tracker.metrics_history["train_loss"] = [0.8, 0.6, 0.4]
        tracker.metrics_history["val_loss"] = [0.9, 0.7, 0.5]
        tracker.metrics_history["train_accuracy"] = [0.6, 0.7, 0.8]
        tracker.metrics_history["val_accuracy"] = [0.5, 0.6, 0.7]
        
        # Create plots
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker.create_performance_plots(temp_dir)
            
            # Check that plot file was created
            plot_files = list(Path(temp_dir).glob("*.png"))
            assert len(plot_files) > 0
    
    def test_finish(self, sample_config) -> Any:
        """Test finishing experiment."""
        tracker = ExperimentTracker("test_experiment", sample_config)
        
        # Add some metrics
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.4}, step=2)
        
        tracker.finish("completed")
        
        assert tracker.metadata.status == "completed"
        assert tracker.metadata.end_time is not None


# =============================================================================
# CONTEXT MANAGER TESTS
# =============================================================================

class TestExperimentTrackingContextManager:
    """Test experiment_tracking context manager."""
    
    def test_context_manager_success(self, sample_config, sample_model, sample_optimizer) -> Any:
        """Test successful experiment tracking with context manager."""
        with experiment_tracking("test_experiment", sample_config) as tracker:
            # Log some metrics
            tracker.log_metrics({"loss": 0.5}, step=1)
            
            # Save checkpoint
            tracker.save_checkpoint(
                model=sample_model,
                epoch=1,
                step=100,
                optimizer=sample_optimizer,
                train_loss=0.5,
                val_loss=0.6
            )
        
        # Check that experiment was marked as completed
        # (This would require accessing the tracker after context exit,
        # which is not possible with the current implementation)
    
    def test_context_manager_exception(self, sample_config) -> Any:
        """Test experiment tracking with exception."""
        with pytest.raises(ValueError):
            with experiment_tracking("test_experiment", sample_config) as tracker:
                tracker.log_metrics({"loss": 0.5}, step=1)
                raise ValueError("Test exception")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the experiment tracking system."""
    
    def test_full_training_workflow(self, temp_dir, sample_config, sample_model, sample_optimizer) -> Any:
        """Test complete training workflow with experiment tracking."""
        sample_config["checkpoint_dir"] = temp_dir
        
        with experiment_tracking("integration_test", sample_config) as tracker:
            # Simulate training loop
            for epoch in range(3):
                for step in range(5):
                    # Simulate training
                    train_loss = 1.0 - (epoch * 5 + step) * 0.05
                    val_loss = train_loss + 0.1
                    train_acc = 0.5 + (epoch * 5 + step) * 0.02
                    val_acc = train_acc - 0.05
                    
                    # Log metrics
                    tracker.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc
                    }, step=epoch * 5 + step)
                    
                    # Save checkpoint every 5 steps
                    if step % 5 == 0:
                        tracker.save_checkpoint(
                            model=sample_model,
                            epoch=epoch,
                            step=epoch * 5 + step,
                            optimizer=sample_optimizer,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            train_accuracy=train_acc,
                            val_accuracy=val_acc,
                            is_best=(val_loss < 0.5)
                        )
            
            # Test loading best checkpoint
            best_checkpoint = tracker.get_best_checkpoint()
            if best_checkpoint:
                new_model = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 1)
                )
                new_optimizer = torch.optim.Adam(new_model.parameters())
                
                epoch, step = tracker.load_checkpoint(best_checkpoint, new_model, new_optimizer)
                assert epoch >= 0
                assert step >= 0
    
    def test_multiple_experiments(self, temp_dir, sample_config, sample_model, sample_optimizer) -> Any:
        """Test running multiple experiments."""
        sample_config["checkpoint_dir"] = temp_dir
        
        experiments = ["exp_1", "exp_2", "exp_3"]
        
        for exp_name in experiments:
            with experiment_tracking(exp_name, sample_config) as tracker:
                # Simulate training
                for step in range(3):
                    tracker.log_metrics({"loss": 0.5 - step * 0.1}, step=step)
                    
                    tracker.save_checkpoint(
                        model=sample_model,
                        epoch=0,
                        step=step,
                        optimizer=sample_optimizer,
                        train_loss=0.5 - step * 0.1,
                        val_loss=0.6 - step * 0.1
                    )
        
        # Check that all experiments created checkpoints
        checkpoint_dir = Path(temp_dir)
        assert checkpoint_dir.exists()
        
        # Check registry
        registry_file = checkpoint_dir / "checkpoint_registry.json"
        assert registry_file.exists()
        
        with open(registry_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            registry = json.load(f)
        
        # Should have experiments for all three experiments
        assert len(registry["experiments"]) >= 3


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for the experiment tracking system."""
    
    def test_large_model_checkpointing(self, temp_dir, sample_config) -> Any:
        """Test checkpointing with large models."""
        # Create a larger model
        large_model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        optimizer = torch.optim.Adam(large_model.parameters())
        
        sample_config["checkpoint_dir"] = temp_dir
        
        with experiment_tracking("large_model_test", sample_config) as tracker:
            
            start_time = time.time()
            
            # Save multiple checkpoints
            for i in range(5):
                tracker.save_checkpoint(
                    model=large_model,
                    epoch=i,
                    step=i * 100,
                    optimizer=optimizer,
                    train_loss=0.5,
                    val_loss=0.6
                )
            
            end_time = time.time()
            
            # Checkpointing should be reasonably fast
            assert end_time - start_time < 10.0  # Should complete within 10 seconds
    
    def test_high_frequency_logging(self, sample_config) -> Any:
        """Test high-frequency metric logging."""
        tracker = ExperimentTracker("high_freq_test", sample_config)
        
        
        start_time = time.time()
        
        # Log metrics at high frequency
        for i in range(1000):
            tracker.log_metrics({
                "loss": 0.5 + np.random.normal(0, 0.01),
                "accuracy": 0.8 + np.random.normal(0, 0.01)
            }, step=i)
        
        end_time = time.time()
        
        # High-frequency logging should be fast
        assert end_time - start_time < 5.0  # Should complete within 5 seconds


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in the experiment tracking system."""
    
    def test_invalid_checkpoint_path(self, sample_config, sample_model, sample_optimizer) -> Any:
        """Test loading non-existent checkpoint."""
        tracker = ExperimentTracker("error_test", sample_config)
        
        with pytest.raises(FileNotFoundError):
            tracker.load_checkpoint("non_existent_path.pt", sample_model, sample_optimizer)
    
    def test_corrupted_checkpoint(self, temp_dir, sample_config, sample_model, sample_optimizer) -> Any:
        """Test loading corrupted checkpoint file."""
        tracker = ExperimentTracker("error_test", sample_config)
        
        # Create a corrupted checkpoint file
        corrupted_path = Path(temp_dir) / "corrupted.pt"
        with open(corrupted_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("This is not a valid checkpoint file")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        with pytest.raises(Exception):  # Should raise some exception
            tracker.load_checkpoint(str(corrupted_path), sample_model, sample_optimizer)
    
    def test_tracker_initialization_failure(self) -> Any:
        """Test handling of tracker initialization failures."""
        config = {
            "tracking": {
                "use_wandb": True,  # This will fail if wandb is not available
                "use_tensorboard": False,
                "use_mlflow": False
            }
        }
        
        # Should not raise exception, just log warning
        tracker = ExperimentTracker("tracker_failure_test", config)
        assert len(tracker.trackers) == 0  # No trackers should be initialized


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 