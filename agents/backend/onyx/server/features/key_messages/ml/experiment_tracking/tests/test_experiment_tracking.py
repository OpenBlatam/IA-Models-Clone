from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ..tracker import (
from ..checkpointing import (
from ..metrics import (
        import shutil
        import shutil
        import shutil
        import shutil
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for Experiment Tracking System
"""


    ExperimentTracker,
    NoOpTracker,
    TensorBoardTracker,
    WandbTracker,
    MLflowTracker,
    CompositeTracker
)
    CheckpointManager,
    ModelCheckpoint,
    TrainingCheckpoint,
    CheckpointStrategy
)
    MetricsTracker,
    MetricLogger,
    MetricAggregator,
    TrainingMetricsTracker
)

class TestExperimentTracker:
    """Test base experiment tracker."""
    
    def test_experiment_tracker_initialization(self) -> Any:
        """Test experiment tracker initialization."""
        tracker = NoOpTracker()
        
        assert tracker.experiment_name is None
        assert tracker.run_id is None
        assert tracker.config is None
        assert tracker.is_initialized is False
    
    def test_noop_tracker_initialization(self) -> Any:
        """Test NoOpTracker initialization."""
        tracker = NoOpTracker()
        
        run_id = tracker.init_experiment("test_experiment", {"param": "value"})
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.config == {"param": "value"}
        assert tracker.is_initialized is True
        assert run_id.startswith("noop_")
    
    def test_noop_tracker_logging(self) -> Any:
        """Test NoOpTracker logging methods."""
        tracker = NoOpTracker()
        tracker.init_experiment("test_experiment")
        
        # Test logging methods (should not raise exceptions)
        tracker.log_metrics({"loss": 0.5}, step=100)
        tracker.log_config({"param": "value"})
        tracker.log_model("model.pt", "test_model")
        tracker.finalize_experiment()
        
        assert tracker.is_initialized is False

class TestTensorBoardTracker:
    """Test TensorBoard tracker."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_tensorboard_tracker_initialization(self, mock_writer_class) -> Any:
        """Test TensorBoard tracker initialization."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        tracker = TensorBoardTracker(log_dir=self.temp_dir)
        run_id = tracker.init_experiment("test_experiment", {"param": "value"})
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.config == {"param": "value"}
        assert tracker.is_initialized is True
        assert run_id.startswith("tb_")
        assert mock_writer_class.called
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_tensorboard_tracker_logging(self, mock_writer_class) -> Any:
        """Test TensorBoard tracker logging."""
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        tracker = TensorBoardTracker(log_dir=self.temp_dir)
        tracker.init_experiment("test_experiment")
        
        # Test logging methods
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
        tracker.log_config({"param": "value"})
        tracker.log_model("model.pt", "test_model")
        tracker.finalize_experiment()
        
        # Verify writer methods were called
        assert mock_writer.add_scalar.called
        assert mock_writer.add_text.called
        assert mock_writer.flush.called
        assert mock_writer.close.called
    
    def test_tensorboard_tracker_fallback(self) -> Any:
        """Test TensorBoard tracker fallback when not available."""
        with patch('torch.utils.tensorboard.SummaryWriter', side_effect=ImportError):
            tracker = TensorBoardTracker(log_dir=self.temp_dir)
            run_id = tracker.init_experiment("test_experiment")
            
            # Should fall back to NoOpTracker
            assert run_id.startswith("noop_")

class TestWandbTracker:
    """Test Weights & Biases tracker."""
    
    @patch('wandb.init')
    def test_wandb_tracker_initialization(self, mock_wandb_init) -> Any:
        """Test W&B tracker initialization."""
        mock_run = Mock()
        mock_run.id = "test_run_id"
        mock_wandb_init.return_value = mock_run
        
        tracker = WandbTracker(project="test_project", entity="test_entity")
        run_id = tracker.init_experiment("test_experiment", {"param": "value"})
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.config == {"param": "value"}
        assert tracker.is_initialized is True
        assert run_id == "test_run_id"
        assert mock_wandb_init.called
    
    @patch('wandb.init')
    def test_wandb_tracker_logging(self, mock_wandb_init) -> Any:
        """Test W&B tracker logging."""
        mock_run = Mock()
        mock_run.id = "test_run_id"
        mock_wandb_init.return_value = mock_run
        
        tracker = WandbTracker(project="test_project")
        tracker.init_experiment("test_experiment")
        
        # Test logging methods
        tracker.log_metrics({"loss": 0.5}, step=100)
        tracker.log_config({"param": "value"})
        tracker.log_model("model.pt", "test_model")
        tracker.finalize_experiment()
        
        # Verify run methods were called
        assert mock_run.log.called
        assert mock_run.config.update.called
        assert mock_run.finish.called
    
    def test_wandb_tracker_fallback(self) -> Any:
        """Test W&B tracker fallback when not available."""
        with patch('wandb.init', side_effect=ImportError):
            tracker = WandbTracker(project="test_project")
            run_id = tracker.init_experiment("test_experiment")
            
            # Should fall back to NoOpTracker
            assert run_id.startswith("noop_")

class TestMLflowTracker:
    """Test MLflow tracker."""
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('mlflow.set_tracking_uri')
    def test_mlflow_tracker_initialization(self, mock_set_uri, mock_set_exp, mock_start_run) -> Any:
        """Test MLflow tracker initialization."""
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run
        
        tracker = MLflowTracker(tracking_uri="sqlite:///test.db")
        run_id = tracker.init_experiment("test_experiment", {"param": "value"})
        
        assert tracker.experiment_name == "test_experiment"
        assert tracker.config == {"param": "value"}
        assert tracker.is_initialized is True
        assert run_id == "test_run_id"
        assert mock_set_uri.called
        assert mock_set_exp.called
        assert mock_start_run.called
    
    @patch('mlflow.start_run')
    @patch('mlflow.set_experiment')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.end_run')
    def test_mlflow_tracker_logging(self, mock_end_run, mock_log_param, mock_log_metric, 
                                   mock_set_uri, mock_set_exp, mock_start_run) -> Any:
        """Test MLflow tracker logging."""
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value = mock_run
        
        tracker = MLflowTracker()
        tracker.init_experiment("test_experiment")
        
        # Test logging methods
        tracker.log_metrics({"loss": 0.5}, step=100)
        tracker.log_config({"param": "value"})
        tracker.log_model("model.pt", "test_model")
        tracker.finalize_experiment()
        
        # Verify MLflow methods were called
        assert mock_log_metric.called
        assert mock_log_param.called
        assert mock_end_run.called
    
    def test_mlflow_tracker_fallback(self) -> Any:
        """Test MLflow tracker fallback when not available."""
        with patch('mlflow.start_run', side_effect=ImportError):
            tracker = MLflowTracker()
            run_id = tracker.init_experiment("test_experiment")
            
            # Should fall back to NoOpTracker
            assert run_id.startswith("noop_")

class TestCompositeTracker:
    """Test composite tracker."""
    
    def test_composite_tracker_initialization(self) -> Any:
        """Test composite tracker initialization."""
        trackers = [NoOpTracker(), NoOpTracker()]
        composite = CompositeTracker(trackers)
        
        run_id = composite.init_experiment("test_experiment", {"param": "value"})
        
        assert composite.experiment_name == "test_experiment"
        assert composite.config == {"param": "value"}
        assert composite.is_initialized is True
        assert run_id.startswith("noop_")
    
    def test_composite_tracker_logging(self) -> Any:
        """Test composite tracker logging."""
        trackers = [NoOpTracker(), NoOpTracker()]
        composite = CompositeTracker(trackers)
        composite.init_experiment("test_experiment")
        
        # Test logging methods
        composite.log_metrics({"loss": 0.5}, step=100)
        composite.log_config({"param": "value"})
        composite.log_model("model.pt", "test_model")
        composite.finalize_experiment()
        
        # All trackers should be finalized
        for tracker in trackers:
            assert tracker.is_initialized is False

class TestCheckpointStrategy:
    """Test checkpoint strategy."""
    
    def test_checkpoint_strategy_initialization(self) -> Any:
        """Test checkpoint strategy initialization."""
        strategy = CheckpointStrategy(
            save_steps=1000,
            save_total_limit=3,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )
        
        assert strategy.save_steps == 1000
        assert strategy.save_total_limit == 3
        assert strategy.save_best_only is True
        assert strategy.monitor == "val_loss"
        assert strategy.mode == "min"
    
    def test_checkpoint_strategy_validation(self) -> Any:
        """Test checkpoint strategy validation."""
        # Test invalid mode
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            CheckpointStrategy(mode="invalid")
        
        # Test invalid save_steps
        with pytest.raises(ValueError, match="save_steps must be positive"):
            CheckpointStrategy(save_steps=0)
        
        # Test invalid save_total_limit
        with pytest.raises(ValueError, match="save_total_limit must be positive"):
            CheckpointStrategy(save_total_limit=0)

class TestModelCheckpoint:
    """Test model checkpoint."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_checkpoint_creation(self) -> Any:
        """Test model checkpoint creation."""
        model_state = {"layer1.weight": torch.randn(10, 10)}
        model_config = {"model_name": "test_model"}
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        checkpoint = ModelCheckpoint(
            model_state=model_state,
            model_config=model_config,
            model_path="test_path",
            timestamp=time.time(),
            step=100,
            epoch=5,
            metrics=metrics
        )
        
        assert checkpoint.model_state == model_state
        assert checkpoint.model_config == model_config
        assert checkpoint.step == 100
        assert checkpoint.epoch == 5
        assert checkpoint.metrics == metrics
    
    def test_model_checkpoint_save_load(self) -> Any:
        """Test model checkpoint save and load."""
        model_state = {"layer1.weight": torch.randn(10, 10)}
        model_config = {"model_name": "test_model"}
        metrics = {"loss": 0.5}
        
        checkpoint = ModelCheckpoint(
            model_state=model_state,
            model_config=model_config,
            model_path="test_path",
            timestamp=time.time(),
            step=100,
            epoch=5,
            metrics=metrics
        )
        
        # Save checkpoint
        save_path = os.path.join(self.temp_dir, "test_checkpoint")
        checkpoint.save(save_path)
        
        # Load checkpoint
        loaded_checkpoint = ModelCheckpoint.load(save_path)
        
        # Verify loaded checkpoint
        assert loaded_checkpoint.model_config == model_config
        assert loaded_checkpoint.step == 100
        assert loaded_checkpoint.epoch == 5
        assert loaded_checkpoint.metrics == metrics
        
        # Verify model state
        for key in model_state:
            assert torch.equal(loaded_checkpoint.model_state[key], model_state[key])

class TestTrainingCheckpoint:
    """Test training checkpoint."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_training_checkpoint_creation(self) -> Any:
        """Test training checkpoint creation."""
        model_state = {"layer1.weight": torch.randn(10, 10)}
        optimizer_state = {"param_groups": [{"lr": 0.001}]}
        scheduler_state = {"step": 100}
        model_config = {"model_name": "test_model"}
        training_config = {"batch_size": 32}
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        checkpoint = TrainingCheckpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path="test_path",
            timestamp=time.time(),
            step=100,
            epoch=5,
            metrics=metrics,
            best_metric=0.5,
            best_metric_name="loss"
        )
        
        assert checkpoint.model_state == model_state
        assert checkpoint.optimizer_state == optimizer_state
        assert checkpoint.scheduler_state == scheduler_state
        assert checkpoint.model_config == model_config
        assert checkpoint.training_config == training_config
        assert checkpoint.step == 100
        assert checkpoint.epoch == 5
        assert checkpoint.metrics == metrics
        assert checkpoint.best_metric == 0.5
        assert checkpoint.best_metric_name == "loss"
    
    def test_training_checkpoint_save_load(self) -> Any:
        """Test training checkpoint save and load."""
        model_state = {"layer1.weight": torch.randn(10, 10)}
        optimizer_state = {"param_groups": [{"lr": 0.001}]}
        model_config = {"model_name": "test_model"}
        training_config = {"batch_size": 32}
        metrics = {"loss": 0.5}
        
        checkpoint = TrainingCheckpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=None,
            model_config=model_config,
            training_config=training_config,
            checkpoint_path="test_path",
            timestamp=time.time(),
            step=100,
            epoch=5,
            metrics=metrics
        )
        
        # Save checkpoint
        save_path = os.path.join(self.temp_dir, "test_checkpoint")
        checkpoint.save(save_path)
        
        # Load checkpoint
        loaded_checkpoint = TrainingCheckpoint.load(save_path)
        
        # Verify loaded checkpoint
        assert loaded_checkpoint.model_config == model_config
        assert loaded_checkpoint.training_config == training_config
        assert loaded_checkpoint.step == 100
        assert loaded_checkpoint.epoch == 5
        assert loaded_checkpoint.metrics == metrics
        
        # Verify model state
        for key in model_state:
            assert torch.equal(loaded_checkpoint.model_state[key], model_state[key])
        
        # Verify optimizer state
        assert loaded_checkpoint.optimizer_state == optimizer_state

class TestCheckpointManager:
    """Test checkpoint manager."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self) -> Any:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_checkpoint_manager_initialization(self) -> Any:
        """Test checkpoint manager initialization."""
        strategy = CheckpointStrategy(save_steps=1000, save_total_limit=3)
        manager = CheckpointManager(checkpoint_dir=self.temp_dir, strategy=strategy)
        
        assert manager.checkpoint_dir == Path(self.temp_dir)
        assert manager.strategy == strategy
        assert manager.checkpoints == []
        assert manager.best_metric is None
        assert manager.best_checkpoint_path is None
    
    def test_should_save_checkpoint(self) -> Any:
        """Test checkpoint saving decision logic."""
        strategy = CheckpointStrategy(save_steps=1000, save_best_only=False)
        manager = CheckpointManager(checkpoint_dir=self.temp_dir, strategy=strategy)
        
        # Should save every 1000 steps
        assert manager.should_save_checkpoint(1000) is True
        assert manager.should_save_checkpoint(2000) is True
        assert manager.should_save_checkpoint(500) is False
    
    def test_should_save_checkpoint_best_only(self) -> Any:
        """Test checkpoint saving with best_only strategy."""
        strategy = CheckpointStrategy(save_steps=1000, save_best_only=True, monitor="loss", mode="min")
        manager = CheckpointManager(checkpoint_dir=self.temp_dir, strategy=strategy)
        
        # First metric should always save
        assert manager.should_save_checkpoint(1000, 0.5) is True
        
        # Better metric should save
        assert manager.should_save_checkpoint(2000, 0.3) is True
        
        # Worse metric should not save
        assert manager.should_save_checkpoint(3000, 0.7) is False
    
    def test_save_and_load_checkpoint(self) -> Any:
        """Test saving and loading checkpoints."""
        strategy = CheckpointStrategy(save_steps=1000, save_total_limit=3)
        manager = CheckpointManager(checkpoint_dir=self.temp_dir, strategy=strategy)
        
        # Create mock model and optimizer
        model = Mock()
        model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
        
        optimizer = Mock()
        optimizer.state_dict.return_value = {"param_groups": [{"lr": 0.001}]}
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=1,
            step=1000,
            metrics={"loss": 0.5}
        )
        
        assert os.path.exists(checkpoint_path)
        assert len(manager.checkpoints) == 1
        
        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint(checkpoint_path)
        assert loaded_checkpoint.step == 1000
        assert loaded_checkpoint.epoch == 1
        assert loaded_checkpoint.metrics["loss"] == 0.5
    
    def test_cleanup_old_checkpoints(self) -> Any:
        """Test cleanup of old checkpoints."""
        strategy = CheckpointStrategy(save_steps=1000, save_total_limit=2)
        manager = CheckpointManager(checkpoint_dir=self.temp_dir, strategy=strategy)
        
        # Create mock model and optimizer
        model = Mock()
        model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
        
        optimizer = Mock()
        optimizer.state_dict.return_value = {"param_groups": [{"lr": 0.001}]}
        
        # Save multiple checkpoints
        for i in range(3):
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=i,
                step=(i + 1) * 1000,
                metrics={"loss": 0.5}
            )
        
        # Should only keep 2 checkpoints
        assert len(manager.checkpoints) == 2

class TestMetricAggregator:
    """Test metric aggregator."""
    
    def test_metric_aggregator_initialization(self) -> Any:
        """Test metric aggregator initialization."""
        aggregator = MetricAggregator(window_size=100)
        
        assert aggregator.window_size == 100
        assert len(aggregator.values) == 0
    
    def test_metric_aggregator_add_value(self) -> Any:
        """Test adding values to aggregator."""
        aggregator = MetricAggregator(window_size=3)
        
        aggregator.add_value(1.0, 1)
        aggregator.add_value(2.0, 2)
        aggregator.add_value(3.0, 3)
        
        assert len(aggregator.values) == 3
        assert aggregator.values[0].value == 1.0
        assert aggregator.values[1].value == 2.0
        assert aggregator.values[2].value == 3.0
    
    def test_metric_aggregator_window_limit(self) -> Any:
        """Test window size limit."""
        aggregator = MetricAggregator(window_size=2)
        
        aggregator.add_value(1.0, 1)
        aggregator.add_value(2.0, 2)
        aggregator.add_value(3.0, 3)
        
        # Should only keep 2 values
        assert len(aggregator.values) == 2
        assert aggregator.values[0].value == 2.0
        assert aggregator.values[1].value == 3.0
    
    def test_metric_aggregator_statistics(self) -> Any:
        """Test metric aggregator statistics."""
        aggregator = MetricAggregator(window_size=5)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, value in enumerate(values):
            aggregator.add_value(value, i + 1)
        
        assert aggregator.get_mean() == 3.0
        assert aggregator.get_median() == 3.0
        assert abs(aggregator.get_std() - 1.5811) < 0.001
        assert aggregator.get_min() == 1.0
        assert aggregator.get_max() == 5.0
        assert aggregator.get_latest() == 5.0
    
    def test_metric_aggregator_summary(self) -> Any:
        """Test metric aggregator summary."""
        aggregator = MetricAggregator(window_size=3)
        
        aggregator.add_value(1.0, 1)
        aggregator.add_value(2.0, 2)
        aggregator.add_value(3.0, 3)
        
        summary = aggregator.get_summary()
        
        assert summary['mean'] == 2.0
        assert summary['median'] == 2.0
        assert summary['min'] == 1.0
        assert summary['max'] == 3.0
        assert summary['latest'] == 3.0
        assert summary['count'] == 3

class TestMetricsTracker:
    """Test metrics tracker."""
    
    def test_metrics_tracker_initialization(self) -> Any:
        """Test metrics tracker initialization."""
        tracker = MetricsTracker(log_frequency=10, window_size=100)
        
        assert tracker.log_frequency == 10
        assert tracker.window_size == 100
        assert tracker.step == 0
    
    def test_metrics_tracker_log_scalar(self) -> Any:
        """Test logging scalar metrics."""
        tracker = MetricsTracker(log_frequency=1)
        
        tracker.log_scalar("loss", 0.5, step=1)
        tracker.log_scalar("accuracy", 0.9, step=2)
        
        assert tracker.get_latest("loss") == 0.5
        assert tracker.get_latest("accuracy") == 0.9
        assert tracker.get_average("loss") == 0.5
        assert tracker.get_average("accuracy") == 0.9
    
    def test_metrics_tracker_log_scalars(self) -> Any:
        """Test logging multiple scalar metrics."""
        tracker = MetricsTracker(log_frequency=1)
        
        tracker.log_scalars({"loss": 0.5, "accuracy": 0.9}, step=1)
        
        assert tracker.get_latest("loss") == 0.5
        assert tracker.get_latest("accuracy") == 0.9
    
    def test_metrics_tracker_log_histogram(self) -> Any:
        """Test logging histogram data."""
        tracker = MetricsTracker(log_frequency=1)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracker.log_histogram("gradients", values, step=1)
        
        histogram_data = tracker.get_histogram_data("gradients")
        assert len(histogram_data) == 5
        assert histogram_data == values
    
    def test_metrics_tracker_log_text(self) -> Any:
        """Test logging text data."""
        tracker = MetricsTracker(log_frequency=1)
        
        tracker.log_text("generated_text", "Sample text", step=1)
        
        text_logs = tracker.get_text_logs("generated_text")
        assert len(text_logs) == 1
        assert "Sample text" in text_logs[0]
    
    def test_metrics_tracker_get_summary(self) -> Optional[Dict[str, Any]]:
        """Test getting metric summaries."""
        tracker = MetricsTracker(log_frequency=1)
        
        tracker.log_scalar("loss", 0.5, step=1)
        tracker.log_scalar("loss", 0.3, step=2)
        
        summary = tracker.get_summary("loss")
        assert summary['mean'] == 0.4
        assert summary['min'] == 0.3
        assert summary['max'] == 0.5
        assert summary['count'] == 2
    
    def test_metrics_tracker_export_import(self) -> Any:
        """Test exporting and importing metrics."""
        tracker = MetricsTracker(log_frequency=1)
        
        tracker.log_scalar("loss", 0.5, step=1)
        tracker.log_scalar("accuracy", 0.9, step=2)
        tracker.log_histogram("gradients", [1.0, 2.0, 3.0], step=1)
        tracker.log_text("text", "Sample", step=1)
        
        # Export metrics
        export_data = tracker.export_metrics()
        
        # Create new tracker and import
        new_tracker = MetricsTracker()
        new_tracker.import_metrics(export_data)
        
        # Verify imported metrics
        assert new_tracker.get_latest("loss") == 0.5
        assert new_tracker.get_latest("accuracy") == 0.9
        assert len(new_tracker.get_histogram_data("gradients")) == 3
        assert len(new_tracker.get_text_logs("text")) == 1

class TestTrainingMetricsTracker:
    """Test training metrics tracker."""
    
    def test_training_metrics_tracker_initialization(self) -> Any:
        """Test training metrics tracker initialization."""
        tracker = TrainingMetricsTracker(log_frequency=1, window_size=100)
        
        assert tracker.epoch == 0
        assert tracker.best_metrics == {}
    
    def test_training_metrics_tracker_log_training_step(self) -> Any:
        """Test logging training step metrics."""
        tracker = TrainingMetricsTracker(log_frequency=1)
        
        # Mock optimizer
        optimizer = Mock()
        optimizer.param_groups = [{"lr": 0.001}]
        
        # Mock model
        model = Mock()
        model.parameters.return_value = [Mock()]
        
        tracker.log_training_step(0.5, optimizer=optimizer, model=model)
        
        assert tracker.get_latest("train/loss") == 0.5
        assert tracker.get_latest("train/learning_rate") == 0.001
        assert tracker.step == 1
    
    def test_training_metrics_tracker_log_validation_step(self) -> Any:
        """Test logging validation step metrics."""
        tracker = TrainingMetricsTracker(log_frequency=1)
        
        tracker.log_validation_step(0.3, accuracy=0.9)
        
        assert tracker.get_latest("val/loss") == 0.3
        assert tracker.get_latest("val/accuracy") == 0.9
    
    def test_training_metrics_tracker_log_epoch(self) -> Any:
        """Test logging epoch metrics."""
        tracker = TrainingMetricsTracker(log_frequency=1)
        
        train_metrics = {"loss": 0.5, "accuracy": 0.8}
        val_metrics = {"loss": 0.3, "accuracy": 0.9}
        
        tracker.log_epoch(train_metrics, val_metrics)
        
        assert tracker.epoch == 1
        assert tracker.get_latest("epoch/train_loss") == 0.5
        assert tracker.get_latest("epoch/val_loss") == 0.3
        assert tracker.get_latest("epoch/train_accuracy") == 0.8
        assert tracker.get_latest("epoch/val_accuracy") == 0.9
        
        # Check best metrics
        best_metrics = tracker.get_best_metrics()
        assert best_metrics["loss"] == 0.3
        assert best_metrics["accuracy"] == 0.9

match __name__:
    case "__main__":
    pytest.main([__file__]) 