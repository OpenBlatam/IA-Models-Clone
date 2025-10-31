from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import unittest
import tempfile
import shutil
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from onyx.server.features.ads.config_manager import (
from onyx.server.features.ads.experiment_tracker import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive Test Suite for Configuration Management and Experiment Tracking

This module provides extensive testing for:
- Configuration management functionality
- Experiment tracking with different backends
- Model checkpointing and versioning
- Integration with existing systems
- Error handling and edge cases
"""

# Import the modules to test
    ConfigManager, ModelConfig, TrainingConfig, DataConfig,
    ExperimentConfig, OptimizationConfig, DeploymentConfig,
    ConfigType, create_config_from_dict, merge_configs
)

    ExperimentTracker, ExperimentMetadata, CheckpointInfo,
    CheckpointManager, TrackingBackendBase, LocalBackend,
    create_experiment_tracker, experiment_context
)

class TestConfigurationManager(unittest.TestCase):
    """Test cases for Configuration Manager."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_manager = ConfigManager(str(self.config_dir))
        
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_default_configs(self) -> Any:
        """Test creating default configuration files."""
        project_name = "test_project"
        config_files = self.config_manager.create_default_configs(project_name)
        
        # Check that all config files were created
        expected_configs = ['model', 'training', 'data', 'experiment', 'optimization', 'deployment']
        for config_type in expected_configs:
            self.assertIn(config_type, config_files)
            self.assertTrue(Path(config_files[config_type]).exists())
    
    def test_save_and_load_config(self) -> Any:
        """Test saving and loading configurations."""
        # Create test config
        model_config = ModelConfig(
            name="test_model",
            type="transformer",
            architecture="bert-base-uncased",
            input_size=768,
            output_size=10,
            hidden_sizes=[512, 256],
            dropout_rate=0.1
        )
        
        # Save config
        config_path = self.config_manager.save_config(
            model_config, "test_model_config.yaml", ConfigType.MODEL
        )
        
        # Load config
        loaded_config = self.config_manager.load_config(config_path, ModelConfig)
        
        # Verify loaded config matches original
        self.assertEqual(loaded_config.name, model_config.name)
        self.assertEqual(loaded_config.type, model_config.type)
        self.assertEqual(loaded_config.architecture, model_config.architecture)
        self.assertEqual(loaded_config.input_size, model_config.input_size)
        self.assertEqual(loaded_config.output_size, model_config.output_size)
        self.assertEqual(loaded_config.hidden_sizes, model_config.hidden_sizes)
        self.assertEqual(loaded_config.dropout_rate, model_config.dropout_rate)
    
    def test_load_all_configs(self) -> Any:
        """Test loading all configurations for a project."""
        # Create default configs
        project_name = "test_project"
        self.config_manager.create_default_configs(project_name)
        
        # Load all configs
        configs = self.config_manager.load_all_configs(project_name)
        
        # Check that all configs were loaded
        expected_configs = ['model', 'training', 'data', 'experiment', 'optimization', 'deployment']
        for config_type in expected_configs:
            self.assertIn(config_type, configs)
            self.assertIsNotNone(configs[config_type])
    
    def test_update_config(self) -> Any:
        """Test updating existing configuration."""
        # Create and save initial config
        training_config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            epochs=10
        )
        
        config_path = self.config_manager.save_config(
            training_config, "test_training_config.yaml", ConfigType.TRAINING
        )
        
        # Update config
        updates = {
            "batch_size": 64,
            "learning_rate": 2e-4,
            "epochs": 20
        }
        
        updated_path = self.config_manager.update_config(config_path, updates)
        
        # Load updated config
        updated_config = self.config_manager.load_config(updated_path, TrainingConfig)
        
        # Verify updates
        self.assertEqual(updated_config.batch_size, 64)
        self.assertEqual(updated_config.learning_rate, 2e-4)
        self.assertEqual(updated_config.epochs, 20)
    
    def test_validate_config(self) -> bool:
        """Test configuration validation."""
        # Valid model config
        valid_model_config = ModelConfig(
            name="test_model",
            type="transformer",
            architecture="bert-base-uncased",
            input_size=768,
            output_size=10
        )
        self.assertTrue(self.config_manager.validate_config(valid_model_config, ConfigType.MODEL))
        
        # Invalid model config (missing required fields)
        invalid_model_config = ModelConfig(
            name="",  # Empty name
            type="transformer",
            architecture="",  # Empty architecture
            input_size=768,
            output_size=10
        )
        self.assertFalse(self.config_manager.validate_config(invalid_model_config, ConfigType.MODEL))
        
        # Valid training config
        valid_training_config = TrainingConfig(
            batch_size=32,
            learning_rate=1e-4,
            epochs=10,
            validation_split=0.2
        )
        self.assertTrue(self.config_manager.validate_config(valid_training_config, ConfigType.TRAINING))
        
        # Invalid training config
        invalid_training_config = TrainingConfig(
            batch_size=-1,  # Invalid batch size
            learning_rate=0,  # Invalid learning rate
            epochs=10,
            validation_split=1.5  # Invalid validation split
        )
        self.assertFalse(self.config_manager.validate_config(invalid_training_config, ConfigType.TRAINING))
    
    def test_get_config_summary(self) -> Optional[Dict[str, Any]]:
        """Test generating configuration summary."""
        # Create test configs
        configs = {
            'model': ModelConfig(
                name="test_model",
                type="transformer",
                architecture="bert-base-uncased",
                input_size=768,
                output_size=10
            ),
            'training': TrainingConfig(
                batch_size=32,
                learning_rate=1e-4,
                epochs=10
            ),
            'data': DataConfig(
                num_workers=4,
                augmentation=True
            )
        }
        
        summary = self.config_manager.get_config_summary(configs)
        
        # Check summary structure
        self.assertIn('model_info', summary)
        self.assertIn('training_info', summary)
        self.assertIn('data_info', summary)
        
        # Check key parameters
        self.assertEqual(summary['model_info']['key_parameters']['name'], 'test_model')
        self.assertEqual(summary['training_info']['key_parameters']['batch_size'], 32)
        self.assertEqual(summary['data_info']['key_parameters']['num_workers'], 4)

class TestExperimentTracker(unittest.TestCase):
    """Test cases for Experiment Tracker."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        
        # Create test configuration
        self.experiment_config = ExperimentConfig(
            experiment_name="test_experiment",
            project_name="test_project",
            track_experiments=True,
            tracking_backend="local",
            save_checkpoints=True,
            checkpoint_dir=str(self.checkpoint_dir),
            log_metrics=["loss", "accuracy"],
            log_frequency=10
        )
        
        # Create simple test model
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_tracker_initialization(self) -> Any:
        """Test experiment tracker initialization."""
        tracker = ExperimentTracker(self.experiment_config)
        
        self.assertEqual(tracker.experiment_id, tracker.experiment_id)
        self.assertEqual(tracker.config, self.experiment_config)
        self.assertIsNotNone(tracker.checkpoint_manager)
        self.assertEqual(tracker.current_epoch, 0)
        self.assertEqual(tracker.current_step, 0)
    
    def test_start_experiment(self) -> Any:
        """Test starting an experiment."""
        tracker = ExperimentTracker(self.experiment_config)
        
        metadata = ExperimentMetadata(
            experiment_id="test_123",
            experiment_name="test_experiment",
            project_name="test_project",
            created_at=datetime.now(),
            tags=["test", "example"]
        )
        
        tracker.start_experiment(metadata)
        
        # Check that experiment directory was created
        experiment_dir = self.checkpoint_dir / tracker.experiment_id
        self.assertTrue(experiment_dir.exists())
        
        # Check that metadata was saved
        metadata_file = experiment_dir / "metadata.yaml"
        self.assertTrue(metadata_file.exists())
    
    def test_log_hyperparameters(self) -> Any:
        """Test logging hyperparameters."""
        tracker = ExperimentTracker(self.experiment_config)
        tracker.start_experiment()
        
        hyperparameters = {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 10
        }
        
        tracker.log_hyperparameters(hyperparameters)
        
        # Check that hyperparameters were saved
        hp_file = self.checkpoint_dir / tracker.experiment_id / "hyperparameters.yaml"
        self.assertTrue(hp_file.exists())
        
        # Verify hyperparameters
        with open(hp_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            loaded_hp = yaml.safe_load(f)
        
        self.assertEqual(loaded_hp['learning_rate'], 1e-4)
        self.assertEqual(loaded_hp['batch_size'], 32)
        self.assertEqual(loaded_hp['epochs'], 10)
    
    def test_log_metrics(self) -> Any:
        """Test logging metrics."""
        tracker = ExperimentTracker(self.experiment_config)
        tracker.start_experiment()
        
        metrics = {
            "loss": 0.5,
            "accuracy": 0.85
        }
        
        tracker.log_metrics(metrics, step=100, epoch=5)
        
        # Check that metrics were stored in history
        self.assertIn("loss", tracker.metrics_history)
        self.assertIn("accuracy", tracker.metrics_history)
        
        # Check metric history
        loss_history = tracker.metrics_history["loss"]
        self.assertEqual(len(loss_history), 1)
        self.assertEqual(loss_history[0]['value'], 0.5)
        self.assertEqual(loss_history[0]['step'], 100)
        self.assertEqual(loss_history[0]['epoch'], 5)
    
    def test_log_model_architecture(self) -> Any:
        """Test logging model architecture."""
        tracker = ExperimentTracker(self.experiment_config)
        tracker.start_experiment()
        
        tracker.log_model_architecture(self.model)
        
        # Check that model summary was saved
        summary_file = self.checkpoint_dir / tracker.experiment_id / "model_summary.txt"
        self.assertTrue(summary_file.exists())
        
        # Verify summary content
        with open(summary_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            summary = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        self.assertIn("Model Architecture Summary", summary)
        self.assertIn("Sequential", summary)
        self.assertIn("Total Parameters", summary)
    
    def test_save_and_load_checkpoint(self) -> Any:
        """Test saving and loading checkpoints."""
        tracker = ExperimentTracker(self.experiment_config)
        tracker.start_experiment()
        
        # Save checkpoint
        metrics = {"loss": 0.5, "accuracy": 0.85}
        checkpoint_path = tracker.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=metrics,
            is_best=True
        )
        
        # Verify checkpoint was created
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Create new model for loading
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_optimizer = optim.Adam(new_model.parameters())
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        
        # Load checkpoint
        checkpoint_info = tracker.load_checkpoint(
            model=new_model,
            checkpoint_path=checkpoint_path,
            optimizer=new_optimizer,
            scheduler=new_scheduler
        )
        
        # Verify checkpoint info
        self.assertEqual(checkpoint_info['epoch'], 0)
        self.assertEqual(checkpoint_info['step'], 0)
        self.assertEqual(checkpoint_info['metrics']['loss'], 0.5)
        self.assertEqual(checkpoint_info['metrics']['accuracy'], 0.85)
        self.assertTrue(checkpoint_info['is_best'])
    
    def test_end_experiment(self) -> Any:
        """Test ending an experiment."""
        tracker = ExperimentTracker(self.experiment_config)
        tracker.start_experiment()
        
        # Log some metrics
        tracker.log_metrics({"loss": 0.5}, step=100)
        
        # End experiment
        tracker.end_experiment()
        
        # Check that experiment summary was saved
        summary_file = self.checkpoint_dir / tracker.experiment_id / "experiment_summary.yaml"
        self.assertTrue(summary_file.exists())
        
        # Verify summary content
        with open(summary_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            summary = yaml.safe_load(f)
        
        self.assertEqual(summary['experiment_id'], tracker.experiment_id)
        self.assertEqual(summary['total_steps'], 100)
        self.assertIn('final_metrics', summary)

class TestCheckpointManager(unittest.TestCase):
    """Test cases for Checkpoint Manager."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            max_checkpoints=3,
            save_optimizer=True,
            save_scheduler=True
        )
        
        # Create test model
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self) -> Any:
        """Test saving checkpoints."""
        experiment_id = "test_exp"
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            experiment_id=experiment_id,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=5,
            step=1000,
            metrics=metrics,
            is_best=True
        )
        
        # Verify checkpoint was created
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Verify checkpoint info was saved
        checkpoint_infos = self.checkpoint_manager.get_checkpoint_info(experiment_id)
        self.assertEqual(len(checkpoint_infos), 1)
        
        checkpoint_info = checkpoint_infos[0]
        self.assertEqual(checkpoint_info['epoch'], 5)
        self.assertEqual(checkpoint_info['step'], 1000)
        self.assertEqual(checkpoint_info['metrics']['loss'], 0.5)
        self.assertTrue(checkpoint_info['is_best'])
    
    def test_load_checkpoint(self) -> Any:
        """Test loading checkpoints."""
        experiment_id = "test_exp"
        metrics = {"loss": 0.5, "accuracy": 0.85}
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            experiment_id=experiment_id,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=5,
            step=1000,
            metrics=metrics,
            is_best=True
        )
        
        # Create new model for loading
        new_model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        new_optimizer = optim.Adam(new_model.parameters())
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        
        # Load checkpoint
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            model=new_model,
            checkpoint_path=checkpoint_path,
            optimizer=new_optimizer,
            scheduler=new_scheduler
        )
        
        # Verify checkpoint info
        self.assertEqual(checkpoint_info['epoch'], 5)
        self.assertEqual(checkpoint_info['step'], 1000)
        self.assertEqual(checkpoint_info['metrics']['loss'], 0.5)
        self.assertTrue(checkpoint_info['is_best'])
    
    def test_get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Test getting best checkpoint."""
        experiment_id = "test_exp"
        
        # Save multiple checkpoints
        for i in range(3):
            is_best = (i == 1)  # Second checkpoint is best
            self.checkpoint_manager.save_checkpoint(
                experiment_id=experiment_id,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=i,
                step=i*100,
                metrics={"loss": 1.0 - i*0.1},
                is_best=is_best
            )
        
        # Get best checkpoint
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint(experiment_id)
        self.assertIsNotNone(best_checkpoint)
        self.assertIn("_best.pt", best_checkpoint)
    
    def test_get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Test getting latest checkpoint."""
        experiment_id = "test_exp"
        
        # Save multiple checkpoints
        for i in range(3):
            self.checkpoint_manager.save_checkpoint(
                experiment_id=experiment_id,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=i,
                step=i*100,
                metrics={"loss": 1.0 - i*0.1},
                is_best=False
            )
        
        # Get latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(experiment_id)
        self.assertIsNotNone(latest_checkpoint)
    
    def test_cleanup_old_checkpoints(self) -> Any:
        """Test cleanup of old checkpoints."""
        experiment_id = "test_exp"
        
        # Save more checkpoints than max allowed
        for i in range(5):  # More than max_checkpoints (3)
            self.checkpoint_manager.save_checkpoint(
                experiment_id=experiment_id,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=i,
                step=i*100,
                metrics={"loss": 1.0 - i*0.1},
                is_best=(i == 2)  # Third checkpoint is best
            )
        
        # Check that only max_checkpoints remain
        checkpoint_infos = self.checkpoint_manager.get_checkpoint_info(experiment_id)
        self.assertLessEqual(len(checkpoint_infos), 3)
        
        # Verify best checkpoint is still there
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint(experiment_id)
        self.assertIsNotNone(best_checkpoint)

class TestTrackingBackends(unittest.TestCase):
    """Test cases for tracking backends."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.experiment_config = ExperimentConfig(
            experiment_name="test_experiment",
            project_name="test_project",
            track_experiments=True,
            tracking_backend="local",
            save_checkpoints=True,
            checkpoint_dir="./checkpoints"
        )
    
    def test_local_backend(self) -> Any:
        """Test local tracking backend."""
        backend = LocalBackend(self.experiment_config)
        
        metadata = ExperimentMetadata(
            experiment_id="test_123",
            experiment_name="test_experiment",
            project_name="test_project",
            created_at=datetime.now()
        )
        
        # Test start experiment
        backend.start_experiment(metadata)
        
        # Test log hyperparameters
        hyperparameters = {"learning_rate": 1e-4, "batch_size": 32}
        backend.log_hyperparameters(hyperparameters)
        
        # Test log metrics
        metrics = {"loss": 0.5, "accuracy": 0.85}
        backend.log_metrics(metrics, step=100)
        
        # Test end experiment
        backend.end_experiment()
    
    @patch('onyx.server.features.ads.experiment_tracker.WANDB_AVAILABLE', True)
    @patch('onyx.server.features.ads.experiment_tracker.wandb')
    def test_wandb_backend(self, mock_wandb) -> Any:
        """Test Weights & Biases backend."""
        self.experiment_config.tracking_backend = "wandb"
        
        # Mock wandb.init
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        
        tracker = ExperimentTracker(self.experiment_config)
        
        # Test that wandb backend is initialized
        self.assertIsNotNone(tracker.backend)
        
        # Test start experiment
        metadata = ExperimentMetadata(
            experiment_id="test_123",
            experiment_name="test_experiment",
            project_name="test_project",
            created_at=datetime.now()
        )
        
        tracker.start_experiment(metadata)
        mock_wandb.init.assert_called_once()
        
        # Test log metrics
        tracker.log_metrics({"loss": 0.5}, step=100)
        mock_run.log.assert_called()
        
        # Test end experiment
        tracker.end_experiment()
        mock_run.finish.assert_called_once()

class TestIntegration(unittest.TestCase):
    """Test integration scenarios."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        
        # Create config manager
        self.config_manager = ConfigManager(str(self.config_dir))
        
        # Create experiment config
        self.experiment_config = ExperimentConfig(
            experiment_name="integration_test",
            project_name="test_project",
            track_experiments=True,
            tracking_backend="local",
            save_checkpoints=True,
            checkpoint_dir=str(self.checkpoint_dir)
        )
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self) -> Any:
        """Test complete workflow from config creation to experiment tracking."""
        # 1. Create configurations
        config_files = self.config_manager.create_default_configs("integration_test")
        
        # 2. Load configurations
        configs = self.config_manager.load_all_configs("integration_test")
        
        # 3. Create experiment tracker
        tracker = create_experiment_tracker(self.experiment_config)
        
        # 4. Start experiment
        metadata = ExperimentMetadata(
            experiment_id="integration_123",
            experiment_name="integration_test",
            project_name="test_project",
            created_at=datetime.now(),
            tags=["integration", "test"]
        )
        
        tracker.start_experiment(metadata)
        
        # 5. Log hyperparameters
        hyperparameters = {
            "model": asdict(configs['model']),
            "training": asdict(configs['training']),
            "data": asdict(configs['data'])
        }
        tracker.log_hyperparameters(hyperparameters)
        
        # 6. Create test model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        optimizer = optim.Adam(model.parameters())
        
        # 7. Log model architecture
        tracker.log_model_architecture(model)
        
        # 8. Simulate training loop
        for epoch in range(3):
            for step in range(10):
                # Simulate training step
                loss = torch.tensor(1.0 - epoch * 0.1 - step * 0.01)
                accuracy = torch.tensor(0.5 + epoch * 0.1 + step * 0.01)
                
                # Log metrics
                tracker.log_metrics({
                    "loss": loss.item(),
                    "accuracy": accuracy.item()
                }, step=step, epoch=epoch)
                
                # Save checkpoint periodically
                if step % 5 == 0:
                    tracker.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        metrics={"loss": loss.item(), "accuracy": accuracy.item()},
                        is_best=(epoch == 2 and step == 9)  # Last step is best
                    )
        
        # 9. End experiment
        tracker.end_experiment()
        
        # 10. Verify results
        experiment_dir = self.checkpoint_dir / tracker.experiment_id
        self.assertTrue(experiment_dir.exists())
        
        # Check that files were created
        self.assertTrue((experiment_dir / "metadata.yaml").exists())
        self.assertTrue((experiment_dir / "hyperparameters.yaml").exists())
        self.assertTrue((experiment_dir / "model_summary.txt").exists())
        self.assertTrue((experiment_dir / "experiment_summary.yaml").exists())
        self.assertTrue((experiment_dir / "checkpoint_info.yaml").exists())
        
        # Check that checkpoints were created
        checkpoint_infos = tracker.checkpoint_manager.get_checkpoint_info(tracker.experiment_id)
        self.assertGreater(len(checkpoint_infos), 0)
    
    def test_context_manager(self) -> Any:
        """Test experiment context manager."""
        with experiment_context(self.experiment_config) as tracker:
            # Log some metrics
            tracker.log_metrics({"loss": 0.5}, step=100)
            
            # Verify tracker is working
            self.assertIsNotNone(tracker)
            self.assertEqual(tracker.current_step, 100)

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_manager = ConfigManager(str(self.config_dir))
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_nonexistent_config(self) -> Any:
        """Test loading non-existent configuration."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent_config.yaml")
    
    def test_invalid_config_file(self) -> Any:
        """Test loading invalid configuration file."""
        # Create invalid YAML file
        invalid_config_path = self.config_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("invalid: yaml: content: [")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        with self.assertRaises(Exception):
            self.config_manager.load_config(str(invalid_config_path))
    
    def test_checkpoint_loading_error(self) -> Any:
        """Test checkpoint loading error handling."""
        checkpoint_manager = CheckpointManager()
        
        # Try to load non-existent checkpoint
        with self.assertRaises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=nn.Linear(10, 1),
                checkpoint_path="nonexistent_checkpoint.pt"
            )

def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    
    # Test configuration loading performance
    config_manager = ConfigManager()
    start_time = time.time()
    
    for i in range(100):
        config = ModelConfig(
            name=f"model_{i}",
            type="transformer",
            architecture="bert-base-uncased",
            input_size=768,
            output_size=10
        )
        config_manager.save_config(config, f"test_config_{i}.yaml", ConfigType.MODEL)
    
    end_time = time.time()
    print(f"Configuration saving performance: {end_time - start_time:.2f}s for 100 configs")
    
    # Test experiment tracking performance
    experiment_config = ExperimentConfig(
        experiment_name="perf_test",
        project_name="test_project",
        track_experiments=True,
        tracking_backend="local",
        save_checkpoints=False
    )
    
    tracker = create_experiment_tracker(experiment_config)
    tracker.start_experiment()
    
    start_time = time.time()
    for i in range(1000):
        tracker.log_metrics({"loss": 1.0 - i * 0.001}, step=i)
    
    end_time = time.time()
    print(f"Metrics logging performance: {end_time - start_time:.2f}s for 1000 metrics")
    
    tracker.end_experiment()

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance tests
    run_performance_tests() 