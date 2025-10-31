#!/usr/bin/env python3
"""
🧪 Test Suite for Experiment Tracking System
============================================

Tests the core functionality of the experiment tracking system including
TensorBoard and Weights & Biases integration.
"""

import unittest
import tempfile
import shutil
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np

# Import our experiment tracking system
from experiment_tracking import (
    ExperimentTracker, ExperimentConfig, create_experiment_config,
    create_experiment_tracker, TrainingMetrics, ModelCheckpoint
)


class TestExperimentTracking(unittest.TestCase):
    """Test cases for the experiment tracking system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.tensorboard_dir = os.path.join(self.test_dir, "tensorboard")
        self.model_dir = os.path.join(self.test_dir, "models")
        self.config_dir = os.path.join(self.test_dir, "configs")
        
        # Create directories
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Test configuration
        self.test_config = ExperimentConfig(
            experiment_name="test_experiment",
            project_name="test_project",
            enable_tensorboard=True,
            enable_wandb=False,  # Disable wandb for testing
            tensorboard_dir=self.tensorboard_dir,
            model_save_dir=self.model_dir,
            config_save_dir=self.config_dir,
            log_interval=1
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_experiment_config_creation(self):
        """Test experiment configuration creation."""
        config = create_experiment_config(
            experiment_name="custom_experiment",
            project_name="custom_project",
            enable_tensorboard=False
        )
        
        self.assertEqual(config.experiment_name, "custom_experiment")
        self.assertEqual(config.project_name, "custom_project")
        self.assertFalse(config.enable_tensorboard)
        self.assertTrue(config.enable_wandb)  # Default value
    
    def test_experiment_tracker_initialization(self):
        """Test experiment tracker initialization."""
        tracker = create_experiment_tracker(self.test_config)
        
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.config.experiment_name, "test_experiment")
        self.assertEqual(tracker.current_step, 0)
        self.assertEqual(tracker.current_epoch, 0)
        
        # Clean up
        tracker.close()
    
    def test_training_metrics_logging(self):
        """Test training metrics logging."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Log training step
        tracker.log_training_step(
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            gradient_norm=1.2,
            nan_count=0,
            inf_count=0,
            clipping_applied=True,
            clipping_threshold=1.0,
            training_time=0.1
        )
        
        # Check metrics were logged
        self.assertEqual(tracker.current_step, 1)
        self.assertEqual(len(tracker.metrics_history), 1)
        
        # Check first metric
        first_metric = tracker.metrics_history[0]
        self.assertEqual(first_metric.loss, 0.5)
        self.assertEqual(first_metric.accuracy, 0.8)
        self.assertEqual(first_metric.gradient_norm, 1.2)
        self.assertTrue(first_metric.clipping_applied)
        
        # Clean up
        tracker.close()
    
    def test_epoch_logging(self):
        """Test epoch-level logging."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Log epoch metrics
        epoch_metrics = {
            'epoch_loss': 0.4,
            'epoch_accuracy': 0.85,
            'epoch_time': 120.5
        }
        
        tracker.log_epoch(1, epoch_metrics)
        
        # Check epoch was updated
        self.assertEqual(tracker.current_epoch, 1)
        
        # Clean up
        tracker.close()
    
    def test_hyperparameter_logging(self):
        """Test hyperparameter logging."""
        tracker = create_experiment_tracker(self.test_config)
        
        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100
        }
        
        tracker.log_hyperparameters(hyperparams)
        
        # Clean up
        tracker.close()
    
    def test_model_architecture_logging(self):
        """Test model architecture logging."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        tracker.log_model_architecture(model)
        
        # Clean up
        tracker.close()
    
    def test_gradient_logging(self):
        """Test gradient logging."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Create a simple model with gradients
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Create dummy input and compute gradients
        x = torch.randn(1, 10)
        y = torch.randn(1, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        tracker.log_gradients(model)
        
        # Clean up
        tracker.close()
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint saving and loading."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        tracker.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            loss=0.5,
            metrics={'accuracy': 0.8}
        )
        
        # Check checkpoint was created
        self.assertEqual(len(tracker.checkpoints), 1)
        self.assertEqual(tracker.checkpoints[0].epoch, 1)
        self.assertEqual(tracker.checkpoints[0].step, 100)
        
        # Check checkpoint file exists
        checkpoint_files = list(Path(self.model_dir).glob("*.pt"))
        self.assertGreater(len(checkpoint_files), 0)
        
        # Load checkpoint
        checkpoint_data = tracker.load_checkpoint(str(checkpoint_files[0]))
        self.assertIsNotNone(checkpoint_data)
        self.assertEqual(checkpoint_data['epoch'], 1)
        self.assertEqual(checkpoint_data['step'], 100)
        
        # Clean up
        tracker.close()
    
    def test_visualization_creation(self):
        """Test training visualization creation."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Add some training data
        for i in range(10):
            tracker.log_training_step(
                loss=1.0 / (i + 1),
                accuracy=0.5 + i * 0.05,
                gradient_norm=0.5 + i * 0.1,
                nan_count=i % 2,
                inf_count=0
            )
        
        # Create visualization
        viz_data = tracker.create_visualization()
        
        # Check visualization was created
        self.assertIsNotNone(viz_data)
        self.assertIn('figure', viz_data)
        self.assertIn('metrics_summary', viz_data)
        
        # Check summary data
        summary = viz_data['metrics_summary']
        self.assertEqual(summary['total_steps'], 10)
        self.assertIsNotNone(summary['final_loss'])
        
        # Clean up
        tracker.close()
    
    def test_experiment_summary(self):
        """Test experiment summary generation."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Add some training data
        for i in range(5):
            tracker.log_training_step(
                loss=1.0 / (i + 1),
                accuracy=0.5 + i * 0.1,
                gradient_norm=0.5 + i * 0.1
            )
        
        # Get summary
        summary = tracker.get_experiment_summary()
        
        # Check summary structure
        self.assertIn('experiment_name', summary)
        self.assertIn('total_steps', summary)
        self.assertIn('loss_stats', summary)
        self.assertIn('accuracy_stats', summary)
        self.assertIn('numerical_stability', summary)
        
        # Check values
        self.assertEqual(summary['total_steps'], 5)
        self.assertEqual(summary['current_step'], 5)
        
        # Clean up
        tracker.close()
    
    def test_error_handling(self):
        """Test error handling in the system."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Test logging with invalid data
        try:
            tracker.log_training_step(
                loss=float('nan'),  # Invalid loss
                accuracy=1.5,       # Invalid accuracy
                gradient_norm=-1.0   # Invalid gradient norm
            )
            # Should not raise exception, should handle gracefully
        except Exception as e:
            self.fail(f"Error handling failed: {e}")
        
        # Clean up
        tracker.close()
    
    def test_configuration_persistence(self):
        """Test configuration saving and loading."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Test configuration
        config_data = {
            'experiment_name': 'persistence_test',
            'project_name': 'test_project',
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        # Save configuration
        save_result = tracker.save_experiment_config(config_data)
        self.assertIn("✅", save_result)
        
        # Check file was created
        config_files = list(Path(self.config_dir).glob("*.json"))
        self.assertGreater(len(config_files), 0)
        
        # Clean up
        tracker.close()
    
    def test_multiple_training_steps(self):
        """Test multiple training steps with realistic data."""
        tracker = create_experiment_tracker(self.test_config)
        
        # Simulate realistic training
        num_steps = 50
        for step in range(num_steps):
            # Simulate realistic metrics
            base_loss = 2.0 * np.exp(-step / 20)
            noise = np.random.normal(0, 0.1)
            loss = max(0.01, base_loss + noise)
            
            accuracy = min(0.95, 0.3 + 0.6 * (1 - np.exp(-step / 15)))
            accuracy += np.random.normal(0, 0.02)
            accuracy = max(0.0, min(1.0, accuracy))
            
            gradient_norm = np.random.exponential(0.8) + 0.1
            nan_count = np.random.poisson(0.05)
            inf_count = np.random.poisson(0.02)
            
            tracker.log_training_step(
                loss=loss,
                accuracy=accuracy,
                gradient_norm=gradient_norm,
                nan_count=nan_count,
                inf_count=inf_count,
                clipping_applied=gradient_norm > 1.5,
                clipping_threshold=1.5 if gradient_norm > 1.5 else None
            )
        
        # Verify all steps were logged
        self.assertEqual(tracker.current_step, num_steps)
        self.assertEqual(len(tracker.metrics_history), num_steps)
        
        # Check final metrics
        final_metric = tracker.metrics_history[-1]
        self.assertIsInstance(final_metric.loss, float)
        self.assertIsInstance(final_metric.accuracy, float)
        
        # Clean up
        tracker.close()


class TestTrainingMetrics(unittest.TestCase):
    """Test cases for TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test TrainingMetrics creation with various data types."""
        # Test with all parameters
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            gradient_norm=1.2,
            nan_count=0,
            inf_count=0,
            clipping_applied=True,
            clipping_threshold=1.0,
            training_time=0.1,
            memory_usage=256.5,
            gpu_utilization=75.2
        )
        
        self.assertEqual(metrics.loss, 0.5)
        self.assertEqual(metrics.accuracy, 0.8)
        self.assertEqual(metrics.learning_rate, 0.001)
        self.assertEqual(metrics.gradient_norm, 1.2)
        self.assertEqual(metrics.nan_count, 0)
        self.assertEqual(metrics.inf_count, 0)
        self.assertTrue(metrics.clipping_applied)
        self.assertEqual(metrics.clipping_threshold, 1.0)
        self.assertEqual(metrics.training_time, 0.1)
        self.assertEqual(metrics.memory_usage, 256.5)
        self.assertEqual(metrics.gpu_utilization, 75.2)
    
    def test_training_metrics_defaults(self):
        """Test TrainingMetrics with default values."""
        metrics = TrainingMetrics()
        
        self.assertEqual(metrics.loss, 0.0)
        self.assertIsNone(metrics.accuracy)
        self.assertIsNone(metrics.learning_rate)
        self.assertIsNone(metrics.gradient_norm)
        self.assertEqual(metrics.nan_count, 0)
        self.assertEqual(metrics.inf_count, 0)
        self.assertFalse(metrics.clipping_applied)
        self.assertIsNone(metrics.clipping_threshold)
        self.assertEqual(metrics.training_time, 0.0)
        self.assertIsNone(metrics.memory_usage)
        self.assertIsNone(metrics.gpu_utilization)


class TestModelCheckpoint(unittest.TestCase):
    """Test cases for ModelCheckpoint dataclass."""
    
    def test_model_checkpoint_creation(self):
        """Test ModelCheckpoint creation."""
        checkpoint = ModelCheckpoint(
            epoch=5,
            step=1000,
            loss=0.25,
            metrics={'accuracy': 0.9, 'val_loss': 0.3},
            model_state={'layer1.weight': torch.randn(64, 10)},
            optimizer_state={'param_groups': [{'lr': 0.001}]},
            scheduler_state={'last_epoch': 4}
        )
        
        self.assertEqual(checkpoint.epoch, 5)
        self.assertEqual(checkpoint.step, 1000)
        self.assertEqual(checkpoint.loss, 0.25)
        self.assertEqual(checkpoint.metrics['accuracy'], 0.9)
        self.assertIn('layer1.weight', checkpoint.model_state)
        self.assertIn('param_groups', checkpoint.optimizer_state)
        self.assertIn('last_epoch', checkpoint.scheduler_state)
        self.assertIsInstance(checkpoint.timestamp, str)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)






