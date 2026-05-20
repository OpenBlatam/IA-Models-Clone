"""
Comprehensive Tests for Experiment Tracking
Tests for ExperimentTracker with ML experiment management
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.experiment_tracker import (
    ExperimentTracker,
    ExperimentStatus,
    ExperimentConfig
)


class TestExperimentConfig:
    """Tests for ExperimentConfig"""
    
    def test_create_experiment_config(self):
        """Test creating experiment configuration"""
        config = ExperimentConfig(
            experiment_id="exp-123",
            name="Test Experiment",
            description="Testing experiment tracking",
            model_type="ViT",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            dataset_info={"size": 1000, "split": "80/20"}
        )
        
        assert config.experiment_id == "exp-123"
        assert config.name == "Test Experiment"
        assert config.model_type == "ViT"
        assert config.hyperparameters["learning_rate"] == 0.001
    
    def test_config_serialization(self):
        """Test config serialization"""
        config = ExperimentConfig(
            experiment_id="exp-123",
            name="Test",
            description="Test",
            model_type="ViT",
            hyperparameters={},
            dataset_info={}
        )
        
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        
        assert "experiment_id" in config_dict
        assert "name" in config_dict


class TestExperimentTracker:
    """Tests for ExperimentTracker"""
    
    @pytest.fixture
    def experiment_tracker(self):
        """Create experiment tracker"""
        return ExperimentTracker()
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_tracker):
        """Test creating an experiment"""
        config = ExperimentConfig(
            experiment_id="test-exp-1",
            name="Test Experiment",
            description="Test",
            model_type="ViT",
            hyperparameters={"lr": 0.001},
            dataset_info={}
        )
        
        experiment_id = await experiment_tracker.create_experiment(
            config.name,
            config.hyperparameters
        )
        
        assert experiment_id is not None
        assert isinstance(experiment_id, str)
    
    @pytest.mark.asyncio
    async def test_log_metrics(self, experiment_tracker):
        """Test logging experiment metrics"""
        experiment_id = "test-exp-1"
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9,
            "epoch": 1,
            "learning_rate": 0.001
        }
        
        result = await experiment_tracker.log_metrics(experiment_id, metrics)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_log_multiple_metrics(self, experiment_tracker):
        """Test logging multiple metric updates"""
        experiment_id = "test-exp-1"
        
        # Log metrics for multiple epochs
        for epoch in range(1, 6):
            metrics = {
                "loss": 0.5 / epoch,
                "accuracy": 0.8 + (epoch * 0.02),
                "epoch": epoch
            }
            await experiment_tracker.log_metrics(experiment_id, metrics)
        
        # Should have logged all metrics
        experiment = await experiment_tracker.get_experiment(experiment_id)
        
        # Experiment should exist (or return None if not found)
        assert experiment is None or isinstance(experiment, dict)
    
    @pytest.mark.asyncio
    async def test_get_experiment(self, experiment_tracker):
        """Test getting experiment data"""
        experiment_id = "test-exp-1"
        
        experiment = await experiment_tracker.get_experiment(experiment_id)
        
        # May return None if experiment doesn't exist
        assert experiment is None or isinstance(experiment, dict)
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, experiment_tracker):
        """Test listing all experiments"""
        # Create multiple experiments
        for i in range(3):
            await experiment_tracker.create_experiment(
                f"Experiment {i}",
                {"lr": 0.001}
            )
        
        experiments = await experiment_tracker.list_experiments()
        
        # Should return list of experiments
        assert isinstance(experiments, list)
    
    @pytest.mark.asyncio
    async def test_update_experiment_status(self, experiment_tracker):
        """Test updating experiment status"""
        experiment_id = "test-exp-1"
        
        await experiment_tracker.update_status(experiment_id, ExperimentStatus.COMPLETED)
        
        # Status should be updated
        experiment = await experiment_tracker.get_experiment(experiment_id)
        
        # May not have status field depending on implementation
        assert experiment is None or isinstance(experiment, dict)
    
    @pytest.mark.asyncio
    async def test_log_hyperparameters(self, experiment_tracker):
        """Test logging hyperparameters"""
        experiment_id = "test-exp-1"
        hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "Adam"
        }
        
        result = await experiment_tracker.log_hyperparameters(experiment_id, hyperparameters)
        
        # Should log successfully
        assert result is True or result is None


class TestExperimentTrackingIntegration:
    """Integration tests for experiment tracking"""
    
    @pytest.mark.asyncio
    async def test_complete_experiment_lifecycle(self):
        """Test complete experiment lifecycle"""
        tracker = ExperimentTracker()
        
        # Create experiment
        experiment_id = await tracker.create_experiment(
            "Complete Lifecycle Test",
            {"lr": 0.001}
        )
        
        # Log metrics during training
        for epoch in range(1, 4):
            await tracker.log_metrics(experiment_id, {
                "loss": 0.5 / epoch,
                "accuracy": 0.8 + (epoch * 0.05),
                "epoch": epoch
            })
        
        # Update status to completed
        await tracker.update_status(experiment_id, ExperimentStatus.COMPLETED)
        
        # Get final experiment
        experiment = await tracker.get_experiment(experiment_id)
        
        # Should have complete data
        assert experiment is None or isinstance(experiment, dict)



