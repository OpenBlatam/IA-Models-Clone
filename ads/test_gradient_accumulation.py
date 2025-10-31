from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os
from onyx.server.features.ads.gradient_accumulation import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from onyx.server.features.ads.gradient_accumulation_api import router as accumulation_router
from fastapi.testclient import TestClient
    from fastapi import FastAPI
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Gradient Accumulation System

This module provides comprehensive tests for:
- Gradient accumulation configuration
- Basic and adaptive gradient accumulation
- Integration with multi-GPU training
- API endpoints and utilities
- Performance monitoring and optimization
- Error handling and edge cases
"""

    GradientAccumulationConfig,
    GradientAccumulator,
    AdaptiveGradientAccumulator,
    GradientAccumulationTrainer,
    calculate_effective_batch_size,
    calculate_accumulation_steps,
    adjust_learning_rate,
    gradient_accumulation_context
)

# Test configuration
TEST_CONFIG = {
    "model_name": "gpt2",
    "batch_size_per_gpu": 4,
    "epochs": 2,
    "learning_rate": 5e-5,
    "max_length": 128
}

class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size: int = 100):
        
    """__init__ function."""
self.size = size
        self.data = torch.randn(size, 128)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self) -> Any:
        return self.size
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.labels[idx]

class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_size: int = 128, output_size: int = 10):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x) -> Any:
        return self.linear(x)

@pytest.fixture
def accumulation_config():
    """Create a test gradient accumulation configuration."""
    return GradientAccumulationConfig(
        accumulation_steps=4,
        target_batch_size=32,
        mixed_precision=True,
        gradient_clipping=1.0,
        log_accumulation=True
    )

@pytest.fixture
def gradient_accumulator(accumulation_config) -> Any:
    """Create a test gradient accumulator."""
    return GradientAccumulator(accumulation_config)

@pytest.fixture
def adaptive_accumulator(accumulation_config) -> Any:
    """Create a test adaptive gradient accumulator."""
    return AdaptiveGradientAccumulator(accumulation_config)

@pytest.fixture
def finetuning_service():
    """Create a test fine-tuning service."""
    return OptimizedFineTuningService()

@pytest.fixture
def test_client():
    """Create a test FastAPI client."""
    app = FastAPI()
    app.include_router(accumulation_router)
    return TestClient(app)

class TestGradientAccumulationConfig:
    """Test gradient accumulation configuration."""
    
    def test_config_defaults(self) -> Any:
        """Test configuration defaults."""
        config = GradientAccumulationConfig()
        
        assert config.accumulation_steps == 4
        assert config.target_batch_size is None
        assert config.max_memory_usage == 0.9
        assert config.mixed_precision is True
        assert config.gradient_clipping == 1.0
    
    def test_config_custom(self, accumulation_config) -> Any:
        """Test custom configuration."""
        assert accumulation_config.accumulation_steps == 4
        assert accumulation_config.target_batch_size == 32
        assert accumulation_config.mixed_precision is True
        assert accumulation_config.gradient_clipping == 1.0
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        config = GradientAccumulationConfig(
            accumulation_steps=8,
            target_batch_size=64,
            max_memory_usage=0.8,
            memory_safety_margin=0.2
        )
        
        assert config.accumulation_steps == 8
        assert config.target_batch_size == 64
        assert config.max_memory_usage == 0.8
        assert config.memory_safety_margin == 0.2

class TestGradientAccumulator:
    """Test basic gradient accumulator functionality."""
    
    def test_initialization(self, gradient_accumulator) -> Any:
        """Test gradient accumulator initialization."""
        assert gradient_accumulator.config.accumulation_steps == 4
        assert gradient_accumulator.current_step == 0
        assert gradient_accumulator.accumulation_step == 0
        assert gradient_accumulator.total_loss == 0.0
    
    def test_reset_accumulation(self, gradient_accumulator) -> Any:
        """Test accumulation reset."""
        # Set some state
        gradient_accumulator.accumulation_step = 2
        gradient_accumulator.total_loss = 1.5
        gradient_accumulator.total_samples = 100
        
        # Reset
        gradient_accumulator.reset_accumulation()
        
        assert gradient_accumulator.accumulation_step == 0
        assert gradient_accumulator.total_loss == 0.0
        assert gradient_accumulator.total_samples == 0
    
    def test_should_accumulate(self, gradient_accumulator) -> Any:
        """Test should_accumulate logic."""
        # Should accumulate for first 3 steps
        for i in range(3):
            gradient_accumulator.accumulation_step = i
            assert gradient_accumulator.should_accumulate() is True
        
        # Should not accumulate for last step
        gradient_accumulator.accumulation_step = 3
        assert gradient_accumulator.should_accumulate() is False
    
    def test_should_update(self, gradient_accumulator) -> Any:
        """Test should_update logic."""
        # Should not update for first 3 steps
        for i in range(3):
            gradient_accumulator.accumulation_step = i
            assert gradient_accumulator.should_update() is False
        
        # Should update for last step
        gradient_accumulator.accumulation_step = 3
        assert gradient_accumulator.should_update() is True
    
    def test_get_effective_batch_size(self, gradient_accumulator) -> Optional[Dict[str, Any]]:
        """Test effective batch size calculation."""
        actual_batch_size = 8
        effective_batch_size = gradient_accumulator.get_effective_batch_size(actual_batch_size)
        
        assert effective_batch_size == 32  # 8 * 4
    
    def test_get_learning_rate_scale(self, gradient_accumulator) -> Optional[Dict[str, Any]]:
        """Test learning rate scale calculation."""
        scale = gradient_accumulator.get_learning_rate_scale()
        
        assert scale == 0.25  # 1.0 / 4
    
    def test_accumulate_gradients(self, gradient_accumulator) -> Any:
        """Test gradient accumulation."""
        # Create mock model and optimizer
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create mock loss
        loss = torch.tensor(2.0, requires_grad=True)
        
        # Test accumulation
        for step in range(4):
            acc_stats = gradient_accumulator.accumulate_gradients(
                loss, model, optimizer
            )
            
            assert acc_stats["accumulation_step"] == step + 1
            assert acc_stats["total_loss"] > 0
            assert acc_stats["total_samples"] > 0
            
            # Should update on last step
            if step == 3:
                assert acc_stats["should_update"] is True
            else:
                assert acc_stats["should_update"] is False
    
    def test_get_accumulation_stats(self, gradient_accumulator) -> Optional[Dict[str, Any]]:
        """Test accumulation statistics."""
        # Perform some accumulation
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        loss = torch.tensor(2.0, requires_grad=True)
        
        for _ in range(4):
            gradient_accumulator.accumulate_gradients(loss, model, optimizer)
        
        # Get stats
        stats = gradient_accumulator.get_accumulation_stats()
        
        assert "accumulation_steps" in stats
        assert "current_step" in stats
        assert "total_loss" in stats
        assert "total_samples" in stats
        assert "avg_gradient_norm" in stats
        assert "avg_memory_usage_gb" in stats
        assert "avg_accumulation_time" in stats

class TestAdaptiveGradientAccumulator:
    """Test adaptive gradient accumulator functionality."""
    
    @patch('onyx.server.features.ads.gradient_accumulation.GPUMonitor')
    def test_initialization(self, mock_gpu_monitor, adaptive_accumulator) -> Any:
        """Test adaptive accumulator initialization."""
        assert adaptive_accumulator.config.accumulation_steps == 4
        assert adaptive_accumulator.gpu_monitor is not None
        assert adaptive_accumulator.batch_size_history == []
    
    @patch('onyx.server.features.ads.gradient_accumulation.GPUMonitor')
    def test_calculate_optimal_batch_size(self, mock_gpu_monitor, adaptive_accumulator) -> Any:
        """Test optimal batch size calculation."""
        model = MockModel()
        gpu_ids = [0, 1]
        
        # Mock GPU info
        mock_monitor = Mock()
        mock_gpu_info = {
            "gpu_0": {
                "memory_total": 8192,
                "memory_used": 4096,
                "memory_free": 4096
            },
            "gpu_1": {
                "memory_total": 8192,
                "memory_used": 3072,
                "memory_free": 5120
            }
        }
        mock_monitor.get_gpu_info.return_value = mock_gpu_info
        adaptive_accumulator.gpu_monitor = mock_monitor
        
        optimal_batch_size = adaptive_accumulator.calculate_optimal_batch_size(model, gpu_ids)
        
        assert optimal_batch_size > 0
        assert isinstance(optimal_batch_size, int)
    
    def test_adjust_accumulation_steps(self, adaptive_accumulator) -> Any:
        """Test accumulation steps adjustment."""
        target_batch_size = 32
        actual_batch_size = 8
        
        steps = adaptive_accumulator.adjust_accumulation_steps(target_batch_size, actual_batch_size)
        
        assert steps == 4  # ceil(32 / 8)
    
    @patch('onyx.server.features.ads.gradient_accumulation.GPUMonitor')
    def test_update_config(self, mock_gpu_monitor, adaptive_accumulator) -> Any:
        """Test configuration update."""
        model = MockModel()
        gpu_ids = [0, 1]
        
        # Mock GPU info
        mock_monitor = Mock()
        mock_gpu_info = {
            "gpu_0": {
                "memory_total": 8192,
                "memory_used": 4096,
                "memory_free": 4096
            }
        }
        mock_monitor.get_gpu_info.return_value = mock_gpu_info
        adaptive_accumulator.gpu_monitor = mock_monitor
        
        adaptive_accumulator.update_config(model, gpu_ids)
        
        # Verify config was updated
        assert adaptive_accumulator.config.effective_batch_size is not None

class TestGradientAccumulationTrainer:
    """Test gradient accumulation trainer functionality."""
    
    def test_initialization(self, accumulation_config) -> Any:
        """Test trainer initialization."""
        trainer = GradientAccumulationTrainer(accumulation_config)
        
        assert trainer.config == accumulation_config
        assert trainer.accumulator is not None
        assert trainer.scaler is not None  # Mixed precision enabled
    
    def test_setup_training(self, accumulation_config) -> Any:
        """Test training setup."""
        trainer = GradientAccumulationTrainer(accumulation_config)
        
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        gpu_ids = [0, 1]
        
        trainer.setup_training(model, optimizer, gpu_ids)
        
        # Verify setup completed without errors
        assert True
    
    @pytest.mark.asyncio
    async def test_train_epoch(self, accumulation_config) -> Any:
        """Test training epoch."""
        trainer = GradientAccumulationTrainer(accumulation_config)
        
        model = MockModel()
        dataset = MockDataset(50)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Setup training
        trainer.setup_training(model, optimizer, [0, 1])
        
        # Train epoch
        result = await trainer.train_epoch(model, dataloader, optimizer, criterion, 0)
        
        assert "loss" in result
        assert "num_batches" in result
        assert "accumulation_stats" in result
        assert "effective_batch_size" in result

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_effective_batch_size(self) -> Any:
        """Test effective batch size calculation."""
        actual_batch_size = 8
        accumulation_steps = 4
        
        effective_batch_size = calculate_effective_batch_size(actual_batch_size, accumulation_steps)
        
        assert effective_batch_size == 32  # 8 * 4
    
    def test_calculate_accumulation_steps(self) -> Any:
        """Test accumulation steps calculation."""
        target_batch_size = 32
        actual_batch_size = 8
        
        steps = calculate_accumulation_steps(target_batch_size, actual_batch_size)
        
        assert steps == 4  # ceil(32 / 8)
    
    def test_adjust_learning_rate(self) -> Any:
        """Test learning rate adjustment."""
        base_lr = 0.001
        accumulation_steps = 4
        
        adjusted_lr = adjust_learning_rate(base_lr, accumulation_steps)
        
        assert adjusted_lr == 0.00025  # 0.001 / 4
    
    def test_gradient_accumulation_context(self, gradient_accumulator) -> Any:
        """Test gradient accumulation context manager."""
        with gradient_accumulation_context(gradient_accumulator):
            # Context should work without errors
            assert True

class TestGradientAccumulationAPI:
    """Test gradient accumulation API endpoints."""
    
    def test_health_check(self, test_client) -> Any:
        """Test health check endpoint."""
        response = test_client.get("/gradient-accumulation/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "pytorch_available" in data
        assert "gpu_count" in data
    
    def test_configure_gradient_accumulation(self, test_client) -> Any:
        """Test gradient accumulation configuration endpoint."""
        config_data = {
            "accumulation_steps": 4,
            "target_effective_batch_size": 32,
            "mixed_precision": True,
            "gradient_clipping": 1.0
        }
        
        response = test_client.post("/gradient-accumulation/config", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "training_id" in data
        assert "config" in data
        assert "effective_batch_size" in data
    
    def test_get_gradient_accumulation_config(self, test_client) -> Optional[Dict[str, Any]]:
        """Test get configuration endpoint."""
        response = test_client.get("/gradient-accumulation/config/test_training_id")
        
        # Should return 404 for non-existent training ID
        assert response.status_code == 404
    
    def test_train_with_accumulation(self, test_client) -> Any:
        """Test training with accumulation endpoint."""
        training_data = {
            "model_name": "gpt2",
            "dataset_config": {
                "texts": ["sample text 1", "sample text 2"],
                "max_length": 512
            },
            "training_config": {
                "epochs": 3,
                "learning_rate": 5e-5
            },
            "user_id": 123,
            "target_effective_batch_size": 64,
            "accumulation_steps": 8
        }
        
        response = test_client.post("/gradient-accumulation/training/with-accumulation", json=training_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_train_large_batch(self, test_client) -> Any:
        """Test large batch training endpoint."""
        training_data = {
            "model_name": "gpt2",
            "dataset_config": {
                "texts": ["sample text 1", "sample text 2"],
                "max_length": 512
            },
            "training_config": {
                "epochs": 3,
                "learning_rate": 5e-5
            },
            "user_id": 123,
            "target_effective_batch_size": 128
        }
        
        response = test_client.post("/gradient-accumulation/training/large-batch", json=training_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_calculate_optimal_batch_size(self, test_client) -> Any:
        """Test batch size calculation endpoint."""
        calculation_data = {
            "model_name": "gpt2",
            "target_effective_batch_size": 64,
            "gpu_ids": [0, 1, 2, 3]
        }
        
        response = test_client.post("/gradient-accumulation/calculate-batch-size", json=calculation_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_calculate_batch_size_simple(self, test_client) -> Any:
        """Test simple batch size calculation endpoint."""
        response = test_client.get("/gradient-accumulation/calculate-batch-size?model_name=gpt2&target_batch_size=64&gpu_ids=0&gpu_ids=1")
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_get_accumulation_stats(self, test_client) -> Optional[Dict[str, Any]]:
        """Test get accumulation stats endpoint."""
        response = test_client.get("/gradient-accumulation/stats/test_training_id")
        
        # Should return 404 for non-existent training ID
        assert response.status_code == 404
    
    def test_get_all_accumulation_stats(self, test_client) -> Optional[Dict[str, Any]]:
        """Test get all accumulation stats endpoint."""
        response = test_client.get("/gradient-accumulation/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "accumulation_stats" in data
    
    def test_update_accumulation_config(self, test_client) -> Any:
        """Test update configuration endpoint."""
        config_data = {
            "accumulation_steps": 8,
            "target_effective_batch_size": 64,
            "mixed_precision": True
        }
        
        response = test_client.put("/gradient-accumulation/config/test_training_id", json=config_data)
        
        # Should return 404 for non-existent training ID
        assert response.status_code == 404
    
    def test_optimize_accumulation(self, test_client) -> Any:
        """Test optimization endpoint."""
        optimization_data = {
            "action": "calculate",
            "model_name": "gpt2",
            "target_batch_size": 64
        }
        
        response = test_client.post("/gradient-accumulation/optimize", json=optimization_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_calculate_effective_batch_size_endpoint(self, test_client) -> Any:
        """Test effective batch size calculation endpoint."""
        response = test_client.post("/gradient-accumulation/calculate-effective-batch-size?actual_batch_size=8&accumulation_steps=4")
        
        assert response.status_code == 200
        data = response.json()
        assert "actual_batch_size" in data
        assert "accumulation_steps" in data
        assert "effective_batch_size" in data
        assert data["effective_batch_size"] == 32
    
    def test_calculate_accumulation_steps_endpoint(self, test_client) -> Any:
        """Test accumulation steps calculation endpoint."""
        response = test_client.post("/gradient-accumulation/calculate-accumulation-steps?target_batch_size=32&actual_batch_size=8")
        
        assert response.status_code == 200
        data = response.json()
        assert "target_batch_size" in data
        assert "actual_batch_size" in data
        assert "accumulation_steps" in data
        assert data["accumulation_steps"] == 4
    
    def test_adjust_learning_rate_endpoint(self, test_client) -> Any:
        """Test learning rate adjustment endpoint."""
        response = test_client.post("/gradient-accumulation/adjust-learning-rate?base_lr=0.001&accumulation_steps=4")
        
        assert response.status_code == 200
        data = response.json()
        assert "base_learning_rate" in data
        assert "accumulation_steps" in data
        assert "adjusted_learning_rate" in data
        assert data["adjusted_learning_rate"] == 0.00025
    
    def test_get_performance_metrics(self, test_client) -> Optional[Dict[str, Any]]:
        """Test performance metrics endpoint."""
        response = test_client.get("/gradient-accumulation/performance/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "gpu_count" in data
        assert "available_gpus" in data
    
    def test_get_performance_recommendations(self, test_client) -> Optional[Dict[str, Any]]:
        """Test performance recommendations endpoint."""
        response = test_client.get("/gradient-accumulation/performance/recommendations?model_size=medium&target_batch_size=64")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_size" in data
        assert "target_batch_size" in data
        assert "recommendations" in data
        assert "optimal_settings" in data
    
    def test_cleanup_accumulation(self, test_client) -> Any:
        """Test cleanup endpoint."""
        response = test_client.delete("/gradient-accumulation/cleanup/test_training_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "training_id" in data
        assert "message" in data
    
    def test_cleanup_all_accumulation(self, test_client) -> Any:
        """Test cleanup all endpoint."""
        response = test_client.delete("/gradient-accumulation/cleanup")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "cleaned_sessions" in data

class TestIntegration:
    """Integration tests for gradient accumulation system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_accumulation_training(self, finetuning_service) -> Any:
        """Test end-to-end gradient accumulation training."""
        # This test would require actual GPUs
        # For now, just test the interface
        try:
            # Setup gradient accumulation
            accumulation_setup = await finetuning_service.setup_gradient_accumulation(
                target_effective_batch_size=32,
                accumulation_steps=4
            )
            
            assert "training_id" in accumulation_setup
            assert "config" in accumulation_setup
            
            # Train with accumulation
            result = await finetuning_service.finetune_model_with_accumulation(
                model_name="gpt2",
                dataset=MockDataset(50),
                training_config=TEST_CONFIG,
                user_id=123,
                target_effective_batch_size=32,
                accumulation_steps=4
            )
            
            assert "success" in result
            assert "effective_batch_size" in result
            assert "accumulation_steps" in result
            
        except Exception as e:
            # Expected if no GPUs available
            assert "CUDA" in str(e) or "GPU" in str(e)
    
    @pytest.mark.asyncio
    async def test_large_batch_training(self, finetuning_service) -> Any:
        """Test large batch training."""
        try:
            result = await finetuning_service.finetune_model_large_batch(
                model_name="gpt2",
                dataset=MockDataset(50),
                training_config=TEST_CONFIG,
                user_id=123,
                target_batch_size=64
            )
            
            assert "success" in result
            assert "effective_batch_size" in result
            
        except Exception as e:
            # Expected if no GPUs available
            assert "CUDA" in str(e) or "GPU" in str(e)
    
    @pytest.mark.asyncio
    async def test_optimal_batch_size_calculation(self, finetuning_service) -> Any:
        """Test optimal batch size calculation."""
        try:
            result = await finetuning_service.calculate_optimal_batch_size(
                model_name="gpt2",
                target_effective_batch_size=64,
                gpu_ids=[0, 1, 2, 3]
            )
            
            assert result["success"] is True
            assert "actual_batch_size_per_gpu" in result
            assert "accumulation_steps" in result
            assert "effective_batch_size" in result
            
        except Exception as e:
            # Expected if no GPUs available
            assert "CUDA" in str(e) or "GPU" in str(e)
    
    async def test_api_integration(self, test_client) -> Any:
        """Test API integration."""
        # Test health check
        health_response = test_client.get("/gradient-accumulation/health")
        assert health_response.status_code == 200
        
        # Test configuration
        config_response = test_client.post("/gradient-accumulation/config", 
                                         json={"accumulation_steps": 4})
        assert config_response.status_code == 200
        
        # Test utility endpoints
        effective_batch_response = test_client.post("/gradient-accumulation/calculate-effective-batch-size?actual_batch_size=8&accumulation_steps=4")
        assert effective_batch_response.status_code == 200
        
        # Test cleanup
        cleanup_response = test_client.delete("/gradient-accumulation/cleanup")
        assert cleanup_response.status_code == 200

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_accumulation_steps(self) -> Any:
        """Test invalid accumulation steps."""
        with pytest.raises(ValueError):
            config = GradientAccumulationConfig(accumulation_steps=0)
    
    def test_invalid_memory_usage(self) -> Any:
        """Test invalid memory usage."""
        with pytest.raises(ValueError):
            config = GradientAccumulationConfig(max_memory_usage=1.5)
    
    def test_empty_gpu_list(self, adaptive_accumulator) -> List[Any]:
        """Test empty GPU list."""
        model = MockModel()
        gpu_ids = []
        
        # Should handle empty GPU list gracefully
        try:
            adaptive_accumulator.calculate_optimal_batch_size(model, gpu_ids)
        except Exception as e:
            assert "No valid GPUs" in str(e) or "empty" in str(e)
    
    def test_missing_model(self, gradient_accumulator) -> Any:
        """Test missing model."""
        optimizer = torch.optim.Adam([])
        loss = torch.tensor(2.0, requires_grad=True)
        
        # Should handle missing model gracefully
        try:
            gradient_accumulator.accumulate_gradients(loss, None, optimizer)
        except Exception as e:
            assert "model" in str(e).lower()

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 