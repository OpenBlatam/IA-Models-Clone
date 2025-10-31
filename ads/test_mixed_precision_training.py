from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import pytest
import torch
import torch.nn as nn
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from onyx.server.features.ads.mixed_precision_training import (
from onyx.server.features.ads.gradient_accumulation import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Mixed Precision Training System

This module provides comprehensive tests for:
- Mixed precision configuration
- Basic and adaptive mixed precision trainers
- Integration with gradient accumulation
- Performance monitoring and optimization
- API endpoints and error handling
- Memory efficiency and training stability
"""

    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    AdaptiveMixedPrecisionTrainer,
    MixedPrecisionGradientAccumulator,
    create_mixed_precision_config,
    should_use_mixed_precision,
    optimize_mixed_precision_settings,
    mixed_precision_context
)
    GradientAccumulationConfig
)

# Test fixtures
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.Softmax(dim=1)
    )
    return model

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    class MockDataset:
        def __init__(self, size=100) -> Any:
            self.size = size
            self.data = torch.randn(size, 100)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self) -> Any:
            return self.size
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    return MockDataset()

@pytest.fixture
def basic_mp_config():
    """Create a basic mixed precision configuration."""
    return MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        init_scale=2**16,
        memory_efficient=True
    )

@pytest.fixture
def advanced_mp_config():
    """Create an advanced mixed precision configuration."""
    return MixedPrecisionConfig(
        enabled=True,
        dtype=torch.float16,
        autocast_enabled=True,
        scaler_enabled=True,
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        memory_efficient=True,
        cache_enabled=True,
        deterministic=False,
        log_precision=True,
        log_memory_usage=True,
        min_loss_scale=1e-4,
        max_loss_scale=2**16,
        loss_scale_window=1000,
        hysteresis=2
    )

class TestMixedPrecisionConfig:
    """Test mixed precision configuration."""
    
    def test_basic_config_creation(self) -> Any:
        """Test basic configuration creation."""
        config = MixedPrecisionConfig()
        
        assert config.enabled == True
        assert config.dtype == torch.float16
        assert config.init_scale == 2**16
        assert config.memory_efficient == True
    
    def test_advanced_config_creation(self, advanced_mp_config) -> Any:
        """Test advanced configuration creation."""
        config = advanced_mp_config
        
        assert config.enabled == True
        assert config.dtype == torch.float16
        assert config.autocast_enabled == True
        assert config.scaler_enabled == True
        assert config.init_scale == 2**16
        assert config.growth_factor == 2.0
        assert config.backoff_factor == 0.5
        assert config.memory_efficient == True
        assert config.cache_enabled == True
        assert config.log_precision == True
        assert config.log_memory_usage == True
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Test invalid dtype
        with pytest.raises(ValueError):
            MixedPrecisionConfig(dtype="invalid")
        
        # Test invalid scale values
        with pytest.raises(ValueError):
            MixedPrecisionConfig(init_scale=-1)
        
        # Test valid configuration
        config = MixedPrecisionConfig(
            enabled=True,
            dtype=torch.float16,
            init_scale=2**16
        )
        assert config.enabled == True

class TestMixedPrecisionTrainer:
    """Test mixed precision trainer."""
    
    def test_trainer_initialization(self, basic_mp_config) -> Any:
        """Test trainer initialization."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        assert trainer.config == basic_mp_config
        assert trainer.training_stats["amp_enabled"] == True
        assert trainer.scaler is not None
        assert trainer.autocast_context is not None
    
    def test_should_use_mixed_precision(self, basic_mp_config, sample_model) -> Any:
        """Test mixed precision decision logic."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Test with enabled config
        should_use = trainer.should_use_mixed_precision(sample_model)
        assert should_use == True
        
        # Test with disabled config
        disabled_config = MixedPrecisionConfig(enabled=False)
        trainer_disabled = MixedPrecisionTrainer(disabled_config)
        should_use = trainer_disabled.should_use_mixed_precision(sample_model)
        assert should_use == False
    
    def test_memory_savings_calculation(self, basic_mp_config, sample_model) -> Any:
        """Test memory savings calculation."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        memory_savings = trainer.get_memory_savings(sample_model)
        assert memory_savings > 0
        assert isinstance(memory_savings, float)
    
    @pytest.mark.asyncio
    async def test_forward_pass(self, basic_mp_config, sample_model) -> Any:
        """Test forward pass with mixed precision."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Create sample input
        inputs = torch.randn(32, 100)
        
        # Test forward pass
        outputs = trainer.forward_pass(sample_model, inputs)
        
        assert outputs is not None
        assert outputs.shape == (32, 10)
        assert outputs.dtype in [torch.float16, torch.float32]
    
    @pytest.mark.asyncio
    async def test_backward_pass(self, basic_mp_config, sample_model) -> Any:
        """Test backward pass with mixed precision."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Setup model and optimizer
        sample_model.train()
        optimizer = torch.optim.Adam(sample_model.parameters())
        
        # Create sample data
        inputs = torch.randn(32, 100)
        targets = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = trainer.forward_pass(sample_model, inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
        
        assert backward_stats is not None
        assert "backward_time" in backward_stats
        assert "scaler_scale" in backward_stats
        assert backward_stats["scaler_scale"] > 0
    
    @pytest.mark.asyncio
    async def test_optimizer_step(self, basic_mp_config, sample_model) -> Any:
        """Test optimizer step with mixed precision."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Setup model and optimizer
        sample_model.train()
        optimizer = torch.optim.Adam(sample_model.parameters())
        
        # Create sample data
        inputs = torch.randn(32, 100)
        targets = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward pass
        outputs = trainer.forward_pass(sample_model, inputs)
        loss = criterion(outputs, targets)
        trainer.backward_pass(loss, optimizer, trainer.scaler)
        
        # Optimizer step
        optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
        
        assert optimizer_stats is not None
        assert "optimizer_time" in optimizer_stats
        assert "scaler_scale" in optimizer_stats
    
    def test_training_stats(self, basic_mp_config) -> Any:
        """Test training statistics retrieval."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        stats = trainer.get_training_stats()
        
        assert "amp_enabled" in stats
        assert "scaler_scale" in stats
        assert "memory_saved" in stats
        assert "training_time" in stats
        assert "overflow_count" in stats
        assert "underflow_count" in stats
    
    def test_stats_reset(self, basic_mp_config) -> Any:
        """Test statistics reset."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Modify some stats
        trainer.training_stats["overflow_count"] = 5
        trainer.training_stats["underflow_count"] = 3
        
        # Reset stats
        trainer.reset_stats()
        
        assert trainer.training_stats["overflow_count"] == 0
        assert trainer.training_stats["underflow_count"] == 0

class TestAdaptiveMixedPrecisionTrainer:
    """Test adaptive mixed precision trainer."""
    
    def test_adaptive_trainer_initialization(self, advanced_mp_config) -> Any:
        """Test adaptive trainer initialization."""
        trainer = AdaptiveMixedPrecisionTrainer(advanced_mp_config)
        
        assert trainer.config == advanced_mp_config
        assert trainer.gpu_monitor is not None
        assert trainer.performance_history == []
        assert trainer.memory_thresholds == []
    
    def test_adaptive_mixed_precision_decision(self, advanced_mp_config, sample_model) -> Any:
        """Test adaptive mixed precision decision."""
        trainer = AdaptiveMixedPrecisionTrainer(advanced_mp_config)
        
        should_use = trainer.should_use_mixed_precision_adaptive(sample_model)
        assert should_use == True
    
    def test_precision_settings_optimization(self, advanced_mp_config, sample_model) -> Any:
        """Test precision settings optimization."""
        trainer = AdaptiveMixedPrecisionTrainer(advanced_mp_config)
        
        optimization = trainer.optimize_precision_settings(sample_model)
        
        assert "recommended_dtype" in optimization
        assert "recommended_scale" in optimization
        assert "model_params" in optimization
        assert "gpu_memory_gb" in optimization
        assert optimization["model_params"] > 0
    
    def test_config_adaptive_update(self, advanced_mp_config, sample_model) -> Any:
        """Test adaptive configuration update."""
        trainer = AdaptiveMixedPrecisionTrainer(advanced_mp_config)
        
        # Store original settings
        original_dtype = trainer.config.dtype
        original_scale = trainer.config.init_scale
        
        # Update configuration
        trainer.update_config_adaptive(sample_model)
        
        # Check that settings were updated
        assert trainer.config.dtype == original_dtype  # May not change
        assert trainer.config.init_scale == original_scale  # May not change

class TestMixedPrecisionGradientAccumulator:
    """Test mixed precision gradient accumulator."""
    
    def test_accumulator_initialization(self, basic_mp_config) -> Any:
        """Test accumulator initialization."""
        acc_config = GradientAccumulationConfig(
            accumulation_steps=4,
            mixed_precision=True
        )
        
        accumulator = MixedPrecisionGradientAccumulator(basic_mp_config, acc_config)
        
        assert accumulator.mp_config == basic_mp_config
        assert accumulator.acc_config == acc_config
        assert accumulator.mp_trainer is not None
        assert accumulator.accumulator is not None
    
    @pytest.mark.asyncio
    async def test_gradient_accumulation_mp(self, basic_mp_config, sample_model) -> Any:
        """Test gradient accumulation with mixed precision."""
        acc_config = GradientAccumulationConfig(
            accumulation_steps=4,
            mixed_precision=True
        )
        
        accumulator = MixedPrecisionGradientAccumulator(basic_mp_config, acc_config)
        
        # Setup model and optimizer
        sample_model.train()
        optimizer = torch.optim.Adam(sample_model.parameters())
        
        # Create sample data
        inputs = torch.randn(32, 100)
        targets = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = sample_model(inputs)
        loss = criterion(outputs, targets)
        
        # Accumulate gradients
        acc_stats = accumulator.accumulate_gradients_mp(
            loss, sample_model, optimizer, accumulator.mp_trainer.scaler
        )
        
        assert acc_stats is not None
        assert "should_update" in acc_stats
        assert "accumulation_step" in acc_stats
        assert "total_loss" in acc_stats
        assert "total_samples" in acc_stats
        assert "backward_stats" in acc_stats
        assert "optimizer_stats" in acc_stats
        assert "mp_stats" in acc_stats
    
    def test_combined_stats(self, basic_mp_config) -> Any:
        """Test combined statistics retrieval."""
        acc_config = GradientAccumulationConfig(
            accumulation_steps=4,
            mixed_precision=True
        )
        
        accumulator = MixedPrecisionGradientAccumulator(basic_mp_config, acc_config)
        
        stats = accumulator.get_combined_stats()
        
        assert "accumulation_steps" in stats
        assert "current_step" in stats
        assert "total_loss" in stats
        assert "total_samples" in stats
        assert "mixed_precision_enabled" in stats
        assert "mp_stats" in stats

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_mixed_precision_config(self) -> Any:
        """Test mixed precision configuration creation."""
        config = create_mixed_precision_config(
            enabled=True,
            dtype=torch.float16,
            init_scale=2**16,
            memory_efficient=True
        )
        
        assert config.enabled == True
        assert config.dtype == torch.float16
        assert config.init_scale == 2**16
        assert config.memory_efficient == True
    
    def test_should_use_mixed_precision_utility(self, sample_model) -> Any:
        """Test should_use_mixed_precision utility function."""
        # Test with large model and limited memory
        should_use = should_use_mixed_precision(sample_model, gpu_memory_gb=4.0)
        assert should_use == True
        
        # Test with large model and plenty of memory
        should_use = should_use_mixed_precision(sample_model, gpu_memory_gb=16.0)
        assert should_use == True  # Still recommended for most models
    
    def test_optimize_mixed_precision_settings(self, sample_model) -> Any:
        """Test mixed precision settings optimization."""
        optimization = optimize_mixed_precision_settings(
            model=sample_model,
            gpu_memory_gb=8.0,
            batch_size=16
        )
        
        assert "enabled" in optimization
        assert "dtype" in optimization
        assert "init_scale" in optimization
        assert "memory_efficient" in optimization
        assert "cache_enabled" in optimization
        assert optimization["enabled"] == True

class TestContextManager:
    """Test context manager."""
    
    def test_mixed_precision_context(self, basic_mp_config) -> Any:
        """Test mixed precision context manager."""
        with mixed_precision_context(basic_mp_config) as trainer:
            assert trainer is not None
            assert isinstance(trainer, MixedPrecisionTrainer)
            assert trainer.config == basic_mp_config

class TestIntegrationWithFineTuning:
    """Test integration with fine-tuning service."""
    
    @pytest.mark.asyncio
    async def test_setup_mixed_precision(self) -> Any:
        """Test mixed precision setup in fine-tuning service."""
        service = OptimizedFineTuningService()
        
        # Mock model loading
        with patch.object(service, 'load_model', return_value=Mock()):
            result = await service.setup_mixed_precision(
                model_name="test_model",
                enabled=True,
                dtype=torch.float16,
                init_scale=2**16,
                memory_efficient=True
            )
        
        assert result["enabled"] == True
        assert result["dtype"] == "torch.float16"
        assert result["init_scale"] == 2**16
        assert result["memory_efficient"] == True
        assert "memory_savings_gb" in result
        assert "mp_stats" in result
    
    @pytest.mark.asyncio
    async def test_get_mixed_precision_stats(self) -> Optional[Dict[str, Any]]:
        """Test mixed precision statistics retrieval."""
        service = OptimizedFineTuningService()
        
        # Test without initialization
        result = await service.get_mixed_precision_stats()
        assert result["success"] == False
        assert "error" in result
        
        # Test with initialization
        service.mp_trainer = Mock()
        service.mp_trainer.get_training_stats.return_value = {
            "amp_enabled": True,
            "scaler_scale": 2**16,
            "memory_saved": 2.5,
            "overflow_count": 0,
            "underflow_count": 0
        }
        
        result = await service.get_mixed_precision_stats()
        assert result["success"] == True
        assert result["enabled"] == True
        assert result["scaler_scale"] == 2**16
        assert result["memory_savings_gb"] == 2.5
        assert result["overflow_count"] == 0
        assert result["underflow_count"] == 0
    
    @pytest.mark.asyncio
    async def test_optimize_mixed_precision_settings(self) -> Any:
        """Test mixed precision settings optimization."""
        service = OptimizedFineTuningService()
        
        # Mock model loading
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, 50)]
        
        with patch.object(service, 'load_model', return_value=mock_model):
            result = await service.optimize_mixed_precision_settings(
                model_name="test_model",
                gpu_memory_gb=8.0,
                batch_size=16
            )
        
        assert result["success"] == True
        assert "should_use_mixed_precision" in result
        assert "recommendations" in result
        assert "model_params" in result
        assert "gpu_memory_gb" in result
        assert "batch_size" in result

class TestPerformanceMonitoring:
    """Test performance monitoring."""
    
    def test_performance_monitoring_decorators(self, basic_mp_config) -> Any:
        """Test performance monitoring decorators."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Test that methods have performance monitoring
        assert hasattr(trainer.forward_pass, '__wrapped__')
        assert hasattr(trainer.backward_pass, '__wrapped__')
        assert hasattr(trainer.optimizer_step, '__wrapped__')
    
    def test_memory_monitoring(self, basic_mp_config, sample_model) -> Any:
        """Test memory monitoring."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Test memory savings calculation
        memory_savings = trainer.get_memory_savings(sample_model)
        assert memory_savings >= 0
        assert isinstance(memory_savings, float)

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_config_handling(self) -> Any:
        """Test invalid configuration handling."""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(init_scale=-1)
    
    def test_cuda_unavailable_handling(self) -> Any:
        """Test CUDA unavailable handling."""
        with patch('torch.cuda.is_available', return_value=False):
            config = MixedPrecisionConfig(enabled=True)
            trainer = MixedPrecisionTrainer(config)
            
            # Should gracefully handle no CUDA
            assert trainer.scaler is None
            assert trainer.autocast_context is None
    
    def test_model_compatibility_handling(self, basic_mp_config) -> Any:
        """Test model compatibility handling."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Test with incompatible model
        incompatible_model = Mock()
        incompatible_model.supports_mixed_precision = False
        
        should_use = trainer.should_use_mixed_precision(incompatible_model)
        assert should_use == False

class TestMemoryEfficiency:
    """Test memory efficiency."""
    
    def test_memory_savings_calculation(self, basic_mp_config, sample_model) -> Any:
        """Test memory savings calculation accuracy."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Calculate expected memory savings
        total_params = sum(p.numel() for p in sample_model.parameters())
        fp32_memory = total_params * 4  # 4 bytes per parameter
        fp16_memory = total_params * 2  # 2 bytes per parameter
        expected_savings = (fp32_memory - fp16_memory) / 1024**3  # GB
        
        actual_savings = trainer.get_memory_savings(sample_model)
        
        # Allow small tolerance for floating point precision
        assert abs(actual_savings - expected_savings) < 0.01
    
    def test_memory_efficient_settings(self) -> Any:
        """Test memory efficient settings."""
        config = MixedPrecisionConfig(
            enabled=True,
            memory_efficient=True,
            cache_enabled=False
        )
        
        assert config.memory_efficient == True
        assert config.cache_enabled == False

class TestTrainingStability:
    """Test training stability."""
    
    @pytest.mark.asyncio
    async def test_gradient_scaling(self, basic_mp_config, sample_model) -> Any:
        """Test gradient scaling for stability."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Setup model and optimizer
        sample_model.train()
        optimizer = torch.optim.Adam(sample_model.parameters())
        
        # Create sample data
        inputs = torch.randn(32, 100)
        targets = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Multiple forward/backward passes to test stability
        for i in range(10):
            outputs = trainer.forward_pass(sample_model, inputs)
            loss = criterion(outputs, targets)
            
            backward_stats = trainer.backward_pass(loss, optimizer, trainer.scaler)
            optimizer_stats = trainer.optimizer_step(optimizer, trainer.scaler)
            
            # Check that scaler scale is reasonable
            assert backward_stats["scaler_scale"] > 0
            assert backward_stats["scaler_scale"] <= 2**16
    
    def test_overflow_underflow_handling(self, basic_mp_config) -> Any:
        """Test overflow and underflow handling."""
        trainer = MixedPrecisionTrainer(basic_mp_config)
        
        # Simulate overflow/underflow
        trainer.training_stats["overflow_count"] = 5
        trainer.training_stats["underflow_count"] = 3
        
        stats = trainer.get_training_stats()
        
        assert stats["overflow_count"] == 5
        assert stats["underflow_count"] == 3

# Run tests
match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 