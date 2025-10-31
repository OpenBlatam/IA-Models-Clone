from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import tempfile
import time
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_debugging_optimization import (
from error_handling_debugging import ErrorHandlingDebuggingSystem, ErrorSeverity, ErrorCategory
from training_logging_system import TrainingLogger, create_training_logger
from robust_operations import RobustOperations
import structlog
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for PyTorch Debugging and Optimization System

This test suite covers:
- autograd.detect_anomaly() functionality
- Gradient checking and analysis
- Memory profiling and leak detection
- Performance optimization techniques
- Integration with robust operations
- Decorator functionality
- Error handling and recovery
"""



# Add the current directory to the path to import our modules
sys.path.append(str(Path(__file__).parent))

    PyTorchDebugger,
    PyTorchOptimizer,
    DebugMode,
    OptimizationMode,
    DebugMetrics,
    OptimizationMetrics,
    debug_operation,
    optimize_model
)

# Configure logging for tests
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for debugging."""
    
    def __init__(self, input_size: int = 5, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        
    def forward(self, x) -> Any:
        return F.softmax(self.fc1(x), dim=1)


class ProblematicTestModel(nn.Module):
    """Test model with intentional issues for debugging."""
    
    def __init__(self, input_size: int = 5, num_classes: int = 2):
        
    """__init__ function."""
super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        
    def forward(self, x) -> Any:
        # Return NaN values for testing
        return torch.tensor([[float('nan'), float('inf')]])


class TestPyTorchDebugger:
    """Test suite for PyTorchDebugger."""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path) -> Any:
        """Create temporary log directory."""
        log_dir = tmp_path / "test_debug_logs"
        log_dir.mkdir()
        return str(log_dir)
    
    @pytest.fixture
    def error_system(self) -> Any:
        """Create error handling system."""
        return ErrorHandlingDebuggingSystem({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
    
    @pytest.fixture
    def training_logger(self, temp_log_dir) -> Any:
        """Create training logger."""
        return create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
    
    @pytest.fixture
    def debugger(self, error_system, training_logger) -> Any:
        """Create PyTorchDebugger instance."""
        return PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.FULL_DEBUG
        )
    
    @pytest.fixture
    def sample_debug_metrics(self) -> Any:
        """Create sample debug metrics."""
        return DebugMetrics(
            mode=DebugMode.ANOMALY_DETECTION,
            execution_time=1.5,
            memory_usage=512.0,
            gpu_memory=256.0,
            gradient_norm=2.5,
            gradient_anomalies=["Gradient explosion detected"],
            timestamp="2024-01-01T12:00:00"
        )
    
    @pytest.fixture
    def sample_optimization_metrics(self) -> Any:
        """Create sample optimization metrics."""
        return OptimizationMetrics(
            mode=OptimizationMode.AMP,
            execution_time=0.8,
            memory_usage=384.0,
            gpu_memory=192.0,
            speedup_factor=1.5,
            memory_savings=0.25,
            amp_enabled=True,
            timestamp="2024-01-01T12:00:00"
        )
    
    def test_initialization(self, debugger) -> Any:
        """Test PyTorchDebugger initialization."""
        assert debugger.error_system is not None
        assert debugger.training_logger is not None
        assert debugger.debug_mode == DebugMode.FULL_DEBUG
        assert debugger.anomaly_detection_enabled is True
        assert debugger.profiling_enabled is True
        assert debugger.memory_profiling_enabled is True
        assert debugger.gradient_checking_enabled is True
        assert debugger.debug_metrics == []
        assert debugger.optimization_metrics == []
    
    def test_debug_mode_setup(self, error_system, training_logger) -> Any:
        """Test different debug mode setups."""
        # Test anomaly detection only
        debugger = PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.ANOMALY_DETECTION
        )
        
        assert debugger.anomaly_detection_enabled is True
        assert debugger.profiling_enabled is False
        assert debugger.memory_profiling_enabled is False
        assert debugger.gradient_checking_enabled is False
        
        # Test profiling only
        debugger = PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.PROFILING
        )
        
        assert debugger.anomaly_detection_enabled is False
        assert debugger.profiling_enabled is True
        assert debugger.memory_profiling_enabled is False
        assert debugger.gradient_checking_enabled is False
    
    def test_debug_context_success(self, debugger) -> Any:
        """Test successful debug context execution."""
        with debugger.debug_context("test_operation"):
            # Simple operation
            x = torch.randn(10, 5)
            y = torch.randn(10, 2)
            result = torch.matmul(x, y)
        
        # Check that debug metrics were recorded
        assert len(debugger.debug_metrics) == 1
        assert debugger.debug_metrics[0].mode == DebugMode.FULL_DEBUG
        assert debugger.debug_metrics[0].execution_time > 0
    
    def test_debug_context_error(self, debugger) -> Any:
        """Test debug context with error handling."""
        with pytest.raises(RuntimeError):
            with debugger.debug_context("test_error_operation"):
                raise RuntimeError("Test error")
        
        # Check that error was tracked
        assert len(debugger.error_system.error_tracker.errors) > 0
    
    def test_gradient_checking(self, debugger) -> Any:
        """Test gradient checking functionality."""
        model = SimpleTestModel()
        data = torch.randn(16, 5)
        target = torch.randint(0, 2, (16,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Check gradients
        gradient_info = debugger.check_gradients(model, loss)
        
        assert gradient_info["gradient_norm"] is not None
        assert gradient_info["gradient_norm"] > 0
        assert gradient_info["loss_value"] == loss.item()
        assert "parameter_stats" in gradient_info
        assert len(gradient_info["parameter_stats"]) > 0
    
    def test_gradient_checking_with_anomalies(self, debugger) -> Any:
        """Test gradient checking with anomalies."""
        model = ProblematicTestModel()
        data = torch.randn(16, 5)
        target = torch.randint(0, 2, (16,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass (will produce NaN)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Check gradients
        gradient_info = debugger.check_gradients(model, loss)
        
        # Should detect NaN gradients
        assert len(gradient_info["gradient_anomalies"]) > 0
        assert any("NaN" in anomaly for anomaly in gradient_info["gradient_anomalies"])
    
    def test_memory_profiling(self, debugger) -> Any:
        """Test memory profiling functionality."""
        memory_info = debugger.profile_memory("test_operation")
        
        assert memory_info["operation"] == "test_operation"
        assert memory_info["cpu_memory"] > 0
        assert memory_info["timestamp"] is not None
        assert "memory_leaks" in memory_info
        
        # Check that snapshot was stored
        assert len(debugger.memory_snapshots) == 1
        assert debugger.memory_snapshots[0]["operation"] == "test_operation"
    
    def test_memory_leak_detection(self, debugger) -> Any:
        """Test memory leak detection."""
        # First memory profile
        memory_info1 = debugger.profile_memory("operation_1")
        
        # Simulate memory increase
        large_tensor = torch.randn(1000, 1000)
        
        # Second memory profile
        memory_info2 = debugger.profile_memory("operation_2")
        
        # Check for memory leaks
        if memory_info2["memory_leaks"]:
            assert len(memory_info2["memory_leaks"]) > 0
    
    def test_model_optimization_amp(self, debugger) -> Any:
        """Test model optimization with AMP."""
        model = SimpleTestModel()
        original_model = model
        
        optimized_model = debugger.optimize_model(model, OptimizationMode.AMP)
        
        # Check that optimization metrics were recorded
        assert len(debugger.optimization_metrics) == 1
        assert debugger.optimization_metrics[0].mode == OptimizationMode.AMP
        assert debugger.optimization_metrics[0].amp_enabled is True
        assert debugger.optimization_metrics[0].execution_time > 0
    
    def test_model_optimization_compilation(self, debugger) -> Any:
        """Test model optimization with compilation."""
        model = SimpleTestModel()
        
        optimized_model = debugger.optimize_model(model, OptimizationMode.COMPILATION)
        
        # Check that optimization metrics were recorded
        assert len(debugger.optimization_metrics) == 1
        assert debugger.optimization_metrics[0].mode == OptimizationMode.COMPILATION
        assert debugger.optimization_metrics[0].compilation_time is not None
    
    def test_model_optimization_memory_efficient(self, debugger) -> Any:
        """Test memory-efficient model optimization."""
        model = SimpleTestModel()
        
        optimized_model = debugger.optimize_model(model, OptimizationMode.MEMORY_EFFICIENT)
        
        # Check that optimization metrics were recorded
        assert len(debugger.optimization_metrics) == 1
        assert debugger.optimization_metrics[0].mode == OptimizationMode.MEMORY_EFFICIENT
    
    def test_model_optimization_full(self, debugger) -> Any:
        """Test full model optimization."""
        model = SimpleTestModel()
        
        optimized_model = debugger.optimize_model(model, OptimizationMode.FULL_OPTIMIZATION)
        
        # Check that optimization metrics were recorded
        assert len(debugger.optimization_metrics) == 1
        assert debugger.optimization_metrics[0].mode == OptimizationMode.FULL_OPTIMIZATION
        assert debugger.optimization_metrics[0].amp_enabled is True
    
    def test_model_optimization_error(self, debugger) -> Any:
        """Test model optimization error handling."""
        # Create a model that will cause optimization to fail
        model = Mock(spec=nn.Module)
        model.half.side_effect = Exception("Optimization failed")
        
        # Should return original model on error
        result_model = debugger.optimize_model(model, OptimizationMode.AMP)
        assert result_model is model
        
        # Check that error was tracked
        assert len(debugger.error_system.error_tracker.errors) > 0
    
    def test_get_debug_summary(self, debugger) -> Optional[Dict[str, Any]]:
        """Test debug summary generation."""
        # Add some debug metrics
        for i in range(5):
            metrics = DebugMetrics(
                mode=DebugMode.ANOMALY_DETECTION,
                execution_time=1.0 + i * 0.1,
                memory_usage=100.0 + i * 10.0,
                timestamp="2024-01-01T12:00:00"
            )
            debugger.debug_metrics.append(metrics)
        
        # Add some optimization metrics
        for i in range(3):
            opt_metrics = OptimizationMetrics(
                mode=OptimizationMode.AMP,
                execution_time=0.5 + i * 0.1,
                memory_usage=50.0 + i * 5.0,
                timestamp="2024-01-01T12:00:00"
            )
            debugger.optimization_metrics.append(opt_metrics)
        
        summary = debugger.get_debug_summary()
        
        assert summary["debug_mode"] == DebugMode.FULL_DEBUG.value
        assert summary["total_operations"] == 5
        assert summary["total_optimizations"] == 3
        assert "avg_execution_time" in summary
        assert "avg_optimization_time" in summary
        assert summary["anomaly_detection_enabled"] is True
        assert summary["profiling_enabled"] is True
    
    def test_get_debug_summary_empty(self, debugger) -> Optional[Dict[str, Any]]:
        """Test debug summary with no metrics."""
        summary = debugger.get_debug_summary()
        
        assert "error" in summary
        assert summary["error"] == "No debug metrics available"
    
    def test_cleanup(self, debugger) -> Any:
        """Test cleanup functionality."""
        # Add some data
        debugger.debug_metrics.append(DebugMetrics(
            mode=DebugMode.ANOMALY_DETECTION,
            execution_time=1.0,
            memory_usage=100.0
        ))
        
        debugger.memory_snapshots.append({"test": "data"})
        debugger.profiler_data["test"] = "data"
        
        # Cleanup
        debugger.cleanup()
        
        # Check that data was cleared
        assert len(debugger.debug_metrics) == 0
        assert len(debugger.optimization_metrics) == 0
        assert len(debugger.memory_snapshots) == 0
        assert len(debugger.profiler_data) == 0


class TestPyTorchOptimizer:
    """Test suite for PyTorchOptimizer."""
    
    @pytest.fixture
    def optimizer(self, debugger) -> Any:
        """Create PyTorchOptimizer instance."""
        return PyTorchOptimizer(debugger)
    
    @pytest.fixture
    def sample_dataloader(self) -> Any:
        """Create sample dataloader."""
        data = torch.randn(100, 5)
        target = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, target)
        return DataLoader(dataset, batch_size=16, shuffle=True)
    
    def test_initialization(self, optimizer) -> Any:
        """Test PyTorchOptimizer initialization."""
        assert optimizer.debugger is not None
    
    def test_optimize_training_loop_none(self, optimizer, sample_dataloader) -> Any:
        """Test training loop optimization with no optimization."""
        model = SimpleTestModel()
        optimizer_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        result = optimizer.optimize_training_loop(
            model=model,
            dataloader=sample_dataloader,
            optimizer=optimizer_opt,
            criterion=criterion,
            epochs=1,
            optimization_mode=OptimizationMode.NONE
        )
        
        assert result["total_time"] > 0
        assert len(result["training_metrics"]) > 0
        assert result["model_optimized"] is False
        assert "optimization_metrics" in result
    
    def test_optimize_training_loop_amp(self, optimizer, sample_dataloader) -> Any:
        """Test training loop optimization with AMP."""
        model = SimpleTestModel()
        optimizer_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        result = optimizer.optimize_training_loop(
            model=model,
            dataloader=sample_dataloader,
            optimizer=optimizer_opt,
            criterion=criterion,
            epochs=1,
            optimization_mode=OptimizationMode.AMP
        )
        
        assert result["total_time"] > 0
        assert len(result["training_metrics"]) > 0
        assert result["model_optimized"] is True
        assert result["optimization_metrics"]["amp_enabled"] is True
    
    def test_optimize_training_loop_error(self, optimizer, sample_dataloader) -> Any:
        """Test training loop optimization with error."""
        # Create problematic model
        model = ProblematicTestModel()
        optimizer_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        with pytest.raises(Exception):
            optimizer.optimize_training_loop(
                model=model,
                dataloader=sample_dataloader,
                optimizer=optimizer_opt,
                criterion=criterion,
                epochs=1,
                optimization_mode=OptimizationMode.NONE
            )
        
        # Check that error was tracked
        assert len(optimizer.debugger.error_system.error_tracker.errors) > 0
    
    def test_benchmark_optimizations(self, optimizer, sample_dataloader) -> Any:
        """Test optimization benchmarking."""
        model = SimpleTestModel()
        optimizer_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        results = optimizer.benchmark_optimizations(
            model=model,
            dataloader=sample_dataloader,
            optimizer=optimizer_opt,
            criterion=criterion,
            epochs=1
        )
        
        # Check that all optimization modes were tested
        expected_modes = [
            OptimizationMode.NONE.value,
            OptimizationMode.AMP.value,
            OptimizationMode.COMPILATION.value,
            OptimizationMode.MEMORY_EFFICIENT.value,
            OptimizationMode.FULL_OPTIMIZATION.value
        ]
        
        for mode in expected_modes:
            assert mode in results
            assert "total_time" in results[mode] or "error" in results[mode]


class TestDecorators:
    """Test suite for decorators."""
    
    @pytest.fixture
    def error_system(self) -> Any:
        """Create error handling system."""
        return ErrorHandlingDebuggingSystem({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
    
    def test_debug_operation_decorator_success(self, error_system) -> Any:
        """Test debug_operation decorator with successful execution."""
        debugger = PyTorchDebugger(error_system, debug_mode=DebugMode.ANOMALY_DETECTION)
        
        @debug_operation(debug_mode=DebugMode.ANOMALY_DETECTION)
        def test_function(debugger=debugger) -> Any:
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert len(debugger.debug_metrics) == 1
        assert debugger.debug_metrics[0].mode == DebugMode.ANOMALY_DETECTION
    
    def test_debug_operation_decorator_error(self, error_system) -> Any:
        """Test debug_operation decorator with error."""
        debugger = PyTorchDebugger(error_system, debug_mode=DebugMode.ANOMALY_DETECTION)
        
        @debug_operation(debug_mode=DebugMode.ANOMALY_DETECTION)
        def test_function(debugger=debugger) -> Any:
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_function()
        
        # Check that error was tracked
        assert len(debugger.error_system.error_tracker.errors) > 0
    
    def test_optimize_model_decorator(self, error_system) -> Any:
        """Test optimize_model decorator."""
        debugger = PyTorchDebugger(error_system, debug_mode=DebugMode.NONE)
        
        @optimize_model(optimization_mode=OptimizationMode.AMP)
        def test_function(model, debugger=debugger) -> Any:
            return model
        
        model = SimpleTestModel()
        result = test_function(model=model)
        
        # Check that model was optimized
        assert len(debugger.optimization_metrics) == 1
        assert debugger.optimization_metrics[0].mode == OptimizationMode.AMP
        assert debugger.optimization_metrics[0].amp_enabled is True
    
    def test_optimize_model_decorator_no_model(self, error_system) -> Any:
        """Test optimize_model decorator with no model parameter."""
        debugger = PyTorchDebugger(error_system, debug_mode=DebugMode.NONE)
        
        @optimize_model(optimization_mode=OptimizationMode.AMP)
        def test_function(debugger=debugger) -> Any:
            return "no_model"
        
        result = test_function()
        
        # Should work without model
        assert result == "no_model"
        assert len(debugger.optimization_metrics) == 0


class TestIntegration:
    """Integration tests for PyTorch debugging system."""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path) -> Any:
        """Create temporary log directory."""
        log_dir = tmp_path / "integration_logs"
        log_dir.mkdir()
        return str(log_dir)
    
    def test_integration_with_robust_operations(self, temp_log_dir) -> Any:
        """Test integration with robust operations."""
        # Create systems
        error_system = ErrorHandlingDebuggingSystem({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
        
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        robust_ops = RobustOperations({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
        
        debugger = PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.FULL_DEBUG
        )
        
        # Create model and data
        model = SimpleTestModel()
        data = torch.randn(32, 5)
        target = torch.randint(0, 2, (32,))
        
        # Test robust inference with debugging
        with debugger.debug_context("robust_inference"):
            result = robust_ops.model_inference.safe_inference(
                model=model,
                input_data=data,
                device=torch.device('cpu')
            )
        
        assert result.success is True
        assert len(debugger.debug_metrics) == 1
        assert debugger.debug_metrics[0].execution_time > 0
    
    def test_integration_with_training_logger(self, temp_log_dir) -> Any:
        """Test integration with training logger."""
        # Create systems
        error_system = ErrorHandlingDebuggingSystem({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
        
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        debugger = PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.FULL_DEBUG
        )
        
        # Create model and data
        model = SimpleTestModel()
        data = torch.randn(32, 5)
        target = torch.randint(0, 2, (32,))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test training with debugging
        with debugger.debug_context("training_step"):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Check that metrics were logged
        assert len(debugger.debug_metrics) == 1
        assert len(training_logger.training_metrics) == 0  # No training metrics logged in this test
    
    def test_full_training_cycle(self, temp_log_dir) -> Any:
        """Test full training cycle with debugging and optimization."""
        # Create systems
        error_system = ErrorHandlingDebuggingSystem({
            "max_errors": 100,
            "enable_persistence": False,
            "enable_profiling": False
        })
        
        training_logger = create_training_logger({
            "log_dir": temp_log_dir,
            "enable_rich": False
        })
        
        debugger = PyTorchDebugger(
            error_system=error_system,
            training_logger=training_logger,
            debug_mode=DebugMode.FULL_DEBUG
        )
        
        optimizer = PyTorchOptimizer(debugger)
        
        # Create model and data
        model = SimpleTestModel()
        data = torch.randn(100, 5)
        target = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, target)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Run optimized training
        result = optimizer.optimize_training_loop(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer_opt,
            criterion=criterion,
            epochs=1,
            optimization_mode=OptimizationMode.AMP
        )
        
        # Check results
        assert result["total_time"] > 0
        assert len(result["training_metrics"]) > 0
        assert result["model_optimized"] is True
        
        # Check debug metrics
        assert len(debugger.debug_metrics) > 0
        assert len(debugger.optimization_metrics) > 0
        
        # Get summary
        summary = debugger.get_debug_summary()
        assert summary["total_operations"] > 0
        assert summary["total_optimizations"] > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 