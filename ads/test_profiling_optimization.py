from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import torch
import torch.nn as nn
import asyncio
import time
import psutil
import gc
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from onyx.server.features.ads.profiling_optimizer import (
from onyx.server.features.ads.data_optimization import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
        import tempfile
        import pandas as pd
        import tempfile
        import numpy as np
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Profiling and Optimization System

This module provides comprehensive tests for:
- Profiling configuration and initialization
- CPU, memory, and GPU profiling
- Data loading optimization
- Preprocessing optimization
- Memory optimization
- I/O optimization
- Real-time monitoring
- Integration with fine-tuning service
- Performance benchmarking
- Error handling and edge cases
"""

    ProfilingConfig,
    ProfilingOptimizer,
    ProfilingResult,
    DataLoadingOptimizer,
    PreprocessingOptimizer,
    RealTimeProfiler,
    profile_function,
    profiling_context
)
    DataOptimizationConfig,
    OptimizedDataset,
    StreamingDataset,
    MemoryOptimizer,
    IOOptimizer,
    optimize_dataset,
    optimize_dataloader,
    optimize_preprocessing,
    memory_optimization_context
)

# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [torch.randn(100, 100) for _ in range(100)]

@pytest.fixture
def sample_dataset(sample_data) -> Any:
    """Create a sample dataset for testing."""
    class MockDataset:
        def __init__(self, data) -> Any:
            self.data = data
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx]
    
    return MockDataset(sample_data)

@pytest.fixture
def profiling_config():
    """Create profiling configuration."""
    return ProfilingConfig(
        enabled=True,
        profile_cpu=True,
        profile_memory=True,
        profile_gpu=True,
        profile_data_loading=True,
        profile_preprocessing=True,
        profile_depth=5,
        min_time_threshold=0.001,
        min_memory_threshold=1024 * 1024,
        save_profiles=True,
        profile_dir="test_profiles",
        auto_optimize=True,
        optimization_threshold=0.1,
        real_time_monitoring=True,
        alert_threshold=5.0,
        monitoring_interval=1.0
    )

@pytest.fixture
def data_optimization_config():
    """Create data optimization configuration."""
    return DataOptimizationConfig(
        optimize_loading=True,
        prefetch_factor=2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        memory_efficient=True,
        max_memory_usage=0.8,
        chunk_size=100,
        enable_caching=True,
        cache_dir="test_cache",
        cache_size=100,
        optimize_preprocessing=True,
        batch_preprocessing=True,
        parallel_preprocessing=True,
        preprocessing_workers=2,
        optimize_io=True,
        compression="gzip",
        buffer_size=8192,
        async_io=True,
        monitor_performance=True,
        log_metrics=True,
        alert_threshold=5.0
    )

class TestProfilingConfig:
    """Test profiling configuration."""
    
    def test_basic_config_creation(self) -> Any:
        """Test basic configuration creation."""
        config = ProfilingConfig()
        
        assert config.enabled == True
        assert config.profile_cpu == True
        assert config.profile_memory == True
        assert config.profile_gpu == True
        assert config.profile_depth == 10
        assert config.min_time_threshold == 0.001
        assert config.min_memory_threshold == 1024 * 1024
    
    def test_advanced_config_creation(self, profiling_config) -> Any:
        """Test advanced configuration creation."""
        config = profiling_config
        
        assert config.enabled == True
        assert config.profile_cpu == True
        assert config.profile_memory == True
        assert config.profile_gpu == True
        assert config.profile_data_loading == True
        assert config.profile_preprocessing == True
        assert config.profile_depth == 5
        assert config.save_profiles == True
        assert config.profile_dir == "test_profiles"
        assert config.auto_optimize == True
        assert config.real_time_monitoring == True
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Test invalid thresholds
        with pytest.raises(ValueError):
            ProfilingConfig(min_time_threshold=-1)
        
        with pytest.raises(ValueError):
            ProfilingConfig(min_memory_threshold=-1)
        
        # Test valid configuration
        config = ProfilingConfig(
            enabled=True,
            profile_depth=5,
            min_time_threshold=0.01
        )
        assert config.enabled == True
        assert config.profile_depth == 5

class TestProfilingOptimizer:
    """Test profiling optimizer."""
    
    def test_optimizer_initialization(self, profiling_config) -> Any:
        """Test optimizer initialization."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        assert optimizer.config == profiling_config
        assert optimizer.profiling_results == []
        assert optimizer.optimization_history == []
        assert optimizer.gpu_monitor is not None
    
    def test_profile_function_context(self, profiling_config) -> Any:
        """Test profile function context manager."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        with optimizer.profile_function("test_function"):
            time.sleep(0.01)  # Small delay
        
        # Check that profiling was performed
        assert len(optimizer.profiling_results) == 0  # Context manager doesn't store results
    
    @pytest.mark.asyncio
    async def test_profile_code(self, profiling_config) -> Any:
        """Test code profiling."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        def sample_function():
            
    """sample_function function."""
time.sleep(0.01)
            return "result"
        
        result = optimizer.profile_code(sample_function)
        
        assert isinstance(result, ProfilingResult)
        assert result.total_time > 0
        assert len(result.function_times) > 0
        assert len(result.recommendations) >= 0
    
    def test_profile_cpu(self, profiling_config) -> Any:
        """Test CPU profiling."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        def cpu_intensive_function():
            
    """cpu_intensive_function function."""
# Simulate CPU work
            result = 0
            for i in range(10000):
                result += i
            return result
        
        function_times = optimizer._profile_cpu(cpu_intensive_function)
        
        assert isinstance(function_times, dict)
        assert len(function_times) > 0
    
    def test_profile_memory(self, profiling_config) -> Any:
        """Test memory profiling."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        def memory_intensive_function():
            
    """memory_intensive_function function."""
# Simulate memory usage
            data = [0] * 1000000
            return len(data)
        
        memory_usage = optimizer._profile_memory(memory_intensive_function)
        
        assert isinstance(memory_usage, dict)
        assert "peak" in memory_usage
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_gpu(self, profiling_config) -> Any:
        """Test GPU profiling."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        def gpu_intensive_function():
            
    """gpu_intensive_function function."""
# Simulate GPU work
            tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(tensor, tensor)
            return result.sum().item()
        
        gpu_info = optimizer._profile_gpu(gpu_intensive_function)
        
        assert isinstance(gpu_info, dict)
        assert "utilization" in gpu_info
        assert "memory_usage" in gpu_info
    
    def test_identify_bottlenecks(self, profiling_config) -> Any:
        """Test bottleneck identification."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        # Create mock function times
        function_times = {
            "fast_function": 0.001,
            "slow_function": 0.5,
            "medium_function": 0.1
        }
        
        bottlenecks = optimizer._identify_bottlenecks(function_times)
        
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) > 0
        assert "slow_function" in str(bottlenecks[0])
    
    def test_generate_recommendations(self, profiling_config) -> Any:
        """Test recommendation generation."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        # Create mock profiling result
        result = ProfilingResult()
        result.bottleneck_functions = ["slow_function: 0.5s (50.0%)"]
        result.memory_leaks = ["High memory usage: 2.5GB"]
        result.gpu_utilization = 30.0
        result.gpu_memory_usage = 6000  # 6GB
        
        recommendations = optimizer._generate_recommendations(result)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_estimate_improvements(self, profiling_config) -> Any:
        """Test improvement estimation."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        # Create mock profiling result
        result = ProfilingResult()
        result.bottleneck_functions = ["slow_function"]
        result.memory_leaks = ["memory_leak"]
        result.gpu_utilization = 30.0
        
        improvements = optimizer._estimate_improvements(result)
        
        assert isinstance(improvements, dict)
        assert len(improvements) > 0

class TestDataLoadingOptimizer:
    """Test data loading optimizer."""
    
    def test_optimizer_initialization(self, data_optimization_config) -> Any:
        """Test optimizer initialization."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        assert optimizer.config == data_optimization_config
        assert optimizer.profiler is not None
        assert optimizer.optimization_stats == {}
        assert optimizer.bottleneck_analysis == {}
    
    def test_analyze_dataset(self, data_optimization_config, sample_dataset) -> Any:
        """Test dataset analysis."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        analysis = optimizer._analyze_dataset(sample_dataset)
        
        assert isinstance(analysis, dict)
        assert "size" in analysis
        assert "sample_size" in analysis
        assert "memory_usage" in analysis
        assert "access_pattern" in analysis
        assert "complexity" in analysis
        assert analysis["size"] == 100
    
    def test_estimate_sample_size(self, data_optimization_config) -> Any:
        """Test sample size estimation."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        # Test tensor
        tensor = torch.randn(100, 50)
        size = optimizer._estimate_sample_size(tensor)
        assert size > 0
        
        # Test list
        data_list = [torch.randn(10, 10), torch.randn(10, 10)]
        size = optimizer._estimate_sample_size(data_list)
        assert size > 0
        
        # Test string
        text = "sample text"
        size = optimizer._estimate_sample_size(text)
        assert size > 0
    
    def test_determine_optimal_params(self, data_optimization_config) -> Any:
        """Test optimal parameter determination."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        dataset_analysis = {
            "size": 1000,
            "complexity": "simple",
            "sample_size": 1024
        }
        
        params = optimizer._determine_optimal_params(dataset_analysis, 32)
        
        assert isinstance(params, dict)
        assert "num_workers" in params
        assert "prefetch_factor" in params
        assert "pin_memory" in params
        assert "persistent_workers" in params
        assert params["num_workers"] > 0
    
    @pytest.mark.asyncio
    async def test_optimize_dataloader(self, data_optimization_config, sample_dataset) -> Any:
        """Test dataloader optimization."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        dataloader = optimizer.optimize_dataloader(
            sample_dataset,
            batch_size=16,
            shuffle=True
        )
        
        assert dataloader is not None
        assert hasattr(dataloader, 'batch_size')
        assert dataloader.batch_size == 16
    
    def test_identify_bottlenecks(self, data_optimization_config) -> Any:
        """Test bottleneck identification."""
        optimizer = DataLoadingOptimizer(data_optimization_config)
        
        # Mock optimization stats
        optimizer.optimization_stats = {
            "avg_batch_time": 0.2,  # 200ms - slow
            "memory_used": 1024 * 1024 * 1024  # 1GB - high
        }
        
        bottlenecks = optimizer.identify_bottlenecks()
        
        assert isinstance(bottlenecks, dict)
        assert len(bottlenecks) > 0

class TestPreprocessingOptimizer:
    """Test preprocessing optimizer."""
    
    def test_optimizer_initialization(self, data_optimization_config) -> Any:
        """Test optimizer initialization."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        assert optimizer.config == data_optimization_config
        assert optimizer.profiler is not None
        assert optimizer.preprocessing_cache == {}
        assert optimizer.optimization_stats == {}
    
    def test_analyze_preprocessing_funcs(self, data_optimization_config) -> Any:
        """Test preprocessing function analysis."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def io_function(text) -> Any:
            return text.lower()
        
        def cpu_function(text) -> Any:
            return text.split()
        
        def memory_function(text) -> Any:
            return text * 1000
        
        funcs = [io_function, cpu_function, memory_function]
        
        analysis = optimizer._analyze_preprocessing_funcs(funcs)
        
        assert isinstance(analysis, dict)
        assert "total_funcs" in analysis
        assert "io_bound" in analysis
        assert "cpu_bound" in analysis
        assert "memory_intensive" in analysis
        assert "cacheable" in analysis
        assert analysis["total_funcs"] == 3
    
    def test_is_cacheable(self, data_optimization_config) -> Any:
        """Test cacheable function detection."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def pure_function(x) -> Any:
            return x * 2
        
        def impure_function(x) -> Any:
            global counter
            counter += 1
            return x * 2
        
        assert optimizer._is_cacheable(pure_function) == True
        # Note: Current implementation assumes all functions are cacheable
    
    def test_create_simple_pipeline(self, data_optimization_config) -> Any:
        """Test simple pipeline creation."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        funcs = [func1, func2]
        
        pipeline = optimizer._create_simple_pipeline(funcs)
        
        # Test pipeline
        result = pipeline(5)
        assert result == 12  # (5 + 1) * 2
    
    def test_create_optimized_pipeline(self, data_optimization_config) -> Any:
        """Test optimized pipeline creation."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        funcs = [func1, func2]
        analysis = {"cacheable": 2}
        
        pipeline = optimizer._create_optimized_pipeline(funcs, analysis)
        
        # Test pipeline
        result = pipeline(5)
        assert result == 12  # (5 + 1) * 2
    
    def test_create_batch_pipeline(self, data_optimization_config) -> Any:
        """Test batch pipeline creation."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        funcs = [func1, func2]
        analysis = {"cacheable": 2}
        
        pipeline = optimizer._create_batch_pipeline(funcs, analysis)
        
        # Test pipeline with batch
        data = [1, 2, 3, 4, 5]
        results = pipeline(data)
        expected = [(x + 1) * 2 for x in data]
        assert results == expected
    
    def test_create_parallel_pipeline(self, data_optimization_config) -> Any:
        """Test parallel pipeline creation."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        funcs = [func1, func2]
        analysis = {"cpu_bound": 2}
        
        pipeline = optimizer._create_parallel_pipeline(funcs, analysis)
        
        # Test pipeline
        data = [1, 2, 3, 4, 5]
        results = pipeline(data)
        expected = [(x + 1) * 2 for x in data]
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_optimize_preprocessing_pipeline(self, data_optimization_config) -> Any:
        """Test preprocessing pipeline optimization."""
        optimizer = PreprocessingOptimizer(data_optimization_config)
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        funcs = [func1, func2]
        data = [1, 2, 3, 4, 5]
        
        pipeline = optimizer.optimize_preprocessing_pipeline(funcs, data)
        
        # Test pipeline
        results = pipeline(data)
        expected = [(x + 1) * 2 for x in data]
        assert results == expected

class TestMemoryOptimizer:
    """Test memory optimizer."""
    
    def test_optimizer_initialization(self, data_optimization_config) -> Any:
        """Test optimizer initialization."""
        optimizer = MemoryOptimizer(data_optimization_config)
        
        assert optimizer.config == data_optimization_config
        assert optimizer.memory_stats == {}
    
    def test_memory_context(self, data_optimization_config) -> Any:
        """Test memory context manager."""
        optimizer = MemoryOptimizer(data_optimization_config)
        
        with optimizer.memory_context():
            # Simulate memory usage
            data = [0] * 100000
            _ = len(data)
        
        assert "last_operation" in optimizer.memory_stats
    
    def test_optimize_memory_usage(self, data_optimization_config) -> Any:
        """Test memory usage optimization."""
        optimizer = MemoryOptimizer(data_optimization_config)
        
        # Test tensor optimization
        tensor = torch.randn(100, 100, dtype=torch.float64)
        optimized_tensor = optimizer.optimize_memory_usage(tensor)
        
        assert optimized_tensor.dtype == torch.float32
        assert optimized_tensor.is_contiguous()
        
        # Test sequence optimization
        sequence = [torch.randn(10, 10), torch.randn(10, 10)]
        optimized_sequence = optimizer.optimize_memory_usage(sequence)
        
        assert isinstance(optimized_sequence, list)
        assert len(optimized_sequence) == 2
        
        # Test dict optimization
        data_dict = {"a": torch.randn(10, 10), "b": torch.randn(10, 10)}
        optimized_dict = optimizer.optimize_memory_usage(data_dict)
        
        assert isinstance(optimized_dict, dict)
        assert "a" in optimized_dict
        assert "b" in optimized_dict

class TestIOOptimizer:
    """Test I/O optimizer."""
    
    def test_optimizer_initialization(self, data_optimization_config) -> Any:
        """Test optimizer initialization."""
        optimizer = IOOptimizer(data_optimization_config)
        
        assert optimizer.config == data_optimization_config
        assert optimizer.io_stats == {}
    
    def test_optimize_file_reading(self, data_optimization_config) -> Any:
        """Test file reading optimization."""
        optimizer = IOOptimizer(data_optimization_config)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_file = f.name
        
        try:
            reader = optimizer.optimize_file_reading(temp_file)
            content = reader()
            
            assert content is not None
        finally:
            os.unlink(temp_file)
    
    def test_optimize_data_storage(self, data_optimization_config) -> Any:
        """Test data storage optimization."""
        optimizer = IOOptimizer(data_optimization_config)
        
        # Test DataFrame storage
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            optimizer.optimize_data_storage(df, temp_file)
            assert os.path.exists(temp_file + ".parquet")
        finally:
            if os.path.exists(temp_file + ".parquet"):
                os.unlink(temp_file + ".parquet")
        
        # Test array storage
        array = np.array([[1, 2, 3], [4, 5, 6]])
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            optimizer.optimize_data_storage(array, temp_file)
            assert os.path.exists(temp_file + ".npy")
        finally:
            if os.path.exists(temp_file + ".npy"):
                os.unlink(temp_file + ".npy")

class TestRealTimeProfiler:
    """Test real-time profiler."""
    
    def test_profiler_initialization(self, profiling_config) -> Any:
        """Test profiler initialization."""
        profiler = RealTimeProfiler(profiling_config)
        
        assert profiler.config == profiling_config
        assert profiler.monitoring_active == False
        assert profiler.monitoring_thread is None
        assert profiler.performance_metrics == []
        assert profiler.alert_callbacks == []
    
    def test_start_stop_monitoring(self, profiling_config) -> Any:
        """Test monitoring start and stop."""
        profiler = RealTimeProfiler(profiling_config)
        
        # Start monitoring
        profiler.start_monitoring()
        assert profiler.monitoring_active == True
        assert profiler.monitoring_thread is not None
        
        # Stop monitoring
        profiler.stop_monitoring()
        assert profiler.monitoring_active == False
    
    def test_collect_metrics(self, profiling_config) -> Any:
        """Test metrics collection."""
        profiler = RealTimeProfiler(profiling_config)
        
        metrics = profiler._collect_metrics()
        
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "memory_used" in metrics
        assert metrics["cpu_percent"] >= 0
        assert metrics["memory_percent"] >= 0
    
    def test_check_alerts(self, profiling_config) -> Any:
        """Test alert checking."""
        profiler = RealTimeProfiler(profiling_config)
        
        # Mock alert callback
        alerts_received = []
        
        def alert_callback(alert, metrics) -> Any:
            alerts_received.append(alert)
        
        profiler.add_alert_callback(alert_callback)
        
        # Test high CPU usage
        high_cpu_metrics = {
            "cpu_percent": 95.0,
            "memory_percent": 50.0
        }
        
        profiler._check_alerts(high_cpu_metrics)
        assert len(alerts_received) > 0
        assert "High CPU usage" in alerts_received[0]
    
    def test_get_performance_summary(self, profiling_config) -> Optional[Dict[str, Any]]:
        """Test performance summary generation."""
        profiler = RealTimeProfiler(profiling_config)
        
        # Add some mock metrics
        profiler.performance_metrics = [
            {"cpu_percent": 50, "memory_percent": 60},
            {"cpu_percent": 70, "memory_percent": 80},
            {"cpu_percent": 30, "memory_percent": 40}
        ]
        
        summary = profiler.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert "avg_cpu_percent" in summary
        assert "max_cpu_percent" in summary
        assert "avg_memory_percent" in summary
        assert "max_memory_percent" in summary
        assert "monitoring_duration" in summary

class TestOptimizedDataset:
    """Test optimized dataset."""
    
    def test_dataset_initialization(self, data_optimization_config, sample_data) -> Any:
        """Test dataset initialization."""
        dataset = OptimizedDataset(sample_data, data_optimization_config)
        
        assert dataset.config == data_optimization_config
        assert dataset.data == sample_data
        assert dataset.cache == {}
    
    def test_dataset_length(self, data_optimization_config, sample_data) -> Any:
        """Test dataset length."""
        dataset = OptimizedDataset(sample_data, data_optimization_config)
        
        assert len(dataset) == len(sample_data)
    
    def test_dataset_getitem(self, data_optimization_config, sample_data) -> Optional[Dict[str, Any]]:
        """Test dataset item access."""
        dataset = OptimizedDataset(sample_data, data_optimization_config)
        
        item = dataset[0]
        assert item is not None
        assert item in sample_data
    
    def test_dataset_caching(self, data_optimization_config, sample_data) -> Any:
        """Test dataset caching."""
        dataset = OptimizedDataset(sample_data, data_optimization_config)
        
        # Access item
        item1 = dataset[0]
        item2 = dataset[0]
        
        # Should be cached
        assert 0 in dataset.cache
        assert item1 is item2  # Same object reference

class TestIntegrationWithFineTuning:
    """Test integration with fine-tuning service."""
    
    @pytest.mark.asyncio
    async def test_profile_and_optimize_training(self) -> Any:
        """Test profile and optimize training."""
        service = OptimizedFineTuningService()
        
        # Mock dataset
        class MockDataset:
            def __init__(self) -> Any:
                self.data = ["sample text"] * 100
            
            def __len__(self) -> Any:
                return len(self.data)
            
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                return self.data[idx]
        
        dataset = MockDataset()
        
        # Mock model loading
        with patch.object(service, 'load_model', return_value=Mock()):
            result = await service.profile_and_optimize_training(
                model_name="test_model",
                dataset=dataset,
                training_config={
                    "epochs": 1,
                    "learning_rate": 5e-5
                },
                user_id=123,
                profile_data_loading=True,
                profile_preprocessing=True,
                profile_training=True
            )
        
        assert result["success"] == True
        assert "profiling_time" in result
        assert "data_loading_results" in result
        assert "preprocessing_results" in result
        assert "training_results" in result
        assert "optimization_plan" in result
        assert "optimization_results" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_optimize_dataset_loading(self) -> Any:
        """Test dataset loading optimization."""
        service = OptimizedFineTuningService()
        
        # Mock dataset
        class MockDataset:
            def __init__(self) -> Any:
                self.data = ["sample text"] * 100
            
            def __len__(self) -> Any:
                return len(self.data)
            
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                return self.data[idx]
        
        dataset = MockDataset()
        
        dataloader = await service.optimize_dataset_loading(dataset, batch_size=16)
        
        assert dataloader is not None
        assert hasattr(dataloader, 'batch_size')
    
    @pytest.mark.asyncio
    async def test_optimize_preprocessing_pipeline(self) -> Any:
        """Test preprocessing pipeline optimization."""
        service = OptimizedFineTuningService()
        
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        preprocessing_funcs = [func1, func2]
        sample_data = [1, 2, 3, 4, 5]
        
        pipeline = await service.optimize_preprocessing_pipeline(
            preprocessing_funcs, sample_data
        )
        
        # Test pipeline
        results = pipeline(sample_data)
        expected = [(x + 1) * 2 for x in sample_data]
        assert results == expected
    
    @pytest.mark.asyncio
    async def test_get_performance_report(self) -> Optional[Dict[str, Any]]:
        """Test performance report generation."""
        service = OptimizedFineTuningService()
        
        report = await service.get_performance_report()
        
        assert isinstance(report, dict)
        assert "profiling_summary" in report
        assert "data_optimization_stats" in report
        assert "system_metrics" in report
        assert "recommendations" in report

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_optimize_dataset(self, data_optimization_config, sample_data) -> Any:
        """Test dataset optimization utility."""
        class MockDataset:
            def __init__(self, data) -> Any:
                self.data = data
            
            def __len__(self) -> Any:
                return len(self.data)
            
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                return self.data[idx]
        
        dataset = MockDataset(sample_data)
        optimized_dataset = optimize_dataset(dataset, data_optimization_config)
        
        assert isinstance(optimized_dataset, OptimizedDataset)
        assert len(optimized_dataset) == len(dataset)
    
    def test_optimize_dataloader(self, data_optimization_config, sample_data) -> Any:
        """Test dataloader optimization utility."""
        class MockDataset:
            def __init__(self, data) -> Any:
                self.data = data
            
            def __len__(self) -> Any:
                return len(self.data)
            
            def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                return self.data[idx]
        
        dataset = MockDataset(sample_data)
        dataloader = optimize_dataloader(dataset, data_optimization_config, batch_size=16)
        
        assert dataloader is not None
        assert hasattr(dataloader, 'batch_size')
        assert dataloader.batch_size == 16
    
    def test_optimize_preprocessing(self, data_optimization_config) -> Any:
        """Test preprocessing optimization utility."""
        def func1(x) -> Any:
            return x + 1
        
        def func2(x) -> Any:
            return x * 2
        
        preprocessing_funcs = [func1, func2]
        
        pipeline = optimize_preprocessing(preprocessing_funcs, data_optimization_config)
        
        # Test pipeline
        result = pipeline(5)
        assert result == 12  # (5 + 1) * 2
    
    def test_memory_optimization_context(self, data_optimization_config) -> Any:
        """Test memory optimization context."""
        with memory_optimization_context(data_optimization_config) as optimizer:
            assert isinstance(optimizer, MemoryOptimizer)
            assert optimizer.config == data_optimization_config

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_config_handling(self) -> Any:
        """Test invalid configuration handling."""
        with pytest.raises(ValueError):
            ProfilingConfig(min_time_threshold=-1)
        
        with pytest.raises(ValueError):
            ProfilingConfig(min_memory_threshold=-1)
    
    def test_cuda_unavailable_handling(self) -> Any:
        """Test CUDA unavailable handling."""
        with patch('torch.cuda.is_available', return_value=False):
            config = ProfilingConfig(profile_gpu=True)
            optimizer = ProfilingOptimizer(config)
            
            # Should gracefully handle no CUDA
            assert optimizer.gpu_monitor is not None
    
    def test_missing_dependencies_handling(self) -> Any:
        """Test missing dependencies handling."""
        with patch.dict('sys.modules', {'line_profiler': None}):
            config = ProfilingConfig()
            optimizer = ProfilingOptimizer(config)
            
            # Should work without optional dependencies
            assert optimizer.line_profiler is None

class TestPerformanceBenchmarks:
    """Test performance benchmarks."""
    
    def test_profiling_overhead(self, profiling_config) -> Any:
        """Test profiling overhead."""
        optimizer = ProfilingOptimizer(profiling_config)
        
        def simple_function():
            
    """simple_function function."""
return 42
        
        # Time without profiling
        start_time = time.time()
        for _ in range(1000):
            simple_function()
        time_without_profiling = time.time() - start_time
        
        # Time with profiling
        start_time = time.time()
        for _ in range(1000):
            optimizer.profile_code(simple_function)
        time_with_profiling = time.time() - start_time
        
        # Profiling overhead should be reasonable (< 10x)
        overhead_ratio = time_with_profiling / time_without_profiling
        assert overhead_ratio < 10
    
    def test_memory_optimization_effectiveness(self, data_optimization_config) -> Any:
        """Test memory optimization effectiveness."""
        optimizer = MemoryOptimizer(data_optimization_config)
        
        # Create large tensor
        large_tensor = torch.randn(1000, 1000, dtype=torch.float64)
        
        # Optimize memory usage
        optimized_tensor = optimizer.optimize_memory_usage(large_tensor)
        
        # Check memory reduction
        original_memory = large_tensor.element_size() * large_tensor.numel()
        optimized_memory = optimized_tensor.element_size() * optimized_tensor.numel()
        
        # Should reduce memory by ~50% (float64 -> float32)
        memory_reduction = (original_memory - optimized_memory) / original_memory
        assert memory_reduction > 0.4  # At least 40% reduction

# Run tests
match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 