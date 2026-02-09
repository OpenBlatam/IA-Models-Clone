from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from advanced_code_profiling_optimization import (
        import concurrent.futures
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Advanced Code Profiling and Optimization System

This test suite covers:
- All profiling components (CPU, GPU, Memory, I/O)
- Bottleneck identification and analysis
- Automatic optimization suggestions
- Data loading and preprocessing optimization
- Performance monitoring and alerting
- Edge cases and error handling
- Integration testing
"""



    ProfilingConfig, ProfilingLevel, OptimizationTarget, BottleneckType,
    PerformanceMetrics, CPUMemoryProfiler, GPUProfiler, DataLoadingProfiler,
    AdvancedProfiler, CodeOptimizer, PerformanceMonitor,
    profile_function, profile_context
)


class TestDataset(Dataset):
    """Test dataset for unit testing."""
    
    def __init__(self, num_samples: int = 100, input_dim: int = 64):
        
    """__init__ function."""
self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 5, (num_samples,))
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.labels[idx]


class TestModel(nn.Module):
    """Test model for unit testing."""
    
    def __init__(self, input_dim: int = 64, num_classes: int = 5):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x) -> Any:
        return self.linear(x)


class TestProfilingConfig:
    """Test ProfilingConfig class."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = ProfilingConfig()
        
        assert config.enabled is True
        assert config.level == ProfilingLevel.DETAILED
        assert config.sampling_rate == 0.1
        assert config.max_samples == 10000
        assert config.cpu_threshold == 80.0
        assert config.memory_threshold == 85.0
        assert config.gpu_threshold == 90.0
        assert config.auto_optimize is True
        assert config.enable_monitoring is True
        assert config.save_profiles is True
        assert config.generate_reports is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = ProfilingConfig(
            enabled=False,
            level=ProfilingLevel.BASIC,
            sampling_rate=0.5,
            max_samples=5000,
            cpu_threshold=70.0,
            memory_threshold=80.0,
            gpu_threshold=85.0,
            auto_optimize=False,
            enable_monitoring=False
        )
        
        assert config.enabled is False
        assert config.level == ProfilingLevel.BASIC
        assert config.sampling_rate == 0.5
        assert config.max_samples == 5000
        assert config.cpu_threshold == 70.0
        assert config.memory_threshold == 80.0
        assert config.gpu_threshold == 85.0
        assert config.auto_optimize is False
        assert config.enable_monitoring is False
    
    def test_post_init(self) -> Any:
        """Test post-initialization setup."""
        config = ProfilingConfig()
        
        # Should create profile directory
        assert os.path.exists(config.profile_dir)


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_initialization(self) -> Any:
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        
        assert metrics.execution_time == 0.0
        assert metrics.cpu_time == 0.0
        assert metrics.gpu_time == 0.0
        assert metrics.i_o_time == 0.0
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.gpu_usage == 0.0
        assert metrics.bottleneck_type is None
        assert metrics.bottleneck_severity == 0.0
    
    def test_custom_metrics(self) -> Any:
        """Test custom metrics."""
        metrics = PerformanceMetrics(
            execution_time=1.5,
            cpu_time=0.8,
            gpu_time=0.7,
            cpu_usage=75.0,
            memory_usage=4.2,
            gpu_usage=85.0,
            bottleneck_type=BottleneckType.CPU_BOUND,
            bottleneck_severity=0.8
        )
        
        assert metrics.execution_time == 1.5
        assert metrics.cpu_time == 0.8
        assert metrics.gpu_time == 0.7
        assert metrics.cpu_usage == 75.0
        assert metrics.memory_usage == 4.2
        assert metrics.gpu_usage == 85.0
        assert metrics.bottleneck_type == BottleneckType.CPU_BOUND
        assert metrics.bottleneck_severity == 0.8
    
    def test_to_dict(self) -> Any:
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            execution_time=1.0,
            cpu_usage=50.0,
            memory_usage=2.0,
            bottleneck_type=BottleneckType.MEMORY_BOUND,
            bottleneck_severity=0.6
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['execution_time'] == 1.0
        assert metrics_dict['cpu_usage'] == 50.0
        assert metrics_dict['memory_usage'] == 2.0
        assert metrics_dict['bottleneck_type'] == 'memory_bound'
        assert metrics_dict['bottleneck_severity'] == 0.6


class TestCPUMemoryProfiler:
    """Test CPUMemoryProfiler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(
            enabled=True,
            enable_tracemalloc=False  # Disable for testing
        )
    
    @pytest.fixture
    def profiler(self, config) -> Any:
        """Create test profiler."""
        return CPUMemoryProfiler(config)
    
    def test_initialization(self, config) -> Any:
        """Test profiler initialization."""
        profiler = CPUMemoryProfiler(config)
        
        assert profiler.config == config
        assert len(profiler.metrics_history) == 0
        assert profiler.process is not None
        assert profiler.start_cpu_times is None
        assert profiler.start_memory_info is None
    
    def test_start_profiling(self, profiler) -> Any:
        """Test start profiling."""
        profiler.start_profiling()
        
        assert profiler.start_cpu_times is not None
        assert profiler.start_memory_info is not None
        assert profiler.start_io_counters is not None
    
    def test_stop_profiling(self, profiler) -> Any:
        """Test stop profiling."""
        profiler.start_profiling()
        
        # Simulate some work
        time.sleep(0.1)
        
        metrics = profiler.stop_profiling()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cpu_time >= 0.0
        assert metrics.cpu_usage >= 0.0
        assert metrics.memory_usage >= 0.0
        assert metrics.disk_read_bytes >= 0.0
        assert metrics.disk_write_bytes >= 0.0
    
    def test_get_bottlenecks_no_bottlenecks(self, profiler) -> Optional[Dict[str, Any]]:
        """Test bottleneck identification with no bottlenecks."""
        # Set low usage metrics
        profiler.current_metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0
        )
        
        bottlenecks = profiler.get_bottlenecks()
        assert len(bottlenecks) == 0
    
    def test_get_bottlenecks_cpu_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test CPU bottleneck identification."""
        # Set high CPU usage
        profiler.current_metrics = PerformanceMetrics(
            cpu_usage=90.0  # Above threshold
        )
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.CPU_BOUND
        assert bottlenecks[0]['severity'] == 0.9
        assert 'High CPU usage' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0
    
    def test_get_bottlenecks_memory_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test memory bottleneck identification."""
        # Set high memory usage
        profiler.current_metrics = PerformanceMetrics(
            memory_usage=90.0  # Above threshold
        )
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.MEMORY_BOUND
        assert bottlenecks[0]['severity'] == 0.9
        assert 'High memory usage' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0


class TestGPUProfiler:
    """Test GPUProfiler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(
            enabled=True,
            enable_pytorch_profiler=False  # Disable for testing
        )
    
    @pytest.fixture
    def profiler(self, config) -> Any:
        """Create test profiler."""
        return GPUProfiler(config)
    
    def test_initialization(self, config) -> Any:
        """Test profiler initialization."""
        profiler = GPUProfiler(config)
        
        assert profiler.config == config
        assert profiler.profiler is None
        assert profiler.start_time is None
        assert profiler.gpu_available == torch.cuda.is_available()
    
    def test_start_profiling(self, profiler) -> Any:
        """Test start profiling."""
        profiler.start_profiling()
        
        assert profiler.start_time is not None
    
    def test_stop_profiling(self, profiler) -> Any:
        """Test stop profiling."""
        profiler.start_profiling()
        
        # Simulate some work
        time.sleep(0.1)
        
        metrics = profiler.stop_profiling()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time >= 0.0
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_stop_profiling_no_gpu(self, mock_cuda, profiler) -> Any:
        """Test stop profiling without GPU."""
        profiler.start_profiling()
        metrics = profiler.stop_profiling()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time == 0.0
    
    def test_get_bottlenecks_no_bottlenecks(self, profiler) -> Optional[Dict[str, Any]]:
        """Test bottleneck identification with no bottlenecks."""
        # Set low usage metrics
        profiler.current_metrics = PerformanceMetrics(
            gpu_usage=50.0,
            gpu_memory_usage=60.0
        )
        
        bottlenecks = profiler.get_bottlenecks()
        assert len(bottlenecks) == 0
    
    def test_get_bottlenecks_gpu_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test GPU bottleneck identification."""
        # Set high GPU usage
        profiler.current_metrics = PerformanceMetrics(
            gpu_usage=95.0  # Above threshold
        )
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.GPU_BOUND
        assert bottlenecks[0]['severity'] == 0.95
        assert 'High GPU usage' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0
    
    def test_get_bottlenecks_gpu_memory_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test GPU memory bottleneck identification."""
        # Set high GPU memory usage
        profiler.current_metrics = PerformanceMetrics(
            gpu_memory_usage=95.0  # Above threshold
        )
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.MEMORY_BOUND
        assert bottlenecks[0]['severity'] == 0.95
        assert 'High GPU memory usage' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0


class TestDataLoadingProfiler:
    """Test DataLoadingProfiler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(enabled=True)
    
    @pytest.fixture
    def profiler(self, config) -> Any:
        """Create test profiler."""
        return DataLoadingProfiler(config)
    
    def test_initialization(self, config) -> Any:
        """Test profiler initialization."""
        profiler = DataLoadingProfiler(config)
        
        assert profiler.config == config
        assert len(profiler.dataloader_metrics) == 0
        assert len(profiler.preprocessing_metrics) == 0
        assert profiler.start_time is None
    
    def test_start_profiling(self, profiler) -> Any:
        """Test start profiling."""
        profiler.start_profiling()
        assert profiler.start_time is not None
    
    def test_stop_profiling(self, profiler) -> Any:
        """Test stop profiling."""
        profiler.start_profiling()
        
        # Add some metrics
        profiler.dataloader_metrics['time'] = [0.1, 0.2, 0.3]
        profiler.dataloader_metrics['samples'] = [32, 32, 32]
        profiler.preprocessing_metrics['time'] = [0.05, 0.06, 0.07]
        
        metrics = profiler.stop_profiling()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time >= 0.0
        assert metrics.i_o_time > 0.0
        assert metrics.samples_per_second > 0.0
        assert metrics.data_processed > 0.0
    
    def test_profile_dataloader(self, profiler) -> Any:
        """Test data loader profiling."""
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        profiler.profile_dataloader(dataloader, num_batches=3)
        
        assert len(profiler.dataloader_metrics['time']) == 3
        assert len(profiler.dataloader_metrics['samples']) == 3
    
    def test_profile_preprocessing(self, profiler) -> Any:
        """Test preprocessing profiling."""
        def test_preprocessing(data) -> Any:
            time.sleep(0.01)  # Simulate work
            return data * 2
        
        test_data = torch.randn(10, 5)
        result = profiler.profile_preprocessing(test_preprocessing, test_data)
        
        assert result is not None
        assert len(profiler.preprocessing_metrics['time']) == 1
    
    def test_get_bottlenecks_no_bottlenecks(self, profiler) -> Optional[Dict[str, Any]]:
        """Test bottleneck identification with no bottlenecks."""
        # Set fast metrics
        profiler.dataloader_metrics['time'] = [0.01, 0.02, 0.01]
        profiler.preprocessing_metrics['time'] = [0.01, 0.01, 0.02]
        
        bottlenecks = profiler.get_bottlenecks()
        assert len(bottlenecks) == 0
    
    def test_get_bottlenecks_data_loading_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test data loading bottleneck identification."""
        # Set slow data loading
        profiler.dataloader_metrics['time'] = [0.2, 0.3, 0.25]  # Slow
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.DATA_LOADING_BOUND
        assert 'Slow data loading' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0
    
    def test_get_bottlenecks_preprocessing_bound(self, profiler) -> Optional[Dict[str, Any]]:
        """Test preprocessing bottleneck identification."""
        # Set slow preprocessing
        profiler.preprocessing_metrics['time'] = [0.1, 0.15, 0.12]  # Slow
        
        bottlenecks = profiler.get_bottlenecks()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0]['type'] == BottleneckType.PREPROCESSING_BOUND
        assert 'Slow preprocessing' in bottlenecks[0]['description']
        assert len(bottlenecks[0]['suggestions']) > 0


class TestAdvancedProfiler:
    """Test AdvancedProfiler class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(
            enabled=True,
            enable_tracemalloc=False,
            enable_pytorch_profiler=False
        )
    
    @pytest.fixture
    def profiler(self, config) -> Any:
        """Create test profiler."""
        return AdvancedProfiler(config)
    
    def test_initialization(self, config) -> Any:
        """Test profiler initialization."""
        profiler = AdvancedProfiler(config)
        
        assert profiler.config == config
        assert profiler.cpu_memory_profiler is not None
        assert profiler.gpu_profiler is not None
        assert profiler.data_loading_profiler is not None
        assert len(profiler.profiling_history) == 0
        assert len(profiler.optimization_suggestions) == 0
    
    def test_start_profiling(self, profiler) -> Any:
        """Test start profiling."""
        profiler.start_profiling()
        
        # Check that all sub-profilers started
        assert profiler.cpu_memory_profiler.start_cpu_times is not None
        assert profiler.gpu_profiler.start_time is not None
        assert profiler.data_loading_profiler.start_time is not None
    
    def test_stop_profiling(self, profiler) -> Any:
        """Test stop profiling."""
        profiler.start_profiling()
        
        # Simulate some work
        time.sleep(0.1)
        
        results = profiler.stop_profiling()
        
        assert isinstance(results, dict)
        assert 'combined' in results
        assert 'cpu' in results
        assert 'gpu' in results
        assert 'data' in results
        assert 'bottlenecks' in results
        assert 'suggestions' in results
        
        assert isinstance(results['combined'], PerformanceMetrics)
        assert isinstance(results['cpu'], PerformanceMetrics)
        assert isinstance(results['gpu'], PerformanceMetrics)
        assert isinstance(results['data'], PerformanceMetrics)
        assert isinstance(results['bottlenecks'], list)
        assert isinstance(results['suggestions'], list)
    
    def test_profile_dataloader(self, profiler) -> Any:
        """Test data loader profiling."""
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        profiler.profile_dataloader(dataloader, num_batches=3)
        
        # Check that metrics were collected
        assert len(profiler.data_loading_profiler.dataloader_metrics['time']) == 3
    
    def test_profile_preprocessing(self, profiler) -> Any:
        """Test preprocessing profiling."""
        def test_preprocessing(data) -> Any:
            time.sleep(0.01)
            return data * 2
        
        test_data = torch.randn(10, 5)
        result = profiler.profile_preprocessing(test_preprocessing, test_data)
        
        assert result is not None
        assert len(profiler.data_loading_profiler.preprocessing_metrics['time']) == 1
    
    def test_get_profiling_summary(self, profiler) -> Optional[Dict[str, Any]]:
        """Test profiling summary."""
        # Add some profiling history
        profiler.profiling_history = [
            {
                'metrics': PerformanceMetrics(
                    execution_time=1.0,
                    cpu_usage=50.0,
                    memory_usage=2.0,
                    gpu_usage=60.0
                ),
                'bottlenecks': [{'type': BottleneckType.CPU_BOUND, 'severity': 0.5}]
            },
            {
                'metrics': PerformanceMetrics(
                    execution_time=1.5,
                    cpu_usage=70.0,
                    memory_usage=3.0,
                    gpu_usage=80.0
                ),
                'bottlenecks': [{'type': BottleneckType.GPU_BOUND, 'severity': 0.8}]
            }
        ]
        
        summary = profiler.get_profiling_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_executions'] == 2
        assert summary['avg_execution_time'] == 1.25
        assert summary['avg_cpu_usage'] == 60.0
        assert summary['avg_memory_usage'] == 2.5
        assert summary['avg_gpu_usage'] == 70.0
        assert summary['most_common_bottleneck'] == (BottleneckType.GPU_BOUND, 1)


class TestCodeOptimizer:
    """Test CodeOptimizer class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(enabled=True)
    
    @pytest.fixture
    def profiler(self, config) -> Any:
        """Create test profiler."""
        return AdvancedProfiler(config)
    
    @pytest.fixture
    def optimizer(self, profiler) -> Any:
        """Create test optimizer."""
        return CodeOptimizer(profiler)
    
    def test_initialization(self, profiler) -> Any:
        """Test optimizer initialization."""
        optimizer = CodeOptimizer(profiler)
        
        assert optimizer.profiler == profiler
        assert len(optimizer.optimizations_applied) == 0
    
    def test_optimize_data_loading(self, optimizer) -> Any:
        """Test data loader optimization."""
        dataset = TestDataset(100)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Intentionally slow
            pin_memory=False
        )
        
        # Profile original dataloader
        optimizer.profiler.profile_dataloader(dataloader, num_batches=3)
        
        # Optimize dataloader
        optimized_dataloader = optimizer.optimize_data_loading(dataloader)
        
        assert optimized_dataloader is not None
        assert optimized_dataloader != dataloader
        assert len(optimizer.optimizations_applied) > 0
    
    def test_optimize_preprocessing(self, optimizer) -> Any:
        """Test preprocessing optimization."""
        def test_preprocessing(data) -> Any:
            time.sleep(0.01)
            return data * 2
        
        # Profile original preprocessing
        test_data = torch.randn(10, 5)
        optimizer.profiler.profile_preprocessing(test_preprocessing, test_data)
        
        # Optimize preprocessing
        optimized_preprocessing = optimizer.optimize_preprocessing(test_preprocessing)
        
        assert optimized_preprocessing is not None
        assert optimized_preprocessing != test_preprocessing
        assert len(optimizer.optimizations_applied) > 0
    
    def test_get_optimization_report(self, optimizer) -> Optional[Dict[str, Any]]:
        """Test optimization report."""
        # Add some optimizations
        optimizer.optimizations_applied = [
            {
                'type': 'increase_num_workers',
                'description': 'Increase num_workers from 0 to 4',
                'expected_improvement': '20-40% faster data loading'
            },
            {
                'type': 'enable_pin_memory',
                'description': 'Enable pin_memory for faster GPU transfer',
                'expected_improvement': '10-20% faster GPU transfer'
            }
        ]
        
        report = optimizer.get_optimization_report()
        
        assert isinstance(report, dict)
        assert report['total_optimizations'] == 2
        assert len(report['optimizations']) == 2
        assert len(report['expected_improvements']) == 2


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    @pytest.fixture
    def config(self) -> Any:
        """Create test configuration."""
        return ProfilingConfig(
            enabled=True,
            monitoring_interval=0.1,
            alert_threshold=0.8
        )
    
    @pytest.fixture
    def monitor(self, config) -> Any:
        """Create test monitor."""
        return PerformanceMonitor(config)
    
    def test_initialization(self, config) -> Any:
        """Test monitor initialization."""
        monitor = PerformanceMonitor(config)
        
        assert monitor.config == config
        assert monitor.monitoring_active is False
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alert_callbacks) == 0
    
    def test_start_stop_monitoring(self, monitor) -> Any:
        """Test start and stop monitoring."""
        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
    
    def test_collect_current_metrics(self, monitor) -> Any:
        """Test current metrics collection."""
        metrics = monitor._collect_current_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.cpu_usage >= 0.0
        assert metrics.memory_usage >= 0.0
        assert metrics.gpu_usage >= 0.0
    
    def test_check_alerts_no_alerts(self, monitor) -> Any:
        """Test alert checking with no alerts."""
        metrics = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            gpu_usage=70.0
        )
        
        alerts = monitor._check_alerts(metrics)
        assert len(alerts) == 0
    
    def test_check_alerts_high_cpu(self, monitor) -> Any:
        """Test alert checking with high CPU usage."""
        metrics = PerformanceMetrics(
            cpu_usage=90.0  # Above threshold
        )
        
        alerts = monitor._check_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'high_cpu_usage'
        assert alerts[0]['severity'] == 0.9
        assert 'High CPU usage' in alerts[0]['message']
    
    def test_check_alerts_high_memory(self, monitor) -> Any:
        """Test alert checking with high memory usage."""
        metrics = PerformanceMetrics(
            memory_usage=90.0  # Above threshold
        )
        
        alerts = monitor._check_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'high_memory_usage'
        assert alerts[0]['severity'] == 0.9
        assert 'High memory usage' in alerts[0]['message']
    
    def test_check_alerts_high_gpu(self, monitor) -> Any:
        """Test alert checking with high GPU usage."""
        metrics = PerformanceMetrics(
            gpu_usage=95.0  # Above threshold
        )
        
        alerts = monitor._check_alerts(metrics)
        
        assert len(alerts) == 1
        assert alerts[0]['type'] == 'high_gpu_usage'
        assert alerts[0]['severity'] == 0.95
        assert 'High GPU usage' in alerts[0]['message']
    
    def test_add_alert_callback(self, monitor) -> Any:
        """Test adding alert callback."""
        async def test_callback(alert) -> Any:
            pass
        
        monitor.add_alert_callback(test_callback)
        assert len(monitor.alert_callbacks) == 1
        assert monitor.alert_callbacks[0] == test_callback
    
    def test_get_monitoring_summary(self, monitor) -> Optional[Dict[str, Any]]:
        """Test monitoring summary."""
        # Add some metrics history
        monitor.metrics_history = deque([
            PerformanceMetrics(cpu_usage=50.0, memory_usage=2.0, gpu_usage=60.0),
            PerformanceMetrics(cpu_usage=70.0, memory_usage=3.0, gpu_usage=80.0),
            PerformanceMetrics(cpu_usage=60.0, memory_usage=2.5, gpu_usage=70.0)
        ], maxlen=1000)
        
        summary = monitor.get_monitoring_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_samples'] == 3
        assert summary['avg_cpu_usage'] == 60.0
        assert summary['avg_memory_usage'] == 2.5
        assert summary['avg_gpu_usage'] == 70.0
        assert summary['max_cpu_usage'] == 70.0
        assert summary['max_memory_usage'] == 3.0
        assert summary['max_gpu_usage'] == 80.0


class TestDecorators:
    """Test profiling decorators."""
    
    def test_profile_function_decorator(self) -> Any:
        """Test profile_function decorator."""
        config = ProfilingConfig(enabled=True)
        
        @profile_function(config)
        def test_function():
            
    """test_function function."""
time.sleep(0.01)
            return "test"
        
        result = test_function()
        assert result == "test"
    
    @pytest.mark.asyncio
    async def test_profile_context_manager(self) -> Any:
        """Test profile_context context manager."""
        config = ProfilingConfig(enabled=True)
        
        with profile_context("test_context", config) as profiler:
            assert profiler is not None
            time.sleep(0.01)


class TestIntegration:
    """Integration tests for the advanced code profiling and optimization system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_profiling(self) -> Any:
        """Test end-to-end profiling workflow."""
        config = ProfilingConfig(
            enabled=True,
            enable_tracemalloc=False,
            enable_pytorch_profiler=False
        )
        
        profiler = AdvancedProfiler(config)
        optimizer = CodeOptimizer(profiler)
        
        # Create test data
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Profile data loading
        profiler.profile_dataloader(dataloader, num_batches=3)
        
        # Optimize data loading
        optimized_dataloader = optimizer.optimize_data_loading(dataloader)
        
        # Profile preprocessing
        def test_preprocessing(data) -> Any:
            time.sleep(0.01)
            return data * 2
        
        test_data = torch.randn(10, 5)
        profiler.profile_preprocessing(test_preprocessing, test_data)
        
        # Optimize preprocessing
        optimized_preprocessing = optimizer.optimize_preprocessing(test_preprocessing)
        
        # Get results
        profiling_summary = profiler.get_profiling_summary()
        optimization_report = optimizer.get_optimization_report()
        
        assert isinstance(profiling_summary, dict)
        assert isinstance(optimization_report, dict)
        assert profiling_summary['total_executions'] > 0
        assert optimization_report['total_optimizations'] > 0
    
    def test_profiling_with_real_model(self) -> Any:
        """Test profiling with a real model."""
        config = ProfilingConfig(
            enabled=True,
            enable_tracemalloc=False,
            enable_pytorch_profiler=False
        )
        
        profiler = AdvancedProfiler(config)
        
        # Create model and data
        model = TestModel()
        dataset = TestDataset(100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Profile training
        profiler.start_profiling()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for i, (data, labels) in enumerate(dataloader):
            if i >= 3:  # Limit training steps
                break
            
            # Forward pass
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        results = profiler.stop_profiling()
        
        assert isinstance(results, dict)
        assert 'combined' in results
        assert 'bottlenecks' in results
        assert 'suggestions' in results
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self) -> Any:
        """Test monitoring integration."""
        config = ProfilingConfig(
            enabled=True,
            monitoring_interval=0.1,
            alert_threshold=0.8
        )
        
        monitor = PerformanceMonitor(config)
        
        # Add alert callback
        alerts_received = []
        
        async def alert_callback(alert) -> Any:
            alerts_received.append(alert)
        
        monitor.add_alert_callback(alert_callback)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate high CPU usage
        for _ in range(10):
            # CPU-intensive work
            _ = sum(i * i for i in range(10000))
            await asyncio.sleep(0.05)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Check results
        summary = monitor.get_monitoring_summary()
        assert isinstance(summary, dict)
        assert summary['total_samples'] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_profiling_with_disabled_config(self) -> Any:
        """Test profiling with disabled configuration."""
        config = ProfilingConfig(enabled=False)
        profiler = AdvancedProfiler(config)
        
        # Should not raise errors even when disabled
        profiler.start_profiling()
        results = profiler.stop_profiling()
        
        assert isinstance(results, dict)
    
    def test_profiling_with_empty_dataset(self) -> Any:
        """Test profiling with empty dataset."""
        config = ProfilingConfig(enabled=True)
        profiler = AdvancedProfiler(config)
        
        # Create empty dataset
        dataset = TestDataset(0)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Should handle gracefully
        profiler.profile_dataloader(dataloader, num_batches=0)
        
        assert len(profiler.data_loading_profiler.dataloader_metrics['time']) == 0
    
    def test_profiling_with_exception(self) -> Any:
        """Test profiling with exceptions."""
        config = ProfilingConfig(enabled=True)
        profiler = AdvancedProfiler(config)
        
        def failing_function():
            
    """failing_function function."""
raise ValueError("Test exception")
        
        # Should handle exceptions gracefully
        try:
            profiler.profile_preprocessing(failing_function, None)
        except ValueError:
            pass  # Expected exception
        
        # Profiler should still be functional
        assert profiler is not None
    
    def test_optimization_with_no_bottlenecks(self) -> Any:
        """Test optimization when no bottlenecks are detected."""
        config = ProfilingConfig(enabled=True)
        profiler = AdvancedProfiler(config)
        optimizer = CodeOptimizer(profiler)
        
        # Create fast dataset
        dataset = TestDataset(100)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Profile and optimize
        profiler.profile_dataloader(dataloader, num_batches=3)
        optimized_dataloader = optimizer.optimize_data_loading(dataloader)
        
        # Should still work even with no optimizations needed
        assert optimized_dataloader is not None


class TestPerformance:
    """Performance tests."""
    
    def test_profiling_overhead(self) -> Any:
        """Test profiling overhead."""
        config = ProfilingConfig(
            enabled=True,
            enable_tracemalloc=False,
            enable_pytorch_profiler=False
        )
        
        profiler = AdvancedProfiler(config)
        
        # Measure time without profiling
        start_time = time.time()
        for _ in range(100):
            _ = sum(i * i for i in range(1000))
        time_without_profiling = time.time() - start_time
        
        # Measure time with profiling
        start_time = time.time()
        profiler.start_profiling()
        for _ in range(100):
            _ = sum(i * i for i in range(1000))
        results = profiler.stop_profiling()
        time_with_profiling = time.time() - start_time
        
        # Profiling overhead should be reasonable (< 50%)
        overhead = (time_with_profiling - time_without_profiling) / time_without_profiling
        assert overhead < 0.5
    
    def test_memory_profiling_efficiency(self) -> Any:
        """Test memory profiling efficiency."""
        config = ProfilingConfig(
            enabled=True,
            enable_tracemalloc=True
        )
        
        profiler = AdvancedProfiler(config)
        
        # Profile memory usage
        profiler.start_profiling()
        
        # Allocate some memory
        large_tensor = torch.randn(1000, 1000)
        
        results = profiler.stop_profiling()
        
        # Should detect memory usage
        assert results['combined'].memory_usage > 0.0
    
    def test_concurrent_profiling(self) -> Any:
        """Test concurrent profiling."""
        config = ProfilingConfig(enabled=True)
        
        def profile_workload():
            
    """profile_workload function."""
profiler = AdvancedProfiler(config)
            profiler.start_profiling()
            time.sleep(0.01)
            return profiler.stop_profiling()
        
        # Run multiple profiling sessions concurrently
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(profile_workload) for _ in range(4)]
            results = [future.result() for future in futures]
        
        # All should complete successfully
        assert len(results) == 4
        for result in results:
            assert isinstance(result, dict)
            assert 'combined' in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 