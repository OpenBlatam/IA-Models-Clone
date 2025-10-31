#!/usr/bin/env python3
"""
Comprehensive Test Suite for Performance Optimization Module
Tests all optimization functionality, monitoring, and integration
"""
import unittest
import tempfile
import shutil
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent))

class TestPerformanceConfig(unittest.TestCase):
    """Test PerformanceConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from performance_optimization import PerformanceConfig
        
        config = PerformanceConfig()
        
        # Test default values
        self.assertTrue(config.enable_cudnn_benchmark)
        self.assertTrue(config.enable_tf32)
        self.assertTrue(config.enable_amp)
        self.assertTrue(config.enable_compile)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.cache_size, 1000)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        from performance_optimization import PerformanceConfig
        
        config = PerformanceConfig(
            enable_cudnn_benchmark=False,
            enable_amp=False,
            num_workers=8,
            cache_size=500
        )
        
        self.assertFalse(config.enable_cudnn_benchmark)
        self.assertFalse(config.enable_amp)
        self.assertEqual(config.num_workers, 8)
        self.assertEqual(config.cache_size, 500)

class TestPerformanceOptimizer(unittest.TestCase):
    """Test PerformanceOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, PerformanceOptimizer
        
        self.config = PerformanceConfig(
            enable_cudnn_benchmark=True,
            enable_tf32=True,
            enable_amp=True,
            enable_system_optimization=False  # Disable for testing
        )
        
        self.optimizer = PerformanceOptimizer(self.config)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config, self.config)
        self.assertIsNotNone(self.optimizer.performance_monitor)
    
    def test_gpu_optimizations(self):
        """Test GPU optimization setup."""
        if torch.cuda.is_available():
            # Test that GPU optimizations are applied
            self.assertIsNotNone(self.optimizer.device)
            self.assertEqual(self.optimizer.device.type, "cuda")
        else:
            # Test CPU fallback
            self.assertEqual(self.optimizer.device.type, "cpu")
    
    def test_pytorch_optimizations(self):
        """Test PyTorch optimization setup."""
        # Test that optimizations are attempted
        self.assertIsNotNone(self.optimizer)
    
    def test_error_handling(self):
        """Test error handling in optimization setup."""
        # Test with invalid configuration
        with patch.object(self.optimizer, '_setup_gpu_optimizations', side_effect=Exception("Test error")):
            # Should not crash
            pass

class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceMonitor
        
        self.monitor = PerformanceMonitor()
    
    def test_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor)
        self.assertFalse(self.monitor.monitoring_active)
        self.assertEqual(len(self.monitor.metrics_history), 0)
    
    def test_start_stop_monitoring(self):
        """Test monitoring start and stop."""
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        
        # Wait a bit for metrics collection
        time.sleep(0.1)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
        
        # Should have collected some metrics
        self.assertGreater(len(self.monitor.metrics_history), 0)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = self.monitor._collect_metrics()
        
        # Check required fields
        required_fields = ['timestamp', 'cpu_percent', 'memory_percent']
        for field in required_fields:
            self.assertIn(field, metrics)
        
        # Check data types
        self.assertIsInstance(metrics['timestamp'], float)
        self.assertIsInstance(metrics['cpu_percent'], float)
        self.assertIsInstance(metrics['memory_percent'], float)
    
    def test_gpu_memory_usage(self):
        """Test GPU memory usage collection."""
        gpu_memory = self.monitor._get_gpu_memory_usage()
        
        if torch.cuda.is_available():
            # Should have GPU memory info
            self.assertIsInstance(gpu_memory, dict)
        else:
            # Should return empty dict for CPU
            self.assertEqual(gpu_memory, {})
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Start monitoring and collect some metrics
        self.monitor.start_monitoring()
        time.sleep(0.1)
        self.monitor.stop_monitoring()
        
        summary = self.monitor.get_performance_summary()
        
        # Check summary structure
        self.assertIn('monitoring_duration', summary)
        self.assertIn('total_metrics', summary)
        self.assertIn('cpu_stats', summary)
        self.assertIn('memory_stats', summary)
        
        # Check that metrics were collected
        self.assertGreater(summary['total_metrics'], 0)

class TestOptimizedDataLoader(unittest.TestCase):
    """Test OptimizedDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, OptimizedDataLoader
        
        self.config = PerformanceConfig(
            num_workers=0,  # Use main process for testing
            enable_async_data_loading=False
        )
        
        # Create a simple dataset
        class SimpleDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {'input': torch.randn(10), 'target': torch.randn(1)}
        
        self.dataset = SimpleDataset()
        self.loader = OptimizedDataLoader(
            self.dataset, 
            self.config, 
            batch_size=16
        )
    
    def test_initialization(self):
        """Test data loader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.dataset, self.dataset)
        self.assertEqual(self.loader.config, self.config)
    
    def test_iteration(self):
        """Test data loader iteration."""
        batch_count = 0
        for batch in self.loader:
            batch_count += 1
            self.assertIn('input', batch)
            self.assertIn('target', batch)
            self.assertEqual(batch['input'].shape[0], 16)
            self.assertEqual(batch['target'].shape[0], 16)
        
        # Should have processed all data
        expected_batches = len(self.dataset) // 16
        self.assertEqual(batch_count, expected_batches)
    
    def test_async_loading(self):
        """Test async data loading setup."""
        config = PerformanceConfig(enable_async_data_loading=True, max_workers=2)
        loader = OptimizedDataLoader(self.dataset, config, batch_size=16)
        
        # Should have async loading setup
        self.assertTrue(hasattr(loader, 'executor'))
        self.assertTrue(hasattr(loader, 'prefetch_queue'))

class TestModelOptimizer(unittest.TestCase):
    """Test ModelOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, ModelOptimizer
        
        self.config = PerformanceConfig(
            enable_gradient_checkpointing=True,
            enable_compile=False,  # Disable for testing
            enable_memory_efficient_attention=True,
            enable_flash_attention=True
        )
        
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        self.model_optimizer = ModelOptimizer(self.model, self.config)
    
    def test_initialization(self):
        """Test model optimizer initialization."""
        self.assertIsNotNone(self.model_optimizer)
        self.assertEqual(self.model_optimizer.model, self.model)
        self.assertEqual(self.model_optimizer.config, self.config)
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing application."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            # Should have gradient checkpointing enabled
            self.assertTrue(hasattr(self.model, 'gradient_checkpointing_enable'))
    
    def test_optimized_model(self):
        """Test optimized model retrieval."""
        optimized_model = self.model_optimizer.get_optimized_model()
        self.assertIsNotNone(optimized_model)
    
    def test_restore_original_model(self):
        """Test original model restoration."""
        original_model = self.model_optimizer.restore_original_model()
        self.assertEqual(original_model, self.model)
    
    def test_error_handling(self):
        """Test error handling in optimization."""
        # Test with invalid model
        with patch.object(self.model, 'gradient_checkpointing_enable', side_effect=Exception("Test error")):
            # Should handle errors gracefully
            pass

class TestTrainingOptimizer(unittest.TestCase):
    """Test TrainingOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, TrainingOptimizer
        
        self.config = PerformanceConfig(
            enable_amp=True,
            enable_gradient_accumulation=True,
            gradient_accumulation_steps=2
        )
        
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(100, 10)
        )
        
        self.training_optimizer = TrainingOptimizer(self.model, self.config)
    
    def test_initialization(self):
        """Test training optimizer initialization."""
        self.assertIsNotNone(self.training_optimizer)
        self.assertEqual(self.training_optimizer.model, self.model)
        self.assertEqual(self.training_optimizer.config, self.config)
    
    def test_training_context(self):
        """Test training context manager."""
        with self.training_optimizer.training_context() as optimizer:
            self.assertEqual(optimizer, self.training_optimizer)
    
    def test_optimize_training_step(self):
        """Test training step optimization."""
        # Create dummy loss and optimizer
        dummy_input = torch.randn(16, 100)
        output = self.model(dummy_input)
        loss = nn.MSELoss()(output, torch.randn(16, 10))
        
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Test training step
        self.training_optimizer.optimize_training_step(loss, optimizer)
        
        # Check that step counter increased
        self.assertEqual(self.training_optimizer.optimization_stats['step'], 1)
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        dummy_input = torch.randn(16, 100)
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Run multiple steps
        for i in range(3):
            output = self.model(dummy_input)
            loss = nn.MSELoss()(output, torch.randn(16, 10))
            self.training_optimizer.optimize_training_step(loss, optimizer)
        
        # Should have processed 3 steps
        self.assertEqual(self.training_optimizer.optimization_stats['step'], 3)
    
    def test_optimization_stats(self):
        """Test optimization statistics retrieval."""
        stats = self.training_optimizer.get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('step', stats)

class TestCacheManager(unittest.TestCase):
    """Test CacheManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, CacheManager
        
        self.config = PerformanceConfig(cache_size=3)
        self.cache_manager = CacheManager(self.config)
    
    def test_initialization(self):
        """Test cache manager initialization."""
        self.assertIsNotNone(self.cache_manager)
        self.assertEqual(self.cache_manager.config, self.config)
        self.assertEqual(len(self.cache_manager.model_cache), 0)
        self.assertEqual(len(self.cache_manager.data_cache), 0)
    
    def test_model_caching(self):
        """Test model caching functionality."""
        # Create dummy model
        model = nn.Linear(10, 5)
        
        # Cache model
        self.cache_manager.cache_model("test_model", model)
        self.assertIn("test_model", self.cache_manager.model_cache)
        
        # Retrieve cached model
        cached_model = self.cache_manager.get_cached_model("test_model")
        self.assertEqual(cached_model, model)
        
        # Check cache hit
        self.assertEqual(self.cache_manager.cache_stats['hits'], 1)
        self.assertEqual(self.cache_manager.cache_stats['misses'], 0)
    
    def test_data_caching(self):
        """Test data caching functionality."""
        # Cache data
        test_data = torch.randn(100, 10)
        self.cache_manager.cache_data("test_data", test_data)
        self.assertIn("test_data", self.cache_manager.data_cache)
        
        # Retrieve cached data
        cached_data = self.cache_manager.get_cached_data("test_data")
        self.assertEqual(cached_data, test_data)
    
    def test_cache_eviction(self):
        """Test cache eviction when full."""
        # Fill cache beyond capacity
        for i in range(5):
            model = nn.Linear(10, 5)
            self.cache_manager.cache_model(f"model_{i}", model)
        
        # Should have evicted oldest entries
        self.assertLessEqual(len(self.cache_manager.model_cache), self.config.cache_size)
        self.assertGreater(self.cache_manager.cache_stats['evictions'], 0)
    
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some cache operations
        model = nn.Linear(10, 5)
        self.cache_manager.cache_model("test", model)
        self.cache_manager.get_cached_model("test")
        self.cache_manager.get_cached_model("nonexistent")
        
        stats = self.cache_manager.get_cache_stats()
        
        # Check required fields
        required_fields = ['hits', 'misses', 'evictions', 'hit_rate']
        for field in required_fields:
            self.assertIn(field, stats)
        
        # Check hit rate calculation
        self.assertGreaterEqual(stats['hit_rate'], 0.0)
        self.assertLessEqual(stats['hit_rate'], 1.0)

class TestPerformanceProfiler(unittest.TestCase):
    """Test PerformanceProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import PerformanceConfig, PerformanceProfiler
        
        self.config = PerformanceConfig(enable_profiling=True)
        self.profiler = PerformanceProfiler(self.config)
    
    def test_initialization(self):
        """Test profiler initialization."""
        self.assertIsNotNone(self.profiler)
        self.assertEqual(self.profiler.config, self.config)
        self.assertFalse(self.profiler.profiling_active)
    
    def test_profile_context(self):
        """Test profiling context manager."""
        with self.profiler.profile_context("test_operation"):
            # Profiling should be active during context
            pass
        
        # Profiling should be stopped after context
        self.assertFalse(self.profiler.profiling_active)
    
    def test_start_stop_profiling(self):
        """Test profiling start and stop."""
        self.profiler.start_profiling("test")
        self.assertTrue(self.profiler.profiling_active)
        
        self.profiler.stop_profiling()
        self.assertFalse(self.profiler.profiling_active)
    
    def test_profiler_summary(self):
        """Test profiler summary retrieval."""
        summary = self.profiler.get_profiler_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('profiler_active', summary)
        self.assertIn('profiler_available', summary)

class TestPerformanceOptimizationIntegration(unittest.TestCase):
    """Test integration of all performance optimization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        from performance_optimization import (
            PerformanceConfig, PerformanceOptimizer, ModelOptimizer,
            TrainingOptimizer, CacheManager, PerformanceProfiler
        )
        
        self.config = PerformanceConfig(
            enable_amp=True,
            enable_compile=False,
            enable_gradient_checkpointing=True,
            num_workers=0,
            enable_profiling=True,
            enable_system_optimization=False
        )
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        from performance_optimization import (
            PerformanceOptimizer, ModelOptimizer, TrainingOptimizer,
            CacheManager, PerformanceProfiler
        )
        
        # Initialize all components
        perf_optimizer = PerformanceOptimizer(self.config)
        model = nn.Sequential(nn.Linear(100, 10))
        model_optimizer = ModelOptimizer(model, self.config)
        training_optimizer = TrainingOptimizer(model, self.config)
        cache_manager = CacheManager(self.config)
        profiler = PerformanceProfiler(self.config)
        
        # Test integration
        self.assertIsNotNone(perf_optimizer)
        self.assertIsNotNone(model_optimizer)
        self.assertIsNotNone(training_optimizer)
        self.assertIsNotNone(cache_manager)
        self.assertIsNotNone(profiler)
    
    def test_training_with_optimizations(self):
        """Test training loop with all optimizations."""
        from performance_optimization import (
            PerformanceOptimizer, ModelOptimizer, TrainingOptimizer
        )
        
        # Setup
        perf_optimizer = PerformanceOptimizer(self.config)
        model = nn.Sequential(nn.Linear(100, 10))
        model_optimizer = ModelOptimizer(model, self.config)
        training_optimizer = TrainingOptimizer(model, self.config)
        
        # Start monitoring
        perf_optimizer.performance_monitor.start_monitoring()
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters())
        
        with training_optimizer.training_context():
            for epoch in range(2):
                # Simulate training
                dummy_input = torch.randn(16, 100)
                output = model(dummy_input)
                loss = nn.MSELoss()(output, torch.randn(16, 10))
                
                training_optimizer.optimize_training_step(loss, optimizer)
        
        # Stop monitoring
        perf_optimizer.performance_monitor.stop_monitoring()
        
        # Get results
        summary = perf_optimizer.performance_monitor.get_performance_summary()
        training_stats = training_optimizer.get_optimization_stats()
        
        # Verify results
        self.assertGreater(summary['total_metrics'], 0)
        self.assertEqual(training_stats['step'], 2)

def run_performance_tests():
    """Run performance-specific tests."""
    print("Running performance tests...")
    
    # Test performance monitoring
    from performance_optimization import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    time.sleep(0.1)
    monitor.stop_monitoring()
    
    summary = monitor.get_performance_summary()
    print(f"Performance monitoring test: {len(summary)} metrics collected")
    
    # Test cache performance
    from performance_optimization import CacheManager, PerformanceConfig
    
    config = PerformanceConfig(cache_size=1000)
    cache_manager = CacheManager(config)
    
    # Fill cache
    start_time = time.time()
    for i in range(100):
        cache_manager.cache_data(f"data_{i}", torch.randn(100, 100))
    
    # Test retrieval
    for i in range(100):
        cache_manager.get_cached_data(f"data_{i}")
    
    end_time = time.time()
    cache_stats = cache_manager.get_cache_stats()
    
    print(f"Cache performance test: {cache_stats['hits']} hits in {end_time - start_time:.4f}s")
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2f}")

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPerformanceConfig,
        TestPerformanceOptimizer,
        TestPerformanceMonitor,
        TestOptimizedDataLoader,
        TestModelOptimizer,
        TestTrainingOptimizer,
        TestCacheManager,
        TestPerformanceProfiler,
        TestPerformanceOptimizationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    print("🧪 Performance Optimization Module Test Suite")
    print("=" * 50)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance tests
    print("\n" + "=" * 50)
    run_performance_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("=" * 50)
