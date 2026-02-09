#!/usr/bin/env python3
"""
Test Bulk Optimization - Comprehensive tests for bulk optimization system
Tests all components of the bulk optimization system
"""

import torch
import torch.nn as nn
import unittest
import time
import tempfile
import os
import json
from typing import List, Tuple
import numpy as np

# Import bulk components
from bulk_optimization_core import (
    BulkOptimizationCore, BulkOptimizationConfig, BulkOptimizationResult,
    create_bulk_optimization_core, optimize_models_bulk
)
from bulk_data_processor import (
    BulkDataProcessor, BulkDataConfig, BulkDataset,
    create_bulk_data_processor, process_dataset_bulk
)
from bulk_operation_manager import (
    BulkOperationManager, BulkOperationConfig, BulkOperation,
    OperationType, OperationStatus,
    create_bulk_operation_manager, submit_bulk_operation
)
from bulk_optimizer import (
    BulkOptimizer, BulkOptimizerConfig,
    create_bulk_optimizer, optimize_models_bulk_simple
)

class TestBulkOptimization(unittest.TestCase):
    """Test cases for bulk optimization system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_models = self._create_test_models()
        self.test_datasets = self._create_test_datasets()
        
    def _create_test_models(self) -> List[Tuple[str, nn.Module]]:
        """Create test models for optimization."""
        models = []
        
        # Simple linear model
        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        # Simple MLP model
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Simple CNN model
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 16, 3)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 5)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        models.extend([
            ("simple_linear", SimpleLinear()),
            ("simple_mlp", SimpleMLP()),
            ("simple_cnn", SimpleCNN())
        ])
        
        return models
    
    def _create_test_datasets(self) -> List[BulkDataset]:
        """Create test datasets."""
        datasets = []
        
        # Create temporary data files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"input": [1, 2, 3, 4, 5], "output": [1, 0]},
                {"input": [6, 7, 8, 9, 10], "output": [0, 1]},
                {"input": [11, 12, 13, 14, 15], "output": [1, 1]}
            ]
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            config = BulkDataConfig(batch_size=2, num_workers=0)
            dataset = BulkDataset(temp_file, config)
            datasets.append(dataset)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        return datasets
    
    def test_bulk_optimization_core_creation(self):
        """Test bulk optimization core creation."""
        config = BulkOptimizationConfig(
            max_workers=2,
            batch_size=4,
            enable_parallel_processing=True
        )
        
        core = BulkOptimizationCore(config)
        self.assertIsInstance(core, BulkOptimizationCore)
        self.assertEqual(core.config.max_workers, 2)
        self.assertEqual(core.config.batch_size, 4)
    
    def test_bulk_optimization_core_optimization(self):
        """Test bulk optimization core optimization."""
        config = BulkOptimizationConfig(
            max_workers=1,
            batch_size=2,
            enable_parallel_processing=False
        )
        
        core = BulkOptimizationCore(config)
        results = core.optimize_models_bulk(self.test_models[:2])
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, BulkOptimizationResult)
            self.assertIn(result.model_name, ["simple_linear", "simple_mlp"])
    
    def test_bulk_data_processor_creation(self):
        """Test bulk data processor creation."""
        config = BulkDataConfig(
            batch_size=4,
            num_workers=2,
            enable_parallel_processing=True
        )
        
        processor = BulkDataProcessor(config)
        self.assertIsInstance(processor, BulkDataProcessor)
        self.assertEqual(processor.config.batch_size, 4)
        self.assertEqual(processor.config.num_workers, 2)
    
    def test_bulk_data_processor_processing(self):
        """Test bulk data processor processing."""
        config = BulkDataConfig(
            batch_size=2,
            num_workers=0,
            enable_parallel_processing=False
        )
        
        processor = BulkDataProcessor(config)
        
        if self.test_datasets:
            result = processor.process_dataset(self.test_datasets[0])
            self.assertIsInstance(result, dict)
            self.assertIn('summary', result)
            self.assertIn('total_batches', result['summary'])
    
    def test_bulk_operation_manager_creation(self):
        """Test bulk operation manager creation."""
        config = BulkOperationConfig(
            max_concurrent_operations=2,
            operation_timeout=300.0,
            enable_operation_queue=True
        )
        
        manager = BulkOperationManager(config)
        self.assertIsInstance(manager, BulkOperationManager)
        self.assertEqual(manager.config.max_concurrent_operations, 2)
    
    def test_bulk_operation_submission(self):
        """Test bulk operation submission."""
        config = BulkOperationConfig(
            max_concurrent_operations=1,
            operation_timeout=60.0,
            enable_operation_queue=False
        )
        
        manager = BulkOperationManager(config)
        
        operation = BulkOperation(
            operation_id="test_operation",
            operation_type=OperationType.OPTIMIZATION,
            models=self.test_models[:2]
        )
        
        operation_id = manager.submit_operation(operation)
        self.assertEqual(operation_id, "test_operation")
        
        # Wait for operation to complete
        time.sleep(2)
        
        status = manager.get_operation_status(operation_id)
        self.assertIn(status, [OperationStatus.COMPLETED, OperationStatus.FAILED])
    
    def test_bulk_optimizer_creation(self):
        """Test bulk optimizer creation."""
        config = BulkOptimizerConfig(
            max_models_per_batch=3,
            enable_parallel_optimization=True,
            optimization_strategies=['memory', 'computational']
        )
        
        optimizer = BulkOptimizer(config)
        self.assertIsInstance(optimizer, BulkOptimizer)
        self.assertEqual(optimizer.config.max_models_per_batch, 3)
    
    def test_bulk_optimizer_optimization(self):
        """Test bulk optimizer optimization."""
        config = BulkOptimizerConfig(
            max_models_per_batch=2,
            enable_parallel_optimization=False,
            optimization_strategies=['memory']
        )
        
        optimizer = BulkOptimizer(config)
        results = optimizer.optimize_models_bulk(self.test_models[:2])
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, BulkOptimizationResult)
    
    def test_bulk_optimizer_dataset_processing(self):
        """Test bulk optimizer dataset processing."""
        config = BulkOptimizerConfig(
            batch_size=2,
            enable_data_processor=True
        )
        
        optimizer = BulkOptimizer(config)
        
        if self.test_datasets:
            results = optimizer.process_datasets_bulk(self.test_datasets)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)
    
    def test_bulk_optimizer_operation_submission(self):
        """Test bulk optimizer operation submission."""
        config = BulkOptimizerConfig(
            enable_operation_manager=True,
            max_concurrent_operations=1
        )
        
        optimizer = BulkOptimizer(config)
        
        operation_id = optimizer.submit_bulk_operation(
            OperationType.OPTIMIZATION,
            self.test_models[:2]
        )
        
        self.assertIsInstance(operation_id, str)
        self.assertTrue(len(operation_id) > 0)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test create_bulk_optimization_core
        core = create_bulk_optimization_core({'max_workers': 1})
        self.assertIsInstance(core, BulkOptimizationCore)
        
        # Test create_bulk_data_processor
        processor = create_bulk_data_processor({'batch_size': 2})
        self.assertIsInstance(processor, BulkDataProcessor)
        
        # Test create_bulk_operation_manager
        manager = create_bulk_operation_manager({'max_concurrent_operations': 1})
        self.assertIsInstance(manager, BulkOperationManager)
        
        # Test create_bulk_optimizer
        optimizer = create_bulk_optimizer({'max_models_per_batch': 2})
        self.assertIsInstance(optimizer, BulkOptimizer)
    
    def test_optimize_models_bulk_simple(self):
        """Test simple bulk optimization function."""
        results = optimize_models_bulk_simple(
            self.test_models[:2],
            {'max_workers': 1, 'enable_parallel_processing': False}
        )
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, BulkOptimizationResult)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        config = BulkOptimizationConfig(
            enable_performance_monitoring=True,
            max_workers=1
        )
        
        core = BulkOptimizationCore(config)
        results = core.optimize_models_bulk(self.test_models[:2])
        
        # Check that performance metrics are collected
        self.assertIsInstance(core.performance_metrics, dict)
    
    def test_memory_management(self):
        """Test memory management during optimization."""
        config = BulkOptimizationConfig(
            memory_limit_gb=1.0,
            enable_memory_pooling=True,
            max_workers=1
        )
        
        core = BulkOptimizationCore(config)
        results = core.optimize_models_bulk(self.test_models[:2])
        
        # Check that memory management is working
        self.assertEqual(len(results), 2)
    
    def test_error_handling(self):
        """Test error handling in bulk operations."""
        config = BulkOptimizationConfig(
            max_workers=1,
            enable_parallel_processing=False
        )
        
        core = BulkOptimizationCore(config)
        
        # Test with invalid model (should handle gracefully)
        invalid_models = [("invalid", None)]
        results = core.optimize_models_bulk(invalid_models)
        
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIsNotNone(results[0].error_message)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = BulkOptimizationConfig(
            max_workers=2,
            batch_size=4,
            memory_limit_gb=8.0
        )
        self.assertIsInstance(valid_config, BulkOptimizationConfig)
        
        # Test configuration with invalid values (should use defaults)
        invalid_config = BulkOptimizationConfig(
            max_workers=-1,  # Invalid, should use default
            batch_size=0     # Invalid, should use default
        )
        self.assertGreater(invalid_config.max_workers, 0)
        self.assertGreater(invalid_config.batch_size, 0)

def run_bulk_optimization_tests():
    """Run all bulk optimization tests."""
    print("🧪 Running Bulk Optimization Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBulkOptimization)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  - Tests run: {result.testsRun}")
    print(f"  - Failures: {len(result.failures)}")
    print(f"  - Errors: {len(result.errors)}")
    print(f"  - Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_bulk_optimization_tests()
    exit(0 if success else 1)

