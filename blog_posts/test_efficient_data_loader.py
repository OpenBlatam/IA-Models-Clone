from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import json
import tempfile
import shutil
from typing import Dict, Any, List
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import h5py
import lmdb
import msgpack
from .production_transformers import DeviceManager
from .efficient_data_loader import (
        from transformers import AutoTokenizer
    import asyncio
from typing import Any, List, Dict, Optional
"""
🧪 Efficient Data Loading Test Suite
====================================

Comprehensive test suite for efficient data loading system with
performance benchmarks and integration tests.
"""



# Import our systems
    DataLoaderManager, DataLoaderFactory, DataLoaderConfig,
    DataFormat, CacheStrategy, CachedDataset, OptimizedTextDataset,
    StreamingDataset, DataLoaderProfiler
)

logger = logging.getLogger(__name__)

class TestDataLoaderConfig(unittest.TestCase):
    """Test DataLoader configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = DataLoaderConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_workers, 4)
        self.assertTrue(config.pin_memory)
        self.assertTrue(config.persistent_workers)
        self.assertEqual(config.prefetch_factor, 2)
        self.assertTrue(config.shuffle)
        self.assertEqual(config.cache_strategy, CacheStrategy.MEMORY)
    
    def test_auto_optimization(self) -> Any:
        """Test auto-optimization of configuration."""
        config = DataLoaderConfig(num_workers=-1)
        
        # Should auto-detect optimal workers
        self.assertGreater(config.num_workers, 0)
        self.assertLessEqual(config.num_workers, 8)
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = DataLoaderConfig(
            batch_size=64,
            num_workers=8,
            pin_memory=False,
            cache_strategy=CacheStrategy.DISK,
            cache_size_gb=20.0
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_workers, 8)
        self.assertFalse(config.pin_memory)
        self.assertEqual(config.cache_strategy, CacheStrategy.DISK)
        self.assertEqual(config.cache_size_gb, 20.0)

class TestOptimizedTextDataset(unittest.TestCase):
    """Test optimized text dataset."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = Path(self.temp_dir) / "test_data.csv"
        
        # Create test dataset
        self.create_test_dataset()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self) -> Any:
        """Create synthetic test dataset."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Create text-like features
        texts = [f"Sample text {i} with features {x[:5]}" for i, x in enumerate(X)]
        
        # Save to CSV
        df = pd.DataFrame({
            'text': texts,
            'label': y
        })
        df.to_csv(self.test_data_path, index=False)
    
    def test_dataset_initialization(self) -> Any:
        """Test dataset initialization."""
        # Load test data
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = OptimizedTextDataset(texts, labels)
        
        self.assertEqual(len(dataset), 100)
        self.assertEqual(len(dataset.texts), 100)
        self.assertEqual(len(dataset.labels), 100)
    
    def test_dataset_item_access(self) -> Any:
        """Test dataset item access."""
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Test single item
        item = dataset[0]
        self.assertIn('text', item)
        self.assertIn('labels', item)
        self.assertIsInstance(item['labels'], torch.Tensor)
        self.assertEqual(item['text'], texts[0])
    
    def test_dataset_with_tokenizer(self) -> Any:
        """Test dataset with tokenizer."""
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = OptimizedTextDataset(
            texts, labels, tokenizer=tokenizer,
            max_length=512, cache_encodings=True
        )
        
        # Test item with tokenizer
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
    
    def test_validation(self) -> Any:
        """Test input validation."""
        # Test mismatched lengths
        texts = ["text1", "text2"]
        labels = [0, 1, 2]  # Mismatched length
        
        with self.assertRaises(ValueError):
            OptimizedTextDataset(texts, labels)
        
        # Test empty texts
        with self.assertRaises(ValueError):
            OptimizedTextDataset([], [])

class TestCachedDataset(unittest.TestCase):
    """Test cached dataset."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_memory_caching(self) -> Any:
        """Test memory caching strategy."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 10
        mock_dataset.__getitem__ = lambda idx: {'data': f'item_{idx}', 'idx': idx}
        
        # Create cached dataset
        cached_dataset = CachedDataset(
            mock_dataset,
            cache_strategy=CacheStrategy.MEMORY,
            cache_dir=str(self.cache_dir)
        )
        
        # Test caching
        item1 = cached_dataset[0]
        item2 = cached_dataset[0]  # Should be from cache
        
        self.assertEqual(item1, item2)
        self.assertIn(0, cached_dataset.cache)
    
    def test_disk_caching(self) -> Any:
        """Test disk caching strategy."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 5
        mock_dataset.__getitem__ = lambda idx: {'data': f'item_{idx}', 'idx': idx}
        
        # Create cached dataset
        cached_dataset = CachedDataset(
            mock_dataset,
            cache_strategy=CacheStrategy.DISK,
            cache_dir=str(self.cache_dir),
            cache_size_gb=1.0
        )
        
        # Access items to populate cache
        for i in range(5):
            cached_dataset[i]
        
        # Save cache
        cached_dataset._save_cache()
        
        # Verify cache files exist
        self.assertTrue(cached_dataset.cache_file.exists())
        self.assertTrue(cached_dataset.metadata_file.exists())
    
    def test_cache_size_limit(self) -> Any:
        """Test cache size limiting."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 1000
        mock_dataset.__getitem__ = lambda idx: {'data': f'item_{idx}', 'idx': idx}
        
        # Create cached dataset with small cache
        cached_dataset = CachedDataset(
            mock_dataset,
            cache_strategy=CacheStrategy.MEMORY,
            cache_dir=str(self.cache_dir),
            cache_size_gb=0.001  # Very small cache
        )
        
        # Access many items
        for i in range(100):
            cached_dataset[i]
        
        # Cache should be limited
        self.assertLess(len(cached_dataset.cache), 100)

class TestDataLoaderFactory(unittest.TestCase):
    """Test DataLoader factory."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.factory = DataLoaderFactory(self.device_manager)
    
    def test_factory_initialization(self) -> Any:
        """Test factory initialization."""
        self.assertIsNotNone(self.factory.device_manager)
        self.assertIsNotNone(self.factory.logger)
    
    def test_create_dataloader(self) -> Any:
        """Test DataLoader creation."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 100
        
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True
        )
        
        dataloader = self.factory.create_dataloader(mock_dataset, config)
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertEqual(dataloader.batch_size, 16)
        self.assertEqual(dataloader.num_workers, 2)
        self.assertTrue(dataloader.pin_memory)
    
    def test_config_optimization(self) -> Any:
        """Test configuration optimization."""
        config = DataLoaderConfig(num_workers=-1)
        
        optimized_config = self.factory._optimize_config(config)
        
        # Should have auto-detected workers
        self.assertGreater(optimized_config.num_workers, 0)
        self.assertLessEqual(optimized_config.num_workers, 8)
    
    def test_sampler_creation(self) -> Any:
        """Test sampler creation."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 100
        
        # Test random sampler
        config = DataLoaderConfig(shuffle=True)
        sampler = self.factory._create_sampler(mock_dataset, config)
        self.assertIsInstance(sampler, torch.utils.data.RandomSampler)
        
        # Test sequential sampler
        config.shuffle = False
        sampler = self.factory._create_sampler(mock_dataset, config)
        self.assertIsInstance(sampler, torch.utils.data.SequentialSampler)
    
    def test_weighted_sampler(self) -> Any:
        """Test weighted sampler creation."""
        # Create mock dataset with weights
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 100
        mock_dataset.weights = [0.1] * 100  # Equal weights
        
        config = DataLoaderConfig(shuffle=True)
        
        dataloader = self.factory.create_weighted_dataloader(
            mock_dataset, config, mock_dataset.weights
        )
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)

class TestDataLoaderManager(unittest.TestCase):
    """Test DataLoader manager."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = Path(self.temp_dir) / "test_data.csv"
        
        # Create test dataset
        self.create_test_dataset()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self) -> Any:
        """Create synthetic test dataset."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Create text-like features
        texts = [f"Sample text {i} with features {x[:5]}" for i, x in enumerate(X)]
        
        # Save to CSV
        df = pd.DataFrame({
            'text': texts,
            'label': y
        })
        df.to_csv(self.test_data_path, index=False)
    
    async def test_manager_initialization(self) -> Any:
        """Test manager initialization."""
        manager = DataLoaderManager(self.device_manager)
        
        self.assertIsNotNone(manager.device_manager)
        self.assertIsNotNone(manager.factory)
        self.assertIsNotNone(manager.logger)
    
    async def test_load_csv_dataset(self) -> Any:
        """Test CSV dataset loading."""
        manager = DataLoaderManager(self.device_manager)
        
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            cache_strategy=CacheStrategy.MEMORY
        )
        
        dataset, dataloader = await manager.load_dataset(
            str(self.test_data_path),
            DataFormat.CSV,
            config
        )
        
        self.assertIsInstance(dataset, OptimizedTextDataset)
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertEqual(len(dataset), 200)
    
    async def test_load_json_dataset(self) -> Any:
        """Test JSON dataset loading."""
        manager = DataLoaderManager(self.device_manager)
        
        # Create JSON dataset
        json_data = [
            {'text': f'text_{i}', 'label': i % 3}
            for i in range(50)
        ]
        json_path = self.temp_dir / "test_data.json"
        
        with open(json_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(json_data, f)
        
        config = DataLoaderConfig(batch_size=8)
        
        dataset, dataloader = await manager.load_dataset(
            str(json_path),
            DataFormat.JSON,
            config
        )
        
        self.assertIsInstance(dataset, OptimizedTextDataset)
        self.assertEqual(len(dataset), 50)
    
    async def test_dataset_splitting(self) -> Any:
        """Test dataset splitting."""
        manager = DataLoaderManager(self.device_manager)
        
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 1000
        
        train_dataset, val_dataset, test_dataset = manager.split_dataset(
            mock_dataset,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        self.assertEqual(len(train_dataset), 700)
        self.assertEqual(len(val_dataset), 200)
        self.assertEqual(len(test_dataset), 100)
    
    async def test_dataloader_creation(self) -> Any:
        """Test DataLoader creation."""
        manager = DataLoaderManager(self.device_manager)
        
        # Create mock datasets
        train_dataset = Mock()
        train_dataset.__len__ = lambda: 100
        val_dataset = Mock()
        val_dataset.__len__ = lambda: 20
        test_dataset = Mock()
        test_dataset.__len__ = lambda: 20
        
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True
        )
        
        train_loader, val_loader, test_loader = manager.create_dataloaders(
            train_dataset, val_dataset, test_dataset, config
        )
        
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
    
    def test_format_detection(self) -> Any:
        """Test data format detection."""
        manager = DataLoaderManager(self.device_manager)
        
        # Test various formats
        test_cases = [
            ("data.csv", DataFormat.CSV),
            ("data.json", DataFormat.JSON),
            ("data.h5", DataFormat.HDF5),
            ("data.hdf5", DataFormat.HDF5),
            ("data.lmdb", DataFormat.LMDB),
            ("data.parquet", DataFormat.PARQUET),
            ("data.pkl", DataFormat.PICKLE),
            ("data.npy", DataFormat.NUMPY),
            ("data.unknown", DataFormat.CSV)  # Default
        ]
        
        for file_path, expected_format in test_cases:
            detected_format = manager._detect_data_format(file_path)
            self.assertEqual(detected_format, expected_format)

class TestDataLoaderProfiler(unittest.TestCase):
    """Test DataLoader profiler."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.profiler = DataLoaderProfiler()
    
    def test_profiler_initialization(self) -> Any:
        """Test profiler initialization."""
        self.assertIsNotNone(self.profiler.logger)
        self.assertEqual(self.profiler.metrics, {})
    
    def test_profiling(self) -> Any:
        """Test DataLoader profiling."""
        # Create mock dataset and DataLoader
        mock_dataset = Mock()
        mock_dataset.__len__ = lambda: 100
        
        # Create simple DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 3, (100,))
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,  # No workers for testing
            pin_memory=False
        )
        
        # Profile
        metrics = self.profiler.profile_dataloader(dataloader, num_batches=3)
        
        # Check metrics
        self.assertIn('total_time', metrics)
        self.assertIn('avg_batch_time', metrics)
        self.assertIn('throughput_batches_per_sec', metrics)
        self.assertIn('throughput_samples_per_sec', metrics)
        
        # Validate metrics
        self.assertGreater(metrics['total_time'], 0)
        self.assertGreater(metrics['avg_batch_time'], 0)
        self.assertGreater(metrics['throughput_batches_per_sec'], 0)
    
    def test_config_optimization(self) -> Any:
        """Test configuration optimization."""
        # Create current config
        current_config = DataLoaderConfig(
            batch_size=16,
            num_workers=2
        )
        
        # Set mock metrics
        self.profiler.metrics = {
            'throughput_samples_per_sec': 100  # Low throughput
        }
        
        # Optimize
        optimized_config = self.profiler.optimize_dataloader_config(
            current_config, target_throughput=200
        )
        
        # Should have increased workers or batch size
        self.assertGreaterEqual(
            optimized_config.num_workers + optimized_config.batch_size,
            current_config.num_workers + current_config.batch_size
        )

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = Path(self.temp_dir) / "test_data.csv"
        
        # Create test dataset
        self.create_test_dataset()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_dataset(self) -> Any:
        """Create synthetic test dataset."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=300,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        # Create text-like features
        texts = [f"Sample text {i} with features {x[:5]}" for i, x in enumerate(X)]
        
        # Save to CSV
        df = pd.DataFrame({
            'text': texts,
            'label': y
        })
        df.to_csv(self.test_data_path, index=False)
    
    async def test_end_to_end_workflow(self) -> Any:
        """Test end-to-end workflow."""
        # Create manager
        manager = DataLoaderManager(self.device_manager)
        
        # Load dataset
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            cache_strategy=CacheStrategy.MEMORY
        )
        
        dataset, dataloader = await manager.load_dataset(
            str(self.test_data_path),
            DataFormat.CSV,
            config
        )
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = manager.split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Create DataLoaders
        train_loader, val_loader, test_loader = manager.create_dataloaders(
            train_dataset, val_dataset, test_dataset, config
        )
        
        # Test DataLoaders
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Test batch loading
        for i, batch in enumerate(train_loader):
            self.assertIn('text', batch)
            self.assertIn('labels', batch)
            if i >= 2:  # Test first 3 batches
                break
        
        # Get statistics
        stats = manager.get_dataloader_stats(train_loader)
        self.assertIn('total_batches', stats)
        self.assertIn('total_samples', stats)
        self.assertIn('batch_size', stats)
    
    async def test_performance_profiling(self) -> Any:
        """Test performance profiling integration."""
        # Create manager and load dataset
        manager = DataLoaderManager(self.device_manager)
        
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2
        )
        
        dataset, dataloader = await manager.load_dataset(
            str(self.test_data_path),
            DataFormat.CSV,
            config
        )
        
        # Profile performance
        profiler = DataLoaderProfiler()
        metrics = profiler.profile_dataloader(dataloader, num_batches=5)
        
        # Validate metrics
        self.assertGreater(metrics['throughput_samples_per_sec'], 0)
        self.assertGreater(metrics['avg_batch_time'], 0)
        
        # Optimize configuration
        optimized_config = profiler.optimize_dataloader_config(
            config, target_throughput=1000
        )
        
        self.assertIsInstance(optimized_config, DataLoaderConfig)

# Performance benchmarks
class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self) -> Any:
        """Set up benchmark environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up benchmark environment."""
        shutil.rmtree(self.temp_dir)
    
    def benchmark_dataloader_creation(self) -> Any:
        """Benchmark DataLoader creation."""
        factory = DataLoaderFactory(self.device_manager)
        
        # Create large dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10000, 100),
            torch.randint(0, 10, (10000,))
        )
        
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=4,
            pin_memory=True
        )
        
        start_time = time.time()
        dataloader = factory.create_dataloader(dataset, config)
        creation_time = time.time() - start_time
        
        self.assertLess(creation_time, 1.0)  # Should complete within 1 second
        logger.info(f"DataLoader creation time: {creation_time:.4f} seconds")
    
    def benchmark_batch_loading(self) -> Any:
        """Benchmark batch loading performance."""
        # Create dataset and DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 50),
            torch.randint(0, 5, (1000,))
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True
        )
        
        # Benchmark loading
        start_time = time.time()
        batch_count = 0
        
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 20:  # Load 20 batches
                break
        
        total_time = time.time() - start_time
        throughput = batch_count / total_time
        
        self.assertGreater(throughput, 10)  # At least 10 batches per second
        logger.info(f"Batch loading throughput: {throughput:.2f} batches/sec")

# Error handling tests
class TestErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_invalid_data_path(self) -> Any:
        """Test handling of invalid data path."""
        manager = DataLoaderManager(self.device_manager)
        
        config = DataLoaderConfig()
        
        with self.assertRaises(FileNotFoundError):
            await manager.load_dataset(
                "nonexistent_file.csv",
                DataFormat.CSV,
                config
            )
    
    async def test_invalid_data_format(self) -> Any:
        """Test handling of invalid data format."""
        manager = DataLoaderManager(self.device_manager)
        
        # Create test file
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test data")
        
        config = DataLoaderConfig()
        
        # Should use default format (CSV)
        try:
            dataset, dataloader = await manager.load_dataset(
                str(test_file),
                DataFormat.CSV,
                config
            )
        except Exception as e:
            # Expected to fail with invalid CSV format
            self.assertIsInstance(e, (pd.errors.EmptyDataError, pd.errors.ParserError))

# Test runner functions
def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_dataloader_creation'))
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_batch_loading'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(benchmark_suite)

def run_all_tests():
    """Run all tests."""
    print("🧪 Running All DataLoader Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataLoaderConfig,
        TestOptimizedTextDataset,
        TestCachedDataset,
        TestDataLoaderFactory,
        TestDataLoaderManager,
        TestDataLoaderProfiler,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

# Quick test functions
async def quick_dataloader_test():
    """Quick test for DataLoader system."""
    print("🧪 Quick DataLoader Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    test_data_path = Path(temp_dir) / "test_data.csv"
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    texts = [f"Sample {i}" for i in range(len(X))]
    
    df = pd.DataFrame({'text': texts, 'label': y})
    df.to_csv(test_data_path, index=False)
    
    try:
        # Test DataLoader creation
        dataset, dataloader = await quick_dataloader(
            data_path=str(test_data_path),
            data_format=DataFormat.CSV,
            batch_size=16
        )
        
        print(f"✅ DataLoader test passed: {len(dataset)} samples, {len(dataloader)} batches")
        
        # Test batch loading
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch.keys()}")
            if i >= 2:  # Show first 3 batches
                break
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

async def quick_performance_test():
    """Quick performance test."""
    print("🧪 Quick Performance Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    test_data_path = Path(temp_dir) / "test_data.csv"
    
    # Generate larger test data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    texts = [f"Sample text {i} with features {x[:5]}" for i, x in enumerate(X)]
    
    df = pd.DataFrame({'text': texts, 'label': y})
    df.to_csv(test_data_path, index=False)
    
    try:
        # Create DataLoader
        dataset, dataloader = await quick_dataloader(
            data_path=str(test_data_path),
            data_format=DataFormat.CSV,
            batch_size=32,
            num_workers=4
        )
        
        # Profile performance
        profiler = DataLoaderProfiler()
        metrics = profiler.profile_dataloader(dataloader, num_batches=10)
        
        print(f"✅ Performance test passed:")
        print(f"  - Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  - Avg batch time: {metrics['avg_batch_time']:.4f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

# Example usage
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
print("🚀 Efficient Data Loading Test Suite")
        print("=" * 50)
        
        # Run quick tests
        print("\n1. Quick Tests:")
        dataloader_success = await quick_dataloader_test()
        performance_success = await quick_performance_test()
        
        # Run performance tests
        print("\n2. Performance Tests:")
        run_performance_tests()
        
        # Run comprehensive tests
        print("\n3. Comprehensive Tests:")
        all_tests_success = run_all_tests()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"DataLoader Test: {'✅ PASSED' if dataloader_success else '❌ FAILED'}")
        print(f"Performance Test: {'✅ PASSED' if performance_success else '❌ FAILED'}")
        print(f"All Tests: {'✅ PASSED' if all_tests_success else '❌ FAILED'}")
        
        if dataloader_success and performance_success and all_tests_success:
            print("\n🎉 All tests passed! The DataLoader system is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
    
    asyncio.run(main()) 