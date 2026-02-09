from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

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
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from .production_transformers import DeviceManager
from .data_splitting_cv import (
from .efficient_data_loader import (
        from torch.utils.data import Subset
    import asyncio
from typing import Any, List, Dict, Optional
"""
🧪 Data Splitting & Cross-Validation Test Suite
===============================================

Comprehensive test suite for data splitting and cross-validation systems with
unit tests, integration tests, and performance benchmarks.
"""



# Import our systems
    DataSplitter, CrossValidator, DataSplittingManager,
    SplitConfig, SplitStrategy, CrossValidationConfig, CrossValidationStrategy,
    SplitResult, CrossValidationResult
)
    DataLoaderManager, DataLoaderConfig, DataFormat, CacheStrategy,
    OptimizedTextDataset
)

logger = logging.getLogger(__name__)

class TestSplitConfig(unittest.TestCase):
    """Test split configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = SplitConfig()
        
        self.assertEqual(config.strategy, SplitStrategy.STRATIFIED)
        self.assertEqual(config.train_ratio, 0.7)
        self.assertEqual(config.val_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)
        self.assertEqual(config.random_state, 42)
        self.assertTrue(config.shuffle)
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = SplitConfig(
            strategy=SplitStrategy.RANDOM,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_state=123,
            shuffle=False
        )
        
        self.assertEqual(config.strategy, SplitStrategy.RANDOM)
        self.assertEqual(config.train_ratio, 0.8)
        self.assertEqual(config.val_ratio, 0.1)
        self.assertEqual(config.test_ratio, 0.1)
        self.assertEqual(config.random_state, 123)
        self.assertFalse(config.shuffle)
    
    def test_invalid_ratios(self) -> Any:
        """Test validation of split ratios."""
        with self.assertRaises(ValueError):
            SplitConfig(
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1.0
            )

class TestCrossValidationConfig(unittest.TestCase):
    """Test cross-validation configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = CrossValidationConfig()
        
        self.assertEqual(config.strategy, CrossValidationStrategy.STRATIFIED_K_FOLD)
        self.assertEqual(config.n_splits, 5)
        self.assertEqual(config.n_repeats, 3)
        self.assertEqual(config.random_state, 42)
        self.assertTrue(config.shuffle)
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = CrossValidationConfig(
            strategy=CrossValidationStrategy.K_FOLD,
            n_splits=10,
            n_repeats=5,
            random_state=123,
            shuffle=False
        )
        
        self.assertEqual(config.strategy, CrossValidationStrategy.K_FOLD)
        self.assertEqual(config.n_splits, 10)
        self.assertEqual(config.n_repeats, 5)
        self.assertEqual(config.random_state, 123)
        self.assertFalse(config.shuffle)

class TestDataSplitter(unittest.TestCase):
    """Test data splitter."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.splitter = DataSplitter(self.device_manager)
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
            n_samples=1000,
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
    
    def test_random_split(self) -> Any:
        """Test random split strategy."""
        # Create dataset
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        config = SplitConfig(
            strategy=SplitStrategy.RANDOM,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # Split dataset
        result = self.splitter.split_dataset(dataset, config)
        
        # Validate results
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(len(result.train_indices), 700)
        self.assertEqual(len(result.val_indices), 150)
        self.assertEqual(len(result.test_indices), 150)
        self.assertEqual(result.split_info['strategy'], 'random')
    
    def test_stratified_split(self) -> Any:
        """Test stratified split strategy."""
        # Create dataset
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # Split dataset
        result = self.splitter.split_dataset(dataset, config)
        
        # Validate results
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(len(result.train_indices), 700)
        self.assertEqual(len(result.val_indices), 150)
        self.assertEqual(len(result.test_indices), 150)
        self.assertEqual(result.split_info['strategy'], 'stratified')
        
        # Check class distribution
        self.assertIn('class_distribution', result.split_info)
    
    def test_time_series_split(self) -> Any:
        """Test time series split strategy."""
        # Create dataset with time information
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Add time column
        for i, item in enumerate(texts):
            if isinstance(item, dict):
                item['timestamp'] = i
            else:
                texts[i] = {'text': item, 'timestamp': i}
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        config = SplitConfig(
            strategy=SplitStrategy.TIME_SERIES,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            time_column='timestamp'
        )
        
        # Split dataset
        result = self.splitter.split_dataset(dataset, config)
        
        # Validate results
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(result.split_info['strategy'], 'time_series')
    
    def test_group_split(self) -> Any:
        """Test group split strategy."""
        # Create dataset with group information
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Add group column
        for i, item in enumerate(texts):
            if isinstance(item, dict):
                item['group'] = i % 10  # 10 groups
            else:
                texts[i] = {'text': item, 'group': i % 10}
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        config = SplitConfig(
            strategy=SplitStrategy.GROUP,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_column='group'
        )
        
        # Split dataset
        result = self.splitter.split_dataset(dataset, config)
        
        # Validate results
        self.assertIsInstance(result, SplitResult)
        self.assertEqual(result.split_info['strategy'], 'group')
    
    def test_split_result_methods(self) -> Any:
        """Test SplitResult methods."""
        # Create mock split result
        result = SplitResult(
            train_indices=[1, 2, 3],
            val_indices=[4, 5],
            test_indices=[6, 7, 8],
            train_dataset=Mock(),
            val_dataset=Mock(),
            test_dataset=Mock(),
            split_info={'strategy': 'test'}
        )
        
        # Test methods
        sizes = result.get_split_sizes()
        self.assertEqual(sizes['train'], 3)
        self.assertEqual(sizes['val'], 2)
        self.assertEqual(sizes['test'], 3)
        self.assertEqual(sizes['total'], 8)
        
        ratios = result.get_split_ratios()
        self.assertEqual(ratios['train'], 3/8)
        self.assertEqual(ratios['val'], 2/8)
        self.assertEqual(ratios['test'], 3/8)

class TestCrossValidator(unittest.TestCase):
    """Test cross-validator."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.cross_validator = CrossValidator(self.device_manager)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cv_splits_creation(self) -> Any:
        """Test CV splits creation."""
        # Create test dataset
        texts = [f"Sample {i}" for i in range(100)]
        labels = [i % 3 for i in range(100)]  # 3 classes
        dataset = OptimizedTextDataset(texts, labels)
        
        # Test K-fold splits
        config = CrossValidationConfig(
            strategy=CrossValidationStrategy.K_FOLD,
            n_splits=5
        )
        
        splits = self.cross_validator._create_cv_splits(dataset, config)
        self.assertEqual(len(splits), 5)
        
        # Test stratified K-fold splits
        config.strategy = CrossValidationStrategy.STRATIFIED_K_FOLD
        splits = self.cross_validator._create_cv_splits(dataset, config)
        self.assertEqual(len(splits), 5)
    
    def test_cv_statistics_calculation(self) -> Any:
        """Test CV statistics calculation."""
        # Mock fold results
        fold_results = [
            {'val_f1_score': 0.8, 'val_accuracy': 0.85, 'train_f1_score': 0.9, 'train_accuracy': 0.92},
            {'val_f1_score': 0.82, 'val_accuracy': 0.87, 'train_f1_score': 0.91, 'train_accuracy': 0.93},
            {'val_f1_score': 0.78, 'val_accuracy': 0.83, 'train_f1_score': 0.89, 'train_accuracy': 0.91}
        ]
        
        mean_scores, std_scores = self.cross_validator._calculate_cv_statistics(fold_results)
        
        self.assertIn('val_f1_score', mean_scores)
        self.assertIn('val_accuracy', mean_scores)
        self.assertIn('val_f1_score', std_scores)
        self.assertIn('val_accuracy', std_scores)
        
        self.assertAlmostEqual(mean_scores['val_f1_score'], 0.8, places=2)
        self.assertAlmostEqual(mean_scores['val_accuracy'], 0.85, places=2)

class TestDataSplittingManager(unittest.TestCase):
    """Test data splitting manager."""
    
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
            n_samples=500,
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
        manager = DataSplittingManager(self.device_manager)
        
        self.assertIsNotNone(manager.device_manager)
        self.assertIsNotNone(manager.splitter)
        self.assertIsNotNone(manager.cross_validator)
        self.assertIsNotNone(manager.logger)
    
    async def test_split_and_validate(self) -> bool:
        """Test split and validate functionality."""
        manager = DataSplittingManager(self.device_manager)
        
        # Create dataset
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        split_config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Create CV config
        cv_config = CrossValidationConfig(
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
            n_splits=3
        )
        
        # Create data loader manager
        data_loader_manager = DataLoaderManager(self.device_manager)
        
        # Split and validate
        result = await manager.split_and_validate(
            dataset, split_config, cv_config, data_loader_manager
        )
        
        # Validate results
        self.assertIn('split_result', result)
        self.assertIn('split_info', result)
        self.assertIn('split_sizes', result)
        self.assertIn('split_ratios', result)
        self.assertIn('cv_result', result)
        self.assertIn('cv_summary', result)
    
    async def test_create_dataloaders_from_split(self) -> Any:
        """Test creating DataLoaders from split."""
        manager = DataSplittingManager(self.device_manager)
        
        # Create mock split result
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        mock_test_dataset = Mock()
        
        split_result = SplitResult(
            train_indices=[1, 2, 3],
            val_indices=[4, 5],
            test_indices=[6, 7, 8],
            train_dataset=mock_train_dataset,
            val_dataset=mock_val_dataset,
            test_dataset=mock_test_dataset,
            split_info={'strategy': 'test'}
        )
        
        # Create DataLoader config
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True
        )
        
        # Create DataLoaders
        train_loader, val_loader, test_loader = manager.create_dataloaders_from_split(
            split_result, config
        )
        
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
    
    def test_analyze_split_quality(self) -> Any:
        """Test split quality analysis."""
        manager = DataSplittingManager(self.device_manager)
        
        # Create mock split result with labels
        
        # Create dataset with known labels
        texts = ["text1", "text2", "text3", "text4", "text5"]
        labels = [0, 1, 0, 1, 0]  # Imbalanced
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create subsets
        train_subset = Subset(dataset, [0, 1, 2])  # 2 class 0, 1 class 1
        val_subset = Subset(dataset, [3])  # 1 class 1
        test_subset = Subset(dataset, [4])  # 1 class 0
        
        split_result = SplitResult(
            train_indices=[0, 1, 2],
            val_indices=[3],
            test_indices=[4],
            train_dataset=train_subset,
            val_dataset=val_subset,
            test_dataset=test_subset,
            split_info={'strategy': 'test'}
        )
        
        # Analyze quality
        quality = manager.analyze_split_quality(split_result)
        
        self.assertIn('class_distributions', quality)
        self.assertIn('class_coverage', quality)
        self.assertIn('distribution_similarity', quality)
        
        # Check distributions
        train_dist = quality['class_distributions']['train']
        self.assertEqual(train_dist[0], 2)  # 2 class 0
        self.assertEqual(train_dist[1], 1)  # 1 class 1

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
        manager = DataSplittingManager(self.device_manager)
        
        # Create dataset
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create split config
        split_config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Split dataset
        split_result = manager.splitter.split_dataset(dataset, split_config)
        
        # Analyze split quality
        quality = manager.analyze_split_quality(split_result)
        
        # Create DataLoaders
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True
        )
        
        train_loader, val_loader, test_loader = manager.create_dataloaders_from_split(
            split_result, config
        )
        
        # Validate results
        self.assertIsInstance(split_result, SplitResult)
        self.assertIn('distribution_similarity', quality)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Check split sizes
        sizes = split_result.get_split_sizes()
        self.assertEqual(sizes['train'], 210)  # 70% of 300
        self.assertEqual(sizes['val'], 45)     # 15% of 300
        self.assertEqual(sizes['test'], 45)    # 15% of 300
    
    async def test_cross_validation_workflow(self) -> Any:
        """Test cross-validation workflow."""
        # Create manager
        manager = DataSplittingManager(self.device_manager)
        
        # Create dataset
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        dataset = OptimizedTextDataset(texts, labels)
        
        # Create CV config
        cv_config = CrossValidationConfig(
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
            n_splits=3
        )
        
        # Create data loader manager
        data_loader_manager = DataLoaderManager(self.device_manager)
        
        # Perform cross-validation
        cv_result = await manager.cross_validator.cross_validate(
            dataset, cv_config, None, None, data_loader_manager
        )
        
        # Validate results
        self.assertIsInstance(cv_result, CrossValidationResult)
        self.assertEqual(len(cv_result.fold_results), 3)
        self.assertIn('mean_scores', cv_result.mean_scores)
        self.assertIn('std_scores', cv_result.std_scores)
        
        # Check summary
        summary = cv_result.get_summary()
        self.assertIn('mean_scores', summary)
        self.assertIn('std_scores', summary)
        self.assertIn('best_fold', summary)
        self.assertIn('worst_fold', summary)
        self.assertIn('n_folds', summary)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self) -> Any:
        """Set up benchmark environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up benchmark environment."""
        shutil.rmtree(self.temp_dir)
    
    def benchmark_splitting_performance(self) -> Any:
        """Benchmark splitting performance."""
        splitter = DataSplitter(self.device_manager)
        
        # Create large dataset
        texts = [f"Sample text {i}" for i in range(10000)]
        labels = [i % 5 for i in range(10000)]  # 5 classes
        dataset = OptimizedTextDataset(texts, labels)
        
        config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        start_time = time.time()
        result = splitter.split_dataset(dataset, config)
        splitting_time = time.time() - start_time
        
        self.assertLess(splitting_time, 1.0)  # Should complete within 1 second
        logger.info(f"Splitting time: {splitting_time:.4f} seconds")
    
    def benchmark_cv_splits_creation(self) -> Any:
        """Benchmark CV splits creation."""
        cross_validator = CrossValidator(self.device_manager)
        
        # Create large dataset
        texts = [f"Sample text {i}" for i in range(5000)]
        labels = [i % 3 for i in range(5000)]  # 3 classes
        dataset = OptimizedTextDataset(texts, labels)
        
        config = CrossValidationConfig(
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
            n_splits=10
        )
        
        start_time = time.time()
        splits = cross_validator._create_cv_splits(dataset, config)
        creation_time = time.time() - start_time
        
        self.assertLess(creation_time, 0.5)  # Should complete within 0.5 seconds
        self.assertEqual(len(splits), 10)
        logger.info(f"CV splits creation time: {creation_time:.4f} seconds")

class TestErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_split_ratios(self) -> Any:
        """Test handling of invalid split ratios."""
        with self.assertRaises(ValueError):
            SplitConfig(
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1.0
            )
    
    def test_missing_time_column(self) -> Any:
        """Test handling of missing time column."""
        splitter = DataSplitter(self.device_manager)
        
        # Create dataset
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        dataset = OptimizedTextDataset(texts, labels)
        
        config = SplitConfig(
            strategy=SplitStrategy.TIME_SERIES,
            time_column=None  # Missing time column
        )
        
        with self.assertRaises(ValueError):
            splitter.split_dataset(dataset, config)
    
    def test_missing_group_column(self) -> Any:
        """Test handling of missing group column."""
        splitter = DataSplitter(self.device_manager)
        
        # Create dataset
        texts = ["text1", "text2", "text3"]
        labels = [0, 1, 0]
        dataset = OptimizedTextDataset(texts, labels)
        
        config = SplitConfig(
            strategy=SplitStrategy.GROUP,
            group_column=None  # Missing group column
        )
        
        with self.assertRaises(ValueError):
            splitter.split_dataset(dataset, config)

# Test runner functions
def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_splitting_performance'))
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_cv_splits_creation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(benchmark_suite)

def run_all_tests():
    """Run all tests."""
    print("🧪 Running All Data Splitting & CV Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSplitConfig,
        TestCrossValidationConfig,
        TestDataSplitter,
        TestCrossValidator,
        TestDataSplittingManager,
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
async def quick_split_test():
    """Quick test for data splitting."""
    print("🧪 Quick Data Splitting Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dataset
        texts = [f"Sample {i}" for i in range(100)]
        labels = [i % 3 for i in range(100)]  # 3 classes
        dataset = OptimizedTextDataset(texts, labels)
        
        # Test splitting
        split_result = await quick_split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            strategy=SplitStrategy.STRATIFIED
        )
        
        print(f"✅ Split test passed: {split_result.get_split_sizes()}")
        return True
        
    except Exception as e:
        print(f"❌ Split test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

async def quick_cv_test():
    """Quick test for cross-validation."""
    print("🧪 Quick Cross-Validation Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dataset
        texts = [f"Sample {i}" for i in range(100)]
        labels = [i % 3 for i in range(100)]  # 3 classes
        dataset = OptimizedTextDataset(texts, labels)
        
        # Test cross-validation
        cv_result = await quick_cross_validate(
            dataset,
            n_splits=3,
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD
        )
        
        print(f"✅ CV test passed: {cv_result.get_summary()}")
        return True
        
    except Exception as e:
        print(f"❌ CV test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

# Example usage
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
print("🚀 Data Splitting & Cross-Validation Test Suite")
        print("=" * 60)
        
        # Run quick tests
        print("\n1. Quick Tests:")
        split_success = await quick_split_test()
        cv_success = await quick_cv_test()
        
        # Run performance tests
        print("\n2. Performance Tests:")
        run_performance_tests()
        
        # Run comprehensive tests
        print("\n3. Comprehensive Tests:")
        all_tests_success = run_all_tests()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Test Summary:")
        print(f"Split Test: {'✅ PASSED' if split_success else '❌ FAILED'}")
        print(f"CV Test: {'✅ PASSED' if cv_success else '❌ FAILED'}")
        print(f"All Tests: {'✅ PASSED' if all_tests_success else '❌ FAILED'}")
        
        if split_success and cv_success and all_tests_success:
            print("\n🎉 All tests passed! The Data Splitting & CV system is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
    
    asyncio.run(main()) 