from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

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
from .model_training import (
from .model_evaluation import (
    import asyncio
from typing import Any, List, Dict, Optional
"""
🧪 Model Training & Evaluation Test Suite
=========================================

Comprehensive test suite for training and evaluation systems with
unit tests, integration tests, and performance benchmarks.
"""



# Import our systems
    ModelTrainer, TrainingConfig, ModelType, TrainingMode,
    CustomDataset, TrainingMetrics, EvaluationResult,
    HyperparameterOptimizer
)
    ModelEvaluator, ModelPerformance, CrossValidationResult,
    ModelComparison, ProductionEvaluationPipeline
)

logger = logging.getLogger(__name__)

class TestTrainingSystem(unittest.TestCase):
    """Test suite for training system."""
    
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
    
    def test_training_config(self) -> Any:
        """Test training configuration."""
        config = TrainingConfig(
            model_type=ModelType.TRANSFORMER,
            training_mode=TrainingMode.FINE_TUNE,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir
        )
        
        self.assertEqual(config.model_type, ModelType.TRANSFORMER)
        self.assertEqual(config.training_mode, TrainingMode.FINE_TUNE)
        self.assertEqual(config.model_name, "test-model")
        self.assertTrue(config.output_dir.exists())
    
    def test_custom_dataset(self) -> Any:
        """Test custom dataset."""
        # Load test data
        df = pd.read_csv(self.test_data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        dataset = CustomDataset(texts, labels)
        
        self.assertEqual(len(dataset), 1000)
        
        # Test single item
        item = dataset[0]
        self.assertIn('text', item)
        self.assertIn('labels', item)
        self.assertIsInstance(item['labels'], torch.Tensor)
    
    def test_model_trainer_initialization(self) -> Any:
        """Test model trainer initialization."""
        trainer = ModelTrainer(self.device_manager)
        
        self.assertIsNotNone(trainer.device)
        self.assertIsNotNone(trainer.logger)
        self.assertEqual(trainer.best_metric, float('inf'))
        self.assertEqual(trainer.patience_counter, 0)
    
    def test_metrics_calculation(self) -> Any:
        """Test metrics calculation."""
        trainer = ModelTrainer(self.device_manager)
        
        # Test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = trainer.calculate_metrics(y_true, y_pred, task_type="classification")
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIsInstance(metrics['accuracy'], float)
    
    @patch('torch.nn.Module')
    def test_create_model(self, mock_module) -> Any:
        """Test model creation."""
        trainer = ModelTrainer(self.device_manager)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir
        )
        
        model, tokenizer = trainer.create_model(config, num_classes=3)
        
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)
    
    def test_optimizer_creation(self) -> Any:
        """Test optimizer creation."""
        trainer = ModelTrainer(self.device_manager)
        
        # Create mock model
        model = nn.Linear(10, 3)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir
        )
        
        optimizer = trainer.create_optimizer(model, config)
        
        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
    
    def test_scheduler_creation(self) -> Any:
        """Test scheduler creation."""
        trainer = ModelTrainer(self.device_manager)
        
        # Create mock model and optimizer
        model = nn.Linear(10, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir
        )
        
        scheduler = trainer.create_scheduler(optimizer, config, num_training_steps=100)
        
        self.assertIsNotNone(scheduler)
    
    @patch('torch.utils.data.DataLoader')
    @patch('torch.nn.Module')
    async def test_train_epoch(self, mock_model, mock_dataloader) -> Any:
        """Test training epoch."""
        trainer = ModelTrainer(self.device_manager)
        
        # Mock model
        mock_model.return_value.parameters.return_value = []
        mock_model.return_value.train.return_value = None
        mock_model.return_value.to.return_value = mock_model.return_value
        
        # Mock dataloader
        mock_batch = {
            'input_ids': torch.randn(2, 10),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 3, (2,))
        }
        mock_dataloader.return_value = [mock_batch]
        
        # Mock optimizer and scheduler
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_scheduler.get_last_lr.return_value = [1e-3]
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir
        )
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.randn(2, 3)
        mock_model.return_value.return_value = mock_outputs
        
        metrics = await trainer.train_epoch(
            mock_model.return_value, mock_dataloader.return_value,
            mock_optimizer, mock_scheduler, config
        )
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
    
    @patch('torch.utils.data.DataLoader')
    @patch('torch.nn.Module')
    async def test_validate_epoch(self, mock_model, mock_dataloader) -> bool:
        """Test validation epoch."""
        trainer = ModelTrainer(self.device_manager)
        
        # Mock model
        mock_model.return_value.eval.return_value = None
        mock_model.return_value.to.return_value = mock_model.return_value
        
        # Mock dataloader
        mock_batch = {
            'input_ids': torch.randn(2, 10),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 3, (2,))
        }
        mock_dataloader.return_value = [mock_batch]
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.randn(2, 3)
        mock_model.return_value.return_value = mock_outputs
        
        metrics = await trainer.validate_epoch(
            mock_model.return_value, mock_dataloader.return_value
        )
        
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)

class TestEvaluationSystem(unittest.TestCase):
    """Test suite for evaluation system."""
    
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
    
    def test_model_evaluator_initialization(self) -> Any:
        """Test model evaluator initialization."""
        evaluator = ModelEvaluator(self.device_manager)
        
        self.assertIsNotNone(evaluator.device)
        self.assertIsNotNone(evaluator.logger)
    
    def test_advanced_metrics_calculation(self) -> Any:
        """Test advanced metrics calculation."""
        evaluator = ModelEvaluator(self.device_manager)
        
        # Test classification metrics
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_prob = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6],
            [0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.1, 0.9]
        ])
        
        metrics = evaluator.calculate_advanced_metrics(
            y_true, y_pred, y_prob, task_type="classification"
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('cohen_kappa', metrics)
        self.assertIn('matthews_corr', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('auc_roc', metrics)
    
    def test_regression_metrics_calculation(self) -> Any:
        """Test regression metrics calculation."""
        evaluator = ModelEvaluator(self.device_manager)
        
        # Test regression metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluator.calculate_advanced_metrics(
            y_true, y_pred, task_type="regression"
        )
        
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
    
    def test_statistical_tests(self) -> Any:
        """Test statistical significance tests."""
        evaluator = ModelEvaluator(self.device_manager)
        
        # Test data
        model_scores = {
            'model1': [0.85, 0.87, 0.86, 0.84, 0.88],
            'model2': [0.83, 0.85, 0.84, 0.82, 0.86]
        }
        
        tests = evaluator.perform_statistical_tests(model_scores)
        
        self.assertIn('paired_t_test', tests)
        self.assertIn('wilcoxon_test', tests)
        self.assertIsInstance(tests['paired_t_test'], float)
        self.assertIsInstance(tests['wilcoxon_test'], float)
    
    def test_model_performance_dataclass(self) -> Any:
        """Test model performance dataclass."""
        performance = ModelPerformance(
            model_name="test-model",
            accuracy=0.85,
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            auc_roc=0.92,
            inference_time_ms=15.5,
            memory_usage_mb=512.0,
            model_size_mb=256.0
        )
        
        self.assertEqual(performance.model_name, "test-model")
        self.assertEqual(performance.accuracy, 0.85)
        self.assertEqual(performance.f1_score, 0.85)
        self.assertEqual(performance.inference_time_ms, 15.5)
    
    def test_cross_validation_result_dataclass(self) -> Any:
        """Test cross-validation result dataclass."""
        cv_result = CrossValidationResult(
            model_name="test-model",
            cv_scores=[0.85, 0.87, 0.86, 0.84, 0.88],
            mean_score=0.86,
            std_score=0.015,
            fold_results=[
                {'fold': 1, 'f1_score': 0.85, 'accuracy': 0.84},
                {'fold': 2, 'f1_score': 0.87, 'accuracy': 0.86}
            ],
            best_fold=5,
            worst_fold=4
        )
        
        self.assertEqual(cv_result.model_name, "test-model")
        self.assertEqual(cv_result.mean_score, 0.86)
        self.assertEqual(cv_result.best_fold, 5)
        self.assertEqual(cv_result.worst_fold, 4)
    
    def test_model_comparison_dataclass(self) -> Any:
        """Test model comparison dataclass."""
        comparison = ModelComparison(
            models=["model1", "model2", "model3"],
            metrics={
                'accuracy': [0.85, 0.87, 0.86],
                'f1_score': [0.84, 0.86, 0.85]
            },
            statistical_tests={
                'paired_t_test': 0.05,
                'wilcoxon_test': 0.03
            },
            ranking=[("model2", 0.87), ("model3", 0.86), ("model1", 0.85)],
            best_model="model2"
        )
        
        self.assertEqual(comparison.models, ["model1", "model2", "model3"])
        self.assertEqual(comparison.best_model, "model2")
        self.assertEqual(len(comparison.ranking), 3)

class TestIntegration(unittest.TestCase):
    """Integration tests for training and evaluation systems."""
    
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
            n_classes=2,
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
    
    @patch('torch.nn.Module')
    @patch('torch.utils.data.DataLoader')
    async def test_training_evaluation_integration(self, mock_dataloader, mock_model) -> Any:
        """Test integration between training and evaluation."""
        # Create trainer and evaluator
        trainer = ModelTrainer(self.device_manager)
        evaluator = ModelEvaluator(self.device_manager)
        
        # Mock model
        mock_model.return_value.parameters.return_value = []
        mock_model.return_value.train.return_value = None
        mock_model.return_value.eval.return_value = None
        mock_model.return_value.to.return_value = mock_model.return_value
        
        # Mock dataloader
        mock_batch = {
            'input_ids': torch.randn(2, 10),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 2, (2,))
        }
        mock_dataloader.return_value = [mock_batch]
        
        # Mock model forward pass
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.randn(2, 2)
        mock_model.return_value.return_value = mock_outputs
        
        # Create config
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir,
            num_epochs=1
        )
        
        # Test training
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        mock_scheduler.get_last_lr.return_value = [1e-3]
        
        train_metrics = await trainer.train_epoch(
            mock_model.return_value, mock_dataloader.return_value,
            mock_optimizer, mock_scheduler, config
        )
        
        # Test evaluation
        val_metrics = await trainer.validate_epoch(
            mock_model.return_value, mock_dataloader.return_value
        )
        
        # Verify metrics
        self.assertIn('loss', train_metrics)
        self.assertIn('accuracy', train_metrics)
        self.assertIn('loss', val_metrics)
        self.assertIn('accuracy', val_metrics)
    
    def test_hyperparameter_optimizer_integration(self) -> Any:
        """Test hyperparameter optimizer integration."""
        trainer = ModelTrainer(self.device_manager)
        optimizer = HyperparameterOptimizer(trainer)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(self.test_data_path),
            output_dir=self.temp_dir,
            enable_hpo=True,
            hpo_trials=2,
            num_epochs=1
        )
        
        # Test optimizer creation
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.trainer, trainer)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for training and evaluation systems."""
    
    def setUp(self) -> Any:
        """Set up benchmark environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up benchmark environment."""
        shutil.rmtree(self.temp_dir)
    
    def benchmark_metrics_calculation(self) -> Any:
        """Benchmark metrics calculation performance."""
        evaluator = ModelEvaluator(self.device_manager)
        
        # Large dataset
        n_samples = 100000
        y_true = np.random.randint(0, 3, n_samples)
        y_pred = np.random.randint(0, 3, n_samples)
        y_prob = np.random.rand(n_samples, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        start_time = time.time()
        metrics = evaluator.calculate_advanced_metrics(
            y_true, y_pred, y_prob, task_type="classification"
        )
        end_time = time.time()
        
        calculation_time = end_time - start_time
        self.assertLess(calculation_time, 1.0)  # Should complete within 1 second
        
        logger.info(f"Metrics calculation time: {calculation_time:.4f} seconds")
    
    def benchmark_model_size_calculation(self) -> Any:
        """Benchmark model size calculation."""
        evaluator = ModelEvaluator(self.device_manager)
        
        # Create large model
        model = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 10)
        )
        
        start_time = time.time()
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size / (1024 * 1024)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        self.assertLess(calculation_time, 0.1)  # Should complete within 0.1 seconds
        
        logger.info(f"Model size calculation time: {calculation_time:.4f} seconds")
        logger.info(f"Model size: {model_size_mb:.2f} MB")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_dataset_path(self) -> Any:
        """Test handling of invalid dataset path."""
        trainer = ModelTrainer(self.device_manager)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path="nonexistent_file.csv",
            output_dir=self.temp_dir
        )
        
        with self.assertRaises(FileNotFoundError):
            trainer.load_dataset(config)
    
    def test_empty_dataset(self) -> Any:
        """Test handling of empty dataset."""
        # Create empty dataset
        empty_data_path = Path(self.temp_dir) / "empty_data.csv"
        df = pd.DataFrame(columns=['text', 'label'])
        df.to_csv(empty_data_path, index=False)
        
        trainer = ModelTrainer(self.device_manager)
        
        config = TrainingConfig(
            model_type=ModelType.CUSTOM,
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path=str(empty_data_path),
            output_dir=self.temp_dir
        )
        
        with self.assertRaises(ValueError):
            trainer.load_dataset(config)
    
    def test_invalid_model_type(self) -> Any:
        """Test handling of invalid model type."""
        trainer = ModelTrainer(self.device_manager)
        
        config = TrainingConfig(
            model_type="invalid_type",  # Invalid type
            training_mode=TrainingMode.FROM_SCRATCH,
            model_name="test-model",
            dataset_path="dummy.csv",
            output_dir=self.temp_dir
        )
        
        with self.assertRaises(ValueError):
            trainer.create_model(config, num_classes=3)

# Performance test runner
def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_metrics_calculation'))
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_model_size_calculation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(benchmark_suite)

# Main test runner
def run_all_tests():
    """Run all tests."""
    print("🧪 Running All Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTrainingSystem,
        TestEvaluationSystem,
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
async def quick_training_test():
    """Quick test for training system."""
    print("🧪 Quick Training Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    test_data_path = Path(temp_dir) / "test_data.csv"
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    texts = [f"Sample {i}" for i in range(len(X))]
    
    df = pd.DataFrame({'text': texts, 'label': y})
    df.to_csv(test_data_path, index=False)
    
    try:
        # Test training
        result = await quick_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path=str(test_data_path),
            output_dir=temp_dir,
            num_epochs=1
        )
        
        print(f"✅ Training test passed: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

async def quick_evaluation_test():
    """Quick test for evaluation system."""
    print("🧪 Quick Evaluation Test...")
    
    # Create test data
    temp_dir = tempfile.mkdtemp()
    test_data_path = Path(temp_dir) / "test_data.csv"
    
    # Generate test data
    X, y = make_classification(n_samples=50, n_features=10, n_classes=2, random_state=42)
    texts = [f"Sample {i}" for i in range(len(X))]
    
    df = pd.DataFrame({'text': texts, 'label': y})
    df.to_csv(test_data_path, index=False)
    
    try:
        # Test evaluation (mock model path)
        mock_model_path = Path(temp_dir) / "mock_model.pth"
        torch.save({'dummy': 'data'}, mock_model_path)
        
        result = await quick_model_evaluation(
            model_path=str(mock_model_path),
            test_dataset_path=str(test_data_path),
            output_dir=temp_dir
        )
        
        print(f"✅ Evaluation test passed: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False
    
    finally:
        shutil.rmtree(temp_dir)

# Example usage
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
print("🚀 Model Training & Evaluation Test Suite")
        print("=" * 50)
        
        # Run quick tests
        print("\n1. Quick Tests:")
        training_success = await quick_training_test()
        evaluation_success = await quick_evaluation_test()
        
        # Run performance tests
        print("\n2. Performance Tests:")
        run_performance_tests()
        
        # Run comprehensive tests
        print("\n3. Comprehensive Tests:")
        all_tests_success = run_all_tests()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"Training Test: {'✅ PASSED' if training_success else '❌ FAILED'}")
        print(f"Evaluation Test: {'✅ PASSED' if evaluation_success else '❌ FAILED'}")
        print(f"All Tests: {'✅ PASSED' if all_tests_success else '❌ FAILED'}")
        
        if training_success and evaluation_success and all_tests_success:
            print("\n🎉 All tests passed! The system is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
    
    asyncio.run(main()) 