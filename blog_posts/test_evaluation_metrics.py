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
from .evaluation_metrics import (
    import asyncio
from typing import Any, List, Dict, Optional
"""
🧪 Evaluation Metrics Test Suite
================================

Comprehensive test suite for evaluation metrics system with unit tests,
integration tests, and performance benchmarks.
"""



# Import our systems
    EvaluationMetrics, MetricConfig, TaskType, MetricType, EvaluationResult,
    create_evaluation_metrics, create_metric_config
)

logger = logging.getLogger(__name__)

class TestMetricConfig(unittest.TestCase):
    """Test metric configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration."""
        config = MetricConfig(task_type=TaskType.CLASSIFICATION)
        
        self.assertEqual(config.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(config.average, "weighted")
        self.assertEqual(config.beta, 1.0)
        self.assertEqual(config.k, 5)
        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(len(config.metric_types), 6)  # Default metrics for classification
    
    def test_custom_config(self) -> Any:
        """Test custom configuration."""
        config = MetricConfig(
            task_type=TaskType.REGRESSION,
            metric_types=[MetricType.MSE, MetricType.MAE, MetricType.R2],
            average="macro",
            beta=2.0,
            k=10,
            threshold=0.7
        )
        
        self.assertEqual(config.task_type, TaskType.REGRESSION)
        self.assertEqual(config.average, "macro")
        self.assertEqual(config.beta, 2.0)
        self.assertEqual(config.k, 10)
        self.assertEqual(config.threshold, 0.7)
        self.assertEqual(len(config.metric_types), 3)
    
    def test_default_metrics_by_task(self) -> Any:
        """Test default metrics for different task types."""
        # Classification
        config = MetricConfig(task_type=TaskType.CLASSIFICATION)
        expected_metrics = {
            MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
            MetricType.F1, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX
        }
        self.assertEqual(set(config.metric_types), expected_metrics)
        
        # Regression
        config = MetricConfig(task_type=TaskType.REGRESSION)
        expected_metrics = {
            MetricType.MSE, MetricType.MAE, MetricType.R2,
            MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR
        }
        self.assertEqual(set(config.metric_types), expected_metrics)
        
        # Generation
        config = MetricConfig(task_type=TaskType.GENERATION)
        expected_metrics = {
            MetricType.BLEU, MetricType.ROUGE, MetricType.METEOR,
            MetricType.PERPLEXITY
        }
        self.assertEqual(set(config.metric_types), expected_metrics)

class TestEvaluationResult(unittest.TestCase):
    """Test evaluation result."""
    
    def test_evaluation_result_creation(self) -> Any:
        """Test evaluation result creation."""
        result = EvaluationResult(
            task_type=TaskType.CLASSIFICATION,
            metrics={'accuracy': 0.85, 'f1': 0.82},
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            classification_report="Classification Report",
            detailed_metrics={'precision': 0.83, 'recall': 0.81},
            metadata={'num_samples': 100}
        )
        
        self.assertEqual(result.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(result.metrics['accuracy'], 0.85)
        self.assertEqual(result.metrics['f1'], 0.82)
        self.assertIsNotNone(result.confusion_matrix)
        self.assertEqual(result.classification_report, "Classification Report")
        self.assertEqual(result.detailed_metrics['precision'], 0.83)
        self.assertEqual(result.metadata['num_samples'], 100)
    
    def test_to_dict(self) -> Any:
        """Test conversion to dictionary."""
        result = EvaluationResult(
            task_type=TaskType.CLASSIFICATION,
            metrics={'accuracy': 0.85},
            confusion_matrix=np.array([[50, 10], [5, 35]])
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['task_type'], 'classification')
        self.assertEqual(result_dict['metrics']['accuracy'], 0.85)
        self.assertIsInstance(result_dict['confusion_matrix'], list)
    
    def test_save_and_load(self) -> Any:
        """Test saving and loading evaluation result."""
        result = EvaluationResult(
            task_type=TaskType.CLASSIFICATION,
            metrics={'accuracy': 0.85, 'f1': 0.82},
            confusion_matrix=np.array([[50, 10], [5, 35]]),
            classification_report="Test Report"
        )
        
        # Save
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        
        try:
            result.save(temp_file.name)
            
            # Load
            loaded_result = EvaluationResult.load(temp_file.name)
            
            self.assertEqual(loaded_result.task_type, result.task_type)
            self.assertEqual(loaded_result.metrics['accuracy'], result.metrics['accuracy'])
            self.assertEqual(loaded_result.metrics['f1'], result.metrics['f1'])
            self.assertTrue(np.array_equal(loaded_result.confusion_matrix, result.confusion_matrix))
            self.assertEqual(loaded_result.classification_report, result.classification_report)
        finally:
            Path(temp_file.name).unlink()

class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics implementation."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_classification_metrics(self) -> Any:
        """Test classification metrics calculation."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create test data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                 n_informative=10, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 3, 1000)
        y_prob = np.random.rand(1000, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        
        # Check results
        self.assertIn('accuracy', result.metrics)
        self.assertIn('precision', result.metrics)
        self.assertIn('recall', result.metrics)
        self.assertIn('f1', result.metrics)
        self.assertIn('roc_auc', result.metrics)
        self.assertIsNotNone(result.confusion_matrix)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(result.metrics['accuracy'], 0.0)
        self.assertLessEqual(result.metrics['accuracy'], 1.0)
        self.assertGreaterEqual(result.metrics['f1'], 0.0)
        self.assertLessEqual(result.metrics['f1'], 1.0)
    
    async def test_regression_metrics(self) -> Any:
        """Test regression metrics calculation."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create test data
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = y + np.random.randn(1000) * 0.1
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.REGRESSION,
            metric_types=[
                MetricType.MSE, MetricType.MAE, MetricType.R2,
                MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred)
        
        # Check results
        self.assertIn('mse', result.metrics)
        self.assertIn('mae', result.metrics)
        self.assertIn('r2', result.metrics)
        self.assertIn('explained_variance', result.metrics)
        self.assertIn('max_error', result.metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(result.metrics['mse'], 0.0)
        self.assertGreaterEqual(result.metrics['mae'], 0.0)
        self.assertLessEqual(result.metrics['r2'], 1.0)
        self.assertGreaterEqual(result.metrics['explained_variance'], 0.0)
        self.assertLessEqual(result.metrics['explained_variance'], 1.0)
    
    async def test_binary_classification(self) -> Any:
        """Test binary classification metrics."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create binary classification data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                                 n_informative=10, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 2, 1000)
        y_prob = np.random.rand(1000, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.PR_AUC
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        
        # Check binary-specific metrics
        self.assertIn('roc_auc', result.metrics)
        self.assertIn('pr_auc', result.metrics)
        
        # ROC AUC and PR AUC should be between 0 and 1
        self.assertGreaterEqual(result.metrics['roc_auc'], 0.0)
        self.assertLessEqual(result.metrics['roc_auc'], 1.0)
        self.assertGreaterEqual(result.metrics['pr_auc'], 0.0)
        self.assertLessEqual(result.metrics['pr_auc'], 1.0)
    
    async def test_multi_class_classification(self) -> Any:
        """Test multi-class classification metrics."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create multi-class data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                                 n_informative=10, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 5, 1000)
        y_prob = np.random.rand(1000, 5)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.TOP_K_ACCURACY
            ],
            average='weighted',
            k=3
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        
        # Check multi-class metrics
        self.assertIn('top_k_accuracy', result.metrics)
        self.assertIn('roc_auc', result.metrics)
        
        # Top-k accuracy should be between 0 and 1
        self.assertGreaterEqual(result.metrics['top_k_accuracy'], 0.0)
        self.assertLessEqual(result.metrics['top_k_accuracy'], 1.0)
    
    async def test_custom_metrics(self) -> Any:
        """Test custom metrics."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Define custom metric
        def custom_metric(y_true, y_pred, y_prob) -> Any:
            return np.mean(y_true == y_pred) * 100  # Accuracy as percentage
        
        # Create test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        # Create metric config with custom metric
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[MetricType.CUSTOM],
            custom_metrics={'custom_accuracy': custom_metric}
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y_true, y_pred)
        
        # Check custom metric
        self.assertIn('custom_accuracy', result.metrics)
        self.assertEqual(result.metrics['custom_accuracy'], 60.0)  # 3/5 correct = 60%
    
    async def test_different_averages(self) -> Any:
        """Test different averaging methods."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create multi-class data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                 n_informative=10, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 3, 1000)
        
        # Test different averages
        averages = ['micro', 'macro', 'weighted']
        
        for average in averages:
            config = create_metric_config(
                task_type=TaskType.CLASSIFICATION,
                metric_types=[MetricType.PRECISION, MetricType.RECALL, MetricType.F1],
                average=average
            )
            
            result = evaluator.evaluate(config, y, y_pred)
            
            # Check that metrics are calculated
            self.assertIn('precision', result.metrics)
            self.assertIn('recall', result.metrics)
            self.assertIn('f1', result.metrics)
            
            # Check metric values are reasonable
            self.assertGreaterEqual(result.metrics['precision'], 0.0)
            self.assertLessEqual(result.metrics['precision'], 1.0)
            self.assertGreaterEqual(result.metrics['recall'], 0.0)
            self.assertLessEqual(result.metrics['recall'], 1.0)
            self.assertGreaterEqual(result.metrics['f1'], 0.0)
            self.assertLessEqual(result.metrics['f1'], 1.0)
    
    async def test_error_handling(self) -> Any:
        """Test error handling in metric calculation."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Test with invalid data
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2])  # Different length
        
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[MetricType.ACCURACY]
        )
        
        # Should handle the error gracefully
        with self.assertRaises(ValueError):
            evaluator.evaluate(config, y_true, y_pred)
    
    async def test_plot_confusion_matrix(self) -> Any:
        """Test confusion matrix plotting."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create confusion matrix
        cm = np.array([[50, 10], [5, 35]])
        class_names = ['Class 0', 'Class 1']
        
        # Test plotting
        plot_path = Path(self.temp_dir) / "confusion_matrix.png"
        evaluator.plot_confusion_matrix(cm, class_names, str(plot_path))
        
        # Check if plot was saved
        self.assertTrue(plot_path.exists())
    
    async def test_plot_metrics_comparison(self) -> Any:
        """Test metrics comparison plotting."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create multiple results
        results = []
        for i in range(3):
            result = EvaluationResult(
                task_type=TaskType.CLASSIFICATION,
                metrics={'accuracy': 0.8 + i * 0.05, 'f1': 0.75 + i * 0.05}
            )
            results.append(result)
        
        # Test plotting
        plot_path = Path(self.temp_dir) / "metrics_comparison.png"
        evaluator.plot_metrics_comparison(results, str(plot_path))
        
        # Check if plot was saved
        self.assertTrue(plot_path.exists())

class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_end_to_end_classification(self) -> Any:
        """Test end-to-end classification evaluation."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create comprehensive test data
        X, y = make_classification(n_samples=2000, n_features=50, n_classes=4, 
                                 n_informative=25, n_redundant=10, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Simulate model predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 4, len(y_test))
        y_prob = np.random.rand(len(y_test), 4)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create comprehensive metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.PR_AUC,
                MetricType.CONFUSION_MATRIX, MetricType.CLASSIFICATION_REPORT,
                MetricType.JACCARD, MetricType.HAMMING_LOSS, MetricType.LOG_LOSS,
                MetricType.MATTHEWS_CORR, MetricType.COHEN_KAPPA,
                MetricType.BALANCED_ACCURACY, MetricType.TOP_K_ACCURACY
            ],
            average='weighted',
            k=3
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y_test, y_pred, y_prob)
        
        # Check comprehensive results
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
            'jaccard', 'hamming_loss', 'log_loss', 'matthews_corr',
            'cohen_kappa', 'balanced_accuracy', 'top_k_accuracy'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, result.metrics)
            if result.metrics[metric] is not None:
                self.assertIsInstance(result.metrics[metric], (int, float))
        
        # Check additional results
        self.assertIsNotNone(result.confusion_matrix)
        self.assertIsNotNone(result.classification_report)
        self.assertIn('num_samples', result.metadata)
        self.assertEqual(result.metadata['num_samples'], len(y_test))
    
    async def test_end_to_end_regression(self) -> Any:
        """Test end-to-end regression evaluation."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create comprehensive test data
        X, y = make_regression(n_samples=2000, n_features=50, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Simulate model predictions
        np.random.seed(42)
        y_pred = y_test + np.random.randn(len(y_test)) * 0.1
        
        # Create comprehensive metric config
        config = create_metric_config(
            task_type=TaskType.REGRESSION,
            metric_types=[
                MetricType.MSE, MetricType.MAE, MetricType.R2,
                MetricType.EXPLAINED_VARIANCE, MetricType.MAX_ERROR,
                MetricType.MAPE, MetricType.MSLE, MetricType.MEDIAN_AE,
                MetricType.MEAN_PINBALL_LOSS
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y_test, y_pred)
        
        # Check comprehensive results
        expected_metrics = [
            'mse', 'mae', 'r2', 'explained_variance', 'max_error',
            'mape', 'msle', 'median_ae', 'mean_pinball_loss'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, result.metrics)
            if result.metrics[metric] is not None:
                self.assertIsInstance(result.metrics[metric], (int, float))
        
        # Check metadata
        self.assertIn('num_samples', result.metadata)
        self.assertEqual(result.metadata['num_samples'], len(y_test))
    
    async def test_task_specific_evaluation(self) -> Any:
        """Test task-specific evaluation scenarios."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Test different task types
        task_configs = [
            (TaskType.SENTIMENT_ANALYSIS, 2),  # Binary
            (TaskType.TEXT_CLASSIFICATION, 5),  # Multi-class
            (TaskType.MULTI_LABEL, 3),  # Multi-label
            (TaskType.ANOMALY_DETECTION, 2),  # Binary
        ]
        
        for task_type, n_classes in task_configs:
            # Create appropriate data
            if task_type == TaskType.MULTI_LABEL:
                # Multi-label data
                y_true = np.random.randint(0, 2, (1000, n_classes))
                y_pred = np.random.randint(0, 2, (1000, n_classes))
                y_prob = np.random.rand(1000, n_classes)
            else:
                # Single-label data
                y_true = np.random.randint(0, n_classes, 1000)
                y_pred = np.random.randint(0, n_classes, 1000)
                y_prob = np.random.rand(1000, n_classes)
                y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
            
            # Create metric config
            config = create_metric_config(task_type=task_type)
            
            # Evaluate
            result = evaluator.evaluate(config, y_true, y_pred, y_prob)
            
            # Check that metrics were calculated
            self.assertGreater(len(result.metrics), 0)
            self.assertEqual(result.task_type, task_type)
            self.assertIn('num_samples', result.metadata)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks."""
    
    def setUp(self) -> Any:
        """Set up benchmark environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up benchmark environment."""
        shutil.rmtree(self.temp_dir)
    
    async def benchmark_large_dataset_evaluation(self) -> Any:
        """Benchmark evaluation on large dataset."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create large dataset
        X, y = make_classification(n_samples=100000, n_features=100, n_classes=10, 
                                 n_informative=50, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 10, 100000)
        y_prob = np.random.rand(100000, 10)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX
            ]
        )
        
        # Benchmark
        start_time = time.time()
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        evaluation_time = time.time() - start_time
        
        self.assertLess(evaluation_time, 10.0)  # Should complete within 10 seconds
        logger.info(f"Large dataset evaluation benchmark: {evaluation_time:.4f} seconds")
    
    async def benchmark_multiple_metrics(self) -> Any:
        """Benchmark evaluation with many metrics."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Create test data
        X, y = make_classification(n_samples=10000, n_features=50, n_classes=5, 
                                 n_informative=25, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 5, 10000)
        y_prob = np.random.rand(10000, 5)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create comprehensive metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.PR_AUC,
                MetricType.JACCARD, MetricType.HAMMING_LOSS, MetricType.LOG_LOSS,
                MetricType.MATTHEWS_CORR, MetricType.COHEN_KAPPA,
                MetricType.BALANCED_ACCURACY, MetricType.TOP_K_ACCURACY
            ]
        )
        
        # Benchmark
        start_time = time.time()
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        evaluation_time = time.time() - start_time
        
        self.assertLess(evaluation_time, 5.0)  # Should complete within 5 seconds
        logger.info(f"Multiple metrics evaluation benchmark: {evaluation_time:.4f} seconds")

class TestErrorHandling(unittest.TestCase):
    """Test error handling."""
    
    def setUp(self) -> Any:
        """Set up test environment."""
        self.device_manager = DeviceManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> Any:
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_invalid_inputs(self) -> Any:
        """Test handling of invalid inputs."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Test with None inputs
        config = create_metric_config(task_type=TaskType.CLASSIFICATION)
        
        with self.assertRaises(ValueError):
            evaluator.evaluate(config, None, np.array([0, 1]))
        
        with self.assertRaises(ValueError):
            evaluator.evaluate(config, np.array([0, 1]), None)
        
        # Test with empty arrays
        with self.assertRaises(ValueError):
            evaluator.evaluate(config, np.array([]), np.array([]))
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            evaluator.evaluate(config, np.array([0, 1, 2]), np.array([0, 1]))
    
    async def test_unsupported_metrics(self) -> Any:
        """Test handling of unsupported metrics."""
        evaluator = await create_evaluation_metrics(self.device_manager)
        
        # Test with unsupported metric type
        config = MetricConfig(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[MetricType.CUSTOM]  # Custom without custom function
        )
        
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        # Should handle gracefully
        result = evaluator.evaluate(config, y_true, y_pred)
        self.assertEqual(len(result.metrics), 0)  # No metrics calculated

# Test runner functions
def run_performance_tests():
    """Run performance benchmarks."""
    print("🚀 Running Performance Benchmarks...")
    
    benchmark_suite = unittest.TestSuite()
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_large_dataset_evaluation'))
    benchmark_suite.addTest(TestPerformanceBenchmarks('benchmark_multiple_metrics'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(benchmark_suite)

def run_all_tests():
    """Run all tests."""
    print("🧪 Running All Evaluation Metrics Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMetricConfig,
        TestEvaluationResult,
        TestEvaluationMetrics,
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
async def quick_classification_test():
    """Quick test for classification evaluation."""
    print("🧪 Quick Classification Evaluation Test...")
    
    try:
        # Create evaluator
        device_manager = DeviceManager()
        evaluator = await create_evaluation_metrics(device_manager)
        
        # Create test data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                                 n_informative=10, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = np.random.randint(0, 3, 1000)
        y_prob = np.random.rand(1000, 3)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.CLASSIFICATION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred, y_prob)
        
        print(f"✅ Classification evaluation test passed")
        print(f"Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"F1 Score: {result.metrics['f1']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Classification evaluation test failed: {e}")
        return False

async def quick_regression_test():
    """Quick test for regression evaluation."""
    print("🧪 Quick Regression Evaluation Test...")
    
    try:
        # Create evaluator
        device_manager = DeviceManager()
        evaluator = await create_evaluation_metrics(device_manager)
        
        # Create test data
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        
        # Simulate predictions
        np.random.seed(42)
        y_pred = y + np.random.randn(1000) * 0.1
        
        # Create metric config
        config = create_metric_config(
            task_type=TaskType.REGRESSION,
            metric_types=[
                MetricType.MSE, MetricType.MAE, MetricType.R2,
                MetricType.EXPLAINED_VARIANCE
            ]
        )
        
        # Evaluate
        result = evaluator.evaluate(config, y, y_pred)
        
        print(f"✅ Regression evaluation test passed")
        print(f"MSE: {result.metrics['mse']:.4f}")
        print(f"R²: {result.metrics['r2']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Regression evaluation test failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    
    async def main():
        
    """main function."""
print("🚀 Evaluation Metrics Test Suite")
        print("=" * 50)
        
        # Run quick tests
        print("\n1. Quick Tests:")
        classification_success = await quick_classification_test()
        regression_success = await quick_regression_test()
        
        # Run performance tests
        print("\n2. Performance Tests:")
        run_performance_tests()
        
        # Run comprehensive tests
        print("\n3. Comprehensive Tests:")
        all_tests_success = run_all_tests()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"Classification Test: {'✅ PASSED' if classification_success else '❌ FAILED'}")
        print(f"Regression Test: {'✅ PASSED' if regression_success else '❌ FAILED'}")
        print(f"All Tests: {'✅ PASSED' if all_tests_success else '❌ FAILED'}")
        
        if classification_success and regression_success and all_tests_success:
            print("\n🎉 All tests passed! The Evaluation Metrics system is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
    
    asyncio.run(main()) 