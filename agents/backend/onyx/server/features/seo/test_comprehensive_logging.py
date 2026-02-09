#!/usr/bin/env python3
"""
Test script for Comprehensive Logging System
Tests the enhanced logging capabilities for training progress and errors
"""

import unittest
import sys
import os
import tempfile
import json
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_llm_seo_engine import (
    AdvancedLLMSEOEngine,
    SEODataset,
    DataLoaderManager,
    DataLoaderConfig,
    EarlyStopping
)

class TestComprehensiveLogging(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()
        self.config.debug_mode = True
        self.config.batch_size = 32
        self.config.learning_rate = 1e-4
        self.config.weight_decay = 1e-5
        self.config.num_epochs = 10
        self.config.use_mixed_precision = False
        self.config.max_grad_norm = 1.0
        self.config.early_stopping_patience = 5
        self.config.early_stopping_min_delta = 1e-4
        self.config.early_stopping_monitor = "val_loss"
        self.config.early_stopping_mode = "min"
        self.config.lr_scheduler = "cosine"
        self.config.warmup_steps = 100
        self.config.lr_scheduler_params = {
            "cosine": {},
            "linear": {}
        }
        
        # Mock device
        self.device = torch.device("cpu")
        
        # Create mock model and optimizer
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        self.mock_optimizer.param_groups = [{'lr': 1e-4}]
        self.mock_scheduler = Mock()
        self.mock_scaler = Mock()
        
        # Create sample data
        self.texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
        self.labels = [0, 1, 0]
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clean up log files
        if os.path.exists("logs"):
            shutil.rmtree("logs", ignore_errors=True)
    
    def test_logging_setup(self):
        """Test that the comprehensive logging system is properly set up."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Check that specialized loggers are created
                        self.assertIsNotNone(logging.getLogger("training_progress"))
                        self.assertIsNotNone(logging.getLogger("model_performance"))
                        self.assertIsNotNone(logging.getLogger("data_loading"))
                        self.assertIsNotNone(logging.getLogger("error_tracker"))
                        
                        # Check that logs directory is created
                        self.assertTrue(os.path.exists("logs"))
    
    def test_log_training_progress(self):
        """Test training progress logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test basic training progress logging
                        engine.log_training_progress(
                            epoch=1,
                            step=100,
                            loss=0.5,
                            learning_rate=1e-4
                        )
                        
                        # Test with validation loss
                        engine.log_training_progress(
                            epoch=1,
                            step=100,
                            loss=0.5,
                            learning_rate=1e-4,
                            validation_loss=0.6
                        )
                        
                        # Test with additional metrics
                        engine.log_training_progress(
                            epoch=1,
                            step=100,
                            loss=0.5,
                            learning_rate=1e-4,
                            validation_loss=0.6,
                            metrics={"accuracy": 0.85, "f1_score": 0.82}
                        )
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("training_progress" in f for f in log_files))
                        self.assertTrue(any("training_detailed" in f for f in log_files))
    
    def test_log_model_performance(self):
        """Test model performance logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test basic performance logging
                        engine.log_model_performance(
                            operation="forward_pass",
                            duration=0.1
                        )
                        
                        # Test with memory usage
                        engine.log_model_performance(
                            operation="forward_pass",
                            duration=0.1,
                            memory_usage=512.5
                        )
                        
                        # Test with GPU utilization
                        engine.log_model_performance(
                            operation="forward_pass",
                            duration=0.1,
                            memory_usage=512.5,
                            gpu_utilization=75.2
                        )
                        
                        # Test with additional metrics
                        engine.log_model_performance(
                            operation="forward_pass",
                            duration=0.1,
                            memory_usage=512.5,
                            gpu_utilization=75.2,
                            additional_metrics={"batch_size": 32, "sequence_length": 512}
                        )
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("model_performance" in f for f in log_files))
    
    def test_log_data_loading(self):
        """Test data loading logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test basic data loading logging
                        engine.log_data_loading(
                            operation="dataset_creation",
                            dataset_size=1000,
                            batch_size=32,
                            duration=0.5
                        )
                        
                        # Test with memory usage
                        engine.log_data_loading(
                            operation="dataset_creation",
                            dataset_size=1000,
                            batch_size=32,
                            duration=0.5,
                            memory_usage=256.0
                        )
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("data_loading" in f for f in log_files))
    
    def test_log_error(self):
        """Test error logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test basic error logging
                        test_error = ValueError("Test error message")
                        engine.log_error(
                            error=test_error,
                            context="Test context",
                            operation="test_operation"
                        )
                        
                        # Test with additional info
                        engine.log_error(
                            error=test_error,
                            context="Test context",
                            operation="test_operation",
                            additional_info={"param1": "value1", "param2": 42}
                        )
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("errors" in f for f in log_files))
                        self.assertTrue(any("error_tracking" in f for f in log_files))
    
    def test_log_training_summary(self):
        """Test training summary logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test training summary logging
                        engine.log_training_summary(
                            total_epochs=10,
                            total_steps=1000,
                            final_loss=0.1,
                            best_loss=0.05,
                            training_duration=3600.0,
                            early_stopping_triggered=False
                        )
                        
                        # Test with early stopping
                        engine.log_training_summary(
                            total_epochs=5,
                            total_steps=500,
                            final_loss=0.2,
                            best_loss=0.1,
                            training_duration=1800.0,
                            early_stopping_triggered=True
                        )
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("training_progress" in f for f in log_files))
    
    def test_log_hyperparameters(self):
        """Test hyperparameters logging functionality."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test hyperparameters logging
                        test_config = {
                            "learning_rate": 1e-4,
                            "batch_size": 32,
                            "num_epochs": 100,
                            "use_mixed_precision": True,
                            "max_grad_norm": 1.0
                        }
                        
                        engine.log_hyperparameters(test_config)
                        
                        # Verify logs were created
                        log_files = os.listdir("logs")
                        self.assertTrue(any("training_progress" in f for f in log_files))
    
    def test_log_file_rotation(self):
        """Test that log files are properly rotated."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Generate many log entries to trigger rotation
                        for i in range(1000):
                            engine.log_training_progress(
                                epoch=i // 100,
                                step=i,
                                loss=0.1 + (i % 10) * 0.01,
                                learning_rate=1e-4
                            )
                        
                        # Verify log files exist
                        log_files = os.listdir("logs")
                        self.assertTrue(len(log_files) > 0)
                        
                        # Check for different log types
                        log_types = ["training_progress", "model_performance", "data_loading", "errors"]
                        for log_type in log_types:
                            self.assertTrue(any(log_type in f for f in log_files))
    
    def test_logging_performance(self):
        """Test that logging doesn't significantly impact performance."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Measure time without logging
                        start_time = time.time()
                        for i in range(100):
                            pass  # Do nothing
                        baseline_time = time.time() - start_time
                        
                        # Measure time with logging
                        start_time = time.time()
                        for i in range(100):
                            engine.log_training_progress(
                                epoch=1,
                                step=i,
                                loss=0.1,
                                learning_rate=1e-4
                            )
                        logging_time = time.time() - start_time
                        
                        # Logging should not take more than 10x the baseline
                        self.assertLess(logging_time, baseline_time * 10)
    
    def test_logging_error_recovery(self):
        """Test that logging system recovers from errors gracefully."""
        with patch('torch.device', return_value=self.device):
            with patch('advanced_llm_seo_engine.CustomSEOModel', return_value=self.mock_model):
                with patch('advanced_llm_seo_engine.AdvancedTokenizer', return_value=Mock()):
                    with patch('advanced_llm_seo_engine.pipeline', return_value=Mock()):
                        engine = AdvancedLLMSEOEngine(self.config)
                        
                        # Test logging with invalid data
                        try:
                            engine.log_training_progress(
                                epoch="invalid",
                                step="invalid",
                                loss="invalid",
                                learning_rate="invalid"
                            )
                            # Should not raise exception
                            self.assertTrue(True)
                        except Exception as e:
                            self.fail(f"Logging should handle invalid data gracefully: {e}")
                        
                        # Test logging with None values
                        try:
                            engine.log_model_performance(
                                operation=None,
                                duration=None,
                                memory_usage=None
                            )
                            # Should not raise exception
                            self.assertTrue(True)
                        except Exception as e:
                            self.fail(f"Logging should handle None values gracefully: {e}")

def run_comprehensive_tests():
    """Run all comprehensive logging tests."""
    print("🧪 Running Comprehensive Logging Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveLogging)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
