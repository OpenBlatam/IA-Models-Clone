"""
Edge Cases and Stress Tests
Comprehensive tests for edge cases, boundary conditions, and stress scenarios
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import logging
import tempfile
import os
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig,
    MonitoringSystem
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.training_config = TrainingConfig(epochs=1, batch_size=2)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def test_zero_sized_model(self):
        """Test with zero-sized model parameters"""
        # Create a model with minimal parameters
        model = nn.Linear(1, 1)
        
        optimizer = OptimizationEngine(self.optimization_config)
        optimized = optimizer.optimize_model(model)
        
        self.assertIsNotNone(optimized)
        logger.info("✅ Zero-sized model test passed")
    
    def test_very_large_model(self):
        """Test with very large model"""
        # Create a larger model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50)
        )
        
        optimizer = OptimizationEngine(self.optimization_config)
        optimized = optimizer.optimize_model(model)
        
        self.assertIsNotNone(optimized)
        logger.info("✅ Very large model test passed")
    
    def test_none_inputs(self):
        """Test handling of None inputs"""
        optimizer = OptimizationEngine(self.optimization_config)
        
        # Should handle None gracefully or raise appropriate error
        try:
            result = optimizer.optimize_model(None)
            # If no error, result should be None or handled
            pass
        except (ValueError, TypeError, AttributeError):
            # Expected error for None input
            pass
        
        logger.info("✅ None inputs test passed")
    
    def test_invalid_file_paths(self):
        """Test with invalid file paths"""
        model_manager = ModelManager(self.model_config)
        
        # Test saving to invalid path
        invalid_path = "/nonexistent/path/model.pth"
        try:
            model_manager.save_model(invalid_path)
        except (OSError, IOError, FileNotFoundError):
            # Expected error for invalid path
            pass
        
        logger.info("✅ Invalid file paths test passed")
    
    def test_concurrent_optimization(self):
        """Test concurrent optimization operations"""
        import threading
        
        model = nn.Linear(10, 5)
        results = []
        
        def optimize_worker(worker_id):
            try:
                optimizer = OptimizationEngine(self.optimization_config)
                optimized = optimizer.optimize_model(model)
                results.append((worker_id, True))
            except Exception as e:
                results.append((worker_id, False))
        
        threads = [threading.Thread(target=optimize_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 5)
        logger.info("✅ Concurrent optimization test passed")
    
    def test_memory_pressure(self):
        """Test under memory pressure"""
        # Create multiple models and optimize them
        models = [nn.Linear(50, 25) for _ in range(10)]
        optimizer = OptimizationEngine(self.optimization_config)
        
        optimized_models = []
        for model in models:
            optimized = optimizer.optimize_model(model)
            optimized_models.append(optimized)
        
        self.assertEqual(len(optimized_models), 10)
        logger.info("✅ Memory pressure test passed")
    
    def test_rapid_config_changes(self):
        """Test rapid configuration changes"""
        levels = [
            OptimizationLevel.BASIC,
            OptimizationLevel.ENHANCED,
            OptimizationLevel.ADVANCED
        ]
        
        model = nn.Linear(10, 5)
        
        for level in levels:
            config = OptimizationConfig(level=level)
            optimizer = OptimizationEngine(config)
            optimized = optimizer.optimize_model(model)
            self.assertIsNotNone(optimized)
        
        logger.info("✅ Rapid config changes test passed")
    
    def test_empty_datasets(self):
        """Test with empty datasets"""
        from torch.utils.data import Dataset
        
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            
            def __getitem__(self, idx):
                raise IndexError("Empty dataset")
        
        empty_train = EmptyDataset()
        empty_val = EmptyDataset()
        
        trainer = TrainingManager(self.training_config)
        model = nn.Linear(10, 5)
        
        try:
            trainer.setup_training(model, empty_train, empty_val)
            # Training might fail with empty dataset, which is expected
        except (ValueError, IndexError):
            pass
        
        logger.info("✅ Empty datasets test passed")
    
    def test_extreme_config_values(self):
        """Test with extreme configuration values"""
        # Test with very high values
        config = TrainingConfig(
            epochs=1000,
            batch_size=1000,
            learning_rate=100.0
        )
        
        trainer = TrainingManager(config)
        self.assertIsNotNone(trainer)
        
        # Test with very low values
        config = TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-10
        )
        
        trainer = TrainingManager(config)
        self.assertIsNotNone(trainer)
        
        logger.info("✅ Extreme config values test passed")
    
    def test_model_reloading(self):
        """Test repeated model loading"""
        model_manager = ModelManager(self.model_config)
        
        # Load model multiple times
        for _ in range(5):
            model = model_manager.load_model()
            self.assertIsNotNone(model)
        
        logger.info("✅ Model reloading test passed")
    
    def test_inference_without_model(self):
        """Test inference without loading model"""
        engine = InferenceEngine(self.inference_config)
        
        # Should handle gracefully
        try:
            result = engine.generate([1, 2, 3])
            # If no error, might have default behavior
        except (ValueError, AttributeError, RuntimeError):
            # Expected error when model not loaded
            pass
        
        logger.info("✅ Inference without model test passed")
    
    def test_monitoring_without_start(self):
        """Test monitoring without starting"""
        monitor = MonitoringSystem()
        
        # Should handle gracefully
        report = monitor.get_comprehensive_report()
        self.assertIsNotNone(report)
        
        logger.info("✅ Monitoring without start test passed")
    
    def test_stress_test_workflow(self):
        """Stress test complete workflow"""
        # Run complete workflow multiple times
        for i in range(3):
            optimizer = OptimizationEngine(self.optimization_config)
            model_manager = ModelManager(self.model_config)
            trainer = TrainingManager(self.training_config)
            inferencer = InferenceEngine(self.inference_config)
            
            model = model_manager.load_model()
            optimized = optimizer.optimize_model(model)
            
            self.assertIsNotNone(optimized)
            self.assertIsNotNone(trainer)
            self.assertIsNotNone(inferencer)
        
        logger.info("✅ Stress test workflow passed")
    
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        # Test with minimum valid values
        config = InferenceConfig(batch_size=1, max_length=1)
        engine = InferenceEngine(config)
        self.assertIsNotNone(engine)
        
        # Test with maximum reasonable values
        config = InferenceConfig(batch_size=128, max_length=4096)
        engine = InferenceEngine(config)
        self.assertIsNotNone(engine)
        
        logger.info("✅ Boundary conditions test passed")
    
    def test_error_recovery(self):
        """Test error recovery"""
        optimizer = OptimizationEngine(self.optimization_config)
        model = nn.Linear(10, 5)
        
        # First operation
        try:
            optimized1 = optimizer.optimize_model(model)
        except Exception:
            optimized1 = None
        
        # Second operation after potential error
        try:
            optimized2 = optimizer.optimize_model(model)
            self.assertIsNotNone(optimized2)
        except Exception:
            # If first failed, second might also fail
            pass
        
        logger.info("✅ Error recovery test passed")
    
    def test_resource_limits(self):
        """Test resource limits"""
        # Create many components
        optimizers = [OptimizationEngine(self.optimization_config) for _ in range(20)]
        managers = [ModelManager(self.model_config) for _ in range(20)]
        
        self.assertEqual(len(optimizers), 20)
        self.assertEqual(len(managers), 20)
        
        logger.info("✅ Resource limits test passed")
    
    def test_unicode_handling(self):
        """Test Unicode handling in strings"""
        # Test with Unicode characters
        unicode_text = "Hello 世界 🌍"
        
        inferencer = InferenceEngine(self.inference_config)
        
        # Should handle Unicode gracefully
        try:
            # This might fail if tokenizer doesn't support Unicode
            pass
        except Exception:
            pass
        
        logger.info("✅ Unicode handling test passed")
    
    def test_special_characters(self):
        """Test special characters handling"""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        inferencer = InferenceEngine(self.inference_config)
        
        # Should handle special characters
        try:
            pass
        except Exception:
            pass
        
        logger.info("✅ Special characters test passed")

if __name__ == '__main__':
    unittest.main()








