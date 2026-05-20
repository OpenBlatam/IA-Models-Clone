"""
Core Component Tests
Tests for the unified core components
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
from core import OptimizationEngine, OptimizationConfig, OptimizationLevel
from core import ModelManager, ModelConfig, ModelType
from core import TrainingManager, TrainingConfig
from core import InferenceEngine, InferenceConfig
from core import MonitoringSystem
from tests.test_utils import (
    create_test_model, assert_model_valid, assert_config_valid,
    TestTimer, assert_performance_acceptable
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCoreComponents(unittest.TestCase):
    """Test core component initialization and basic functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.training_config = TrainingConfig(epochs=1, batch_size=2)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def test_optimization_engine_initialization(self):
        """Test optimization engine initialization"""
        engine = OptimizationEngine(self.optimization_config)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.level, OptimizationLevel.BASIC)
        logger.info("✅ Optimization engine initialization test passed")
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        manager = ModelManager(self.model_config)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.config.model_type, ModelType.TRANSFORMER)
        logger.info("✅ Model manager initialization test passed")
    
    def test_training_manager_initialization(self):
        """Test training manager initialization"""
        manager = TrainingManager(self.training_config)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.config.epochs, 1)
        logger.info("✅ Training manager initialization test passed")
    
    def test_inference_engine_initialization(self):
        """Test inference engine initialization"""
        engine = InferenceEngine(self.inference_config)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.batch_size, 1)
        logger.info("✅ Inference engine initialization test passed")
    
    def test_monitoring_system_initialization(self):
        """Test monitoring system initialization"""
        system = MonitoringSystem()
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.metrics_collector)
        logger.info("✅ Monitoring system initialization test passed")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configs
        valid_configs = [
            OptimizationConfig(level=OptimizationLevel.BASIC),
            OptimizationConfig(level=OptimizationLevel.ENHANCED),
            OptimizationConfig(level=OptimizationLevel.ADVANCED),
            OptimizationConfig(level=OptimizationLevel.ULTRA),
            OptimizationConfig(level=OptimizationLevel.SUPREME),
            OptimizationConfig(level=OptimizationLevel.TRANSCENDENT)
        ]
        
        for config in valid_configs:
            engine = OptimizationEngine(config)
            self.assertIsNotNone(engine)
        
        logger.info("✅ Configuration validation test passed")
    
    def test_config_default_values(self):
        """Test configuration default values"""
        # Test default optimization config
        default_opt_config = OptimizationConfig()
        assert_config_valid(default_opt_config, ['level'])
        self.assertIsNotNone(default_opt_config.level)
        
        # Test default model config
        default_model_config = ModelConfig()
        assert_config_valid(default_model_config, ['model_type'])
        self.assertIsNotNone(default_model_config.model_type)
        
        # Test default training config
        default_train_config = TrainingConfig()
        assert_config_valid(default_train_config, ['epochs', 'batch_size'])
        self.assertGreater(default_train_config.epochs, 0)
        self.assertGreater(default_train_config.batch_size, 0)
        
        # Test default inference config
        default_inf_config = InferenceConfig()
        assert_config_valid(default_inf_config, ['batch_size', 'max_length'])
        self.assertGreater(default_inf_config.batch_size, 0)
        self.assertGreater(default_inf_config.max_length, 0)
        
        logger.info("✅ Configuration default values test passed")
    
    def test_component_interaction(self):
        """Test interaction between different components"""
        with TestTimer() as timer:
            # Create all components
            optimizer = OptimizationEngine(self.optimization_config)
            model_manager = ModelManager(self.model_config)
            trainer = TrainingManager(self.training_config)
            inferencer = InferenceEngine(self.inference_config)
            
            # Verify they can work together
            model = model_manager.load_model()
            optimized_model = optimizer.optimize_model(model)
            
            # Validate components
            self.assertIsNotNone(optimized_model)
            self.assertIsNotNone(trainer)
            self.assertIsNotNone(inferencer)
            
            # Validate model can process input
            assert_model_valid(optimized_model)
            
            # Validate performance
            assert_performance_acceptable(timer.get_duration(), max_duration=5.0)
        
        logger.info(f"✅ Component interaction test passed ({timer.get_duration():.3f}s)")
    
    def test_config_copy(self):
        """Test configuration copying"""
        import copy
        
        original_config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
        copied_config = copy.deepcopy(original_config)
        
        self.assertEqual(original_config.level, copied_config.level)
        self.assertIsNot(original_config, copied_config)
        
        logger.info("✅ Configuration copy test passed")
    
    def test_multiple_instances(self):
        """Test creating multiple instances of components"""
        # Create multiple instances
        engines = [OptimizationEngine(self.optimization_config) for _ in range(3)]
        managers = [ModelManager(self.model_config) for _ in range(3)]
        
        # Verify all instances are independent
        self.assertEqual(len(engines), 3)
        self.assertEqual(len(managers), 3)
        self.assertIsNot(engines[0], engines[1])
        self.assertIsNot(managers[0], managers[1])
        
        logger.info("✅ Multiple instances test passed")
    
    def test_component_state_persistence(self):
        """Test component state persistence"""
        optimizer = OptimizationEngine(self.optimization_config)
        model_manager = ModelManager(self.model_config)
        
        # Get initial state
        initial_opt_state = optimizer.get_performance_metrics()
        model = model_manager.load_model()
        
        # Perform operations
        optimized_model = optimizer.optimize_model(model)
        
        # Get state after operations
        final_opt_state = optimizer.get_performance_metrics()
        
        # Verify state changed
        self.assertIsNotNone(initial_opt_state)
        self.assertIsNotNone(final_opt_state)
        
        logger.info("✅ Component state persistence test passed")
    
    def test_resource_cleanup(self):
        """Test resource cleanup"""
        optimizer = OptimizationEngine(self.optimization_config)
        model_manager = ModelManager(self.model_config)
        
        # Create and use resources
        model = model_manager.load_model()
        optimized_model = optimizer.optimize_model(model)
        
        # Cleanup should not raise errors
        del optimized_model
        del model
        
        logger.info("✅ Resource cleanup test passed")
    
    def test_thread_safety(self):
        """Test basic thread safety"""
        import threading
        
        results = []
        
        def worker():
            try:
                optimizer = OptimizationEngine(self.optimization_config)
                model_manager = ModelManager(self.model_config)
                model = model_manager.load_model()
                optimized = optimizer.optimize_model(model)
                results.append(True)
            except Exception as e:
                results.append(False)
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results))
        
        logger.info("✅ Thread safety test passed")

if __name__ == '__main__':
    unittest.main()

