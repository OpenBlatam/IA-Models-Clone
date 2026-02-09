"""
Core Component Tests
Tests for the unified core components
"""

import unittest
import torch
import torch.nn as nn
import logging
from core import OptimizationEngine, OptimizationConfig, OptimizationLevel
from core import ModelManager, ModelConfig, ModelType
from core import TrainingManager, TrainingConfig
from core import InferenceEngine, InferenceConfig
from core import MonitoringSystem

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

if __name__ == '__main__':
    unittest.main()

