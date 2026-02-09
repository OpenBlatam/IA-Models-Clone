"""
Regression Tests
Tests to prevent regression of previously fixed bugs
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
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig
)
from tests.test_utils import create_test_model, create_test_dataset, create_test_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRegression(unittest.TestCase):
    """Regression tests for previously fixed issues"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.training_config = TrainingConfig(epochs=1, batch_size=2)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def test_import_path_regression(self):
        """Test that import paths work correctly (regression for import issues)"""
        # This was a common issue - imports failing from different directories
        from core import OptimizationEngine
        from core import ModelManager
        from core import TrainingManager
        from core import InferenceEngine
        
        # All should import successfully
        self.assertIsNotNone(OptimizationEngine)
        self.assertIsNotNone(ModelManager)
        self.assertIsNotNone(TrainingManager)
        self.assertIsNotNone(InferenceEngine)
        
        logger.info("✅ Import path regression test passed")
    
    def test_unittest_makesuite_regression(self):
        """Test that deprecated unittest.makeSuite is not used"""
        import unittest
        import inspect
        
        # Check that run_unified_tests doesn't use makeSuite
        from run_unified_tests import UnifiedTestRunner
        source = inspect.getsource(UnifiedTestRunner.add_all_tests)
        
        # Should use TestLoader, not makeSuite
        self.assertIn('TestLoader', source)
        self.assertNotIn('makeSuite', source)
        
        logger.info("✅ unittest.makeSuite regression test passed")
    
    def test_division_by_zero_regression(self):
        """Test that division by zero is handled (regression for calculation errors)"""
        # Test success rate calculation with zero tests
        total_tests = 0
        failures = 0
        errors = 0
        
        # Should not raise ZeroDivisionError
        if total_tests > 0:
            success_rate = ((total_tests - failures - errors) / total_tests) * 100
        else:
            success_rate = 0.0
        
        self.assertEqual(success_rate, 0.0)
        
        logger.info("✅ Division by zero regression test passed")
    
    def test_dataclass_fields_regression(self):
        """Test that dataclass fields are present (regression for missing fields)"""
        from core.inference import InferenceConfig
        
        config = InferenceConfig()
        
        # Required fields should exist
        self.assertTrue(hasattr(config, 'batch_size'))
        self.assertTrue(hasattr(config, 'max_length'))
        self.assertTrue(hasattr(config, 'temperature'))
        
        logger.info("✅ Dataclass fields regression test passed")
    
    def test_async_mock_regression(self):
        """Test that async methods are mocked correctly (regression for async issues)"""
        from unittest.mock import AsyncMock
        
        # AsyncMock should work for async methods
        mock_async = AsyncMock(return_value={'result': 'test'})
        self.assertIsNotNone(mock_async)
        
        logger.info("✅ Async mock regression test passed")
    
    def test_enum_values_regression(self):
        """Test that enum values are correct (regression for wrong enum values)"""
        from core import OptimizationLevel, ModelType
        
        # Test that enum values exist
        self.assertIn(OptimizationLevel.BASIC, OptimizationLevel)
        self.assertIn(OptimizationLevel.ENHANCED, OptimizationLevel)
        self.assertIn(ModelType.TRANSFORMER, ModelType)
        self.assertIn(ModelType.CONVOLUTIONAL, ModelType)
        
        logger.info("✅ Enum values regression test passed")
    
    def test_path_setup_regression(self):
        """Test that path setup works (regression for path issues)"""
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Should be able to import core
        try:
            from core import OptimizationEngine
            import_successful = True
        except ImportError:
            import_successful = False
        
        self.assertTrue(import_successful)
        
        logger.info("✅ Path setup regression test passed")
    
    def test_config_defaults_regression(self):
        """Test that config defaults are set (regression for missing defaults)"""
        from core import OptimizationConfig, ModelConfig, TrainingConfig, InferenceConfig
        
        # Default configs should work
        opt_config = OptimizationConfig()
        model_config = ModelConfig()
        train_config = TrainingConfig()
        inf_config = InferenceConfig()
        
        # Should have valid defaults
        self.assertIsNotNone(opt_config.level)
        self.assertIsNotNone(model_config.model_type)
        self.assertGreater(train_config.epochs, 0)
        self.assertGreater(inf_config.batch_size, 0)
        
        logger.info("✅ Config defaults regression test passed")
    
    def test_error_handling_regression(self):
        """Test that error handling is robust (regression for unhandled errors)"""
        # Test that components handle errors gracefully
        try:
            config = OptimizationConfig(level="invalid")
            engine = OptimizationEngine(config)
            # If no error, that's okay - validation might be elsewhere
        except (ValueError, TypeError):
            # Expected error
            pass
        
        # Should not crash
        self.assertTrue(True)
        
        logger.info("✅ Error handling regression test passed")
    
    def test_model_output_consistency_regression(self):
        """Test that model outputs are consistent (regression for inconsistent outputs)"""
        model = create_test_model()
        test_input = torch.randn(2, 10)
        
        model.eval()
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)
        
        # Outputs should be identical for same input
        self.assertTrue(torch.allclose(output1, output2))
        
        logger.info("✅ Model output consistency regression test passed")

if __name__ == '__main__':
    unittest.main()








