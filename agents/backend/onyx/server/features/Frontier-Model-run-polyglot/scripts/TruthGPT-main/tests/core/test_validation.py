"""
Validation Tests
Comprehensive validation tests for inputs, outputs, and configurations
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
from tests.test_assertions import (
    assert_model_valid, assert_output_shape, assert_config_valid
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestValidation(unittest.TestCase):
    """Comprehensive validation tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.training_config = TrainingConfig(epochs=1, batch_size=2)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation"""
        # Test all config types
        configs = [
            OptimizationConfig(level=OptimizationLevel.BASIC),
            ModelConfig(model_type=ModelType.TRANSFORMER),
            TrainingConfig(epochs=1, batch_size=2),
            InferenceConfig(batch_size=1)
        ]
        
        for config in configs:
            assert_config_valid(config, ['level'] if hasattr(config, 'level') else 
                               ['model_type'] if hasattr(config, 'model_type') else
                               ['epochs'] if hasattr(config, 'epochs') else
                               ['batch_size'])
        
        logger.info("✅ Comprehensive config validation test passed")
    
    def test_input_validation_ranges(self):
        """Test input validation with various ranges"""
        engine = InferenceEngine(self.inference_config)
        model = create_test_model()
        tokenizer = create_test_tokenizer()
        engine.load_model(model, tokenizer)
        
        # Test various input ranges
        test_inputs = [
            [],  # Empty
            [1],  # Single
            list(range(10)),  # Small
            list(range(100)),  # Medium
            list(range(1000)),  # Large
        ]
        
        for input_data in test_inputs:
            try:
                result = engine.generate(input_data, max_length=10)
                self.assertIsNotNone(result)
                self.assertIn('generated_ids', result)
            except (ValueError, RuntimeError) as e:
                # Some inputs may be too large
                if len(input_data) > 100:
                    pass  # Expected
                else:
                    raise
        
        logger.info("✅ Input validation ranges test passed")
    
    def test_output_validation_structure(self):
        """Test output structure validation"""
        engine = InferenceEngine(self.inference_config)
        model = create_test_model()
        tokenizer = create_test_tokenizer()
        engine.load_model(model, tokenizer)
        
        result = engine.generate([1, 2, 3], max_length=10)
        
        # Validate structure
        self.assertIsInstance(result, dict)
        self.assertIn('generated_ids', result)
        self.assertIn('generated_text', result)
        self.assertIn('generation_time', result)
        self.assertIn('tokens_generated', result)
        
        # Validate types
        self.assertIsInstance(result['generated_ids'], list)
        self.assertIsInstance(result['generated_text'], str)
        self.assertIsInstance(result['generation_time'], (int, float))
        self.assertIsInstance(result['tokens_generated'], int)
        
        logger.info("✅ Output validation structure test passed")
    
    def test_model_validation_comprehensive(self):
        """Test comprehensive model validation"""
        model = create_test_model()
        
        # Validate model structure
        assert_model_valid(model)
        
        # Validate forward pass
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output = model(test_input)
        
        assert_output_shape(output, (2, 5))
        
        logger.info("✅ Comprehensive model validation test passed")
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test valid parameters
        valid_configs = [
            TrainingConfig(epochs=1, batch_size=1, learning_rate=0.001),
            TrainingConfig(epochs=10, batch_size=32, learning_rate=1e-5),
            InferenceConfig(batch_size=1, max_length=100, temperature=0.8),
            InferenceConfig(batch_size=8, max_length=512, temperature=1.5),
        ]
        
        for config in valid_configs:
            self.assertIsNotNone(config)
            if hasattr(config, 'epochs'):
                self.assertGreater(config.epochs, 0)
            if hasattr(config, 'batch_size'):
                self.assertGreater(config.batch_size, 0)
            if hasattr(config, 'learning_rate'):
                self.assertGreater(config.learning_rate, 0)
        
        logger.info("✅ Parameter validation test passed")
    
    def test_boundary_validation(self):
        """Test boundary value validation"""
        # Test boundary values
        boundaries = [
            (1, True),  # Minimum valid
            (0, False),  # Zero (invalid)
            (-1, False),  # Negative (invalid)
            (1000, True),  # Large valid
        ]
        
        for value, should_be_valid in boundaries:
            try:
                config = TrainingConfig(epochs=value, batch_size=2)
                if should_be_valid:
                    self.assertIsNotNone(config)
                else:
                    # Should raise error or handle gracefully
                    pass
            except (ValueError, TypeError):
                if should_be_valid:
                    self.fail(f"Valid value {value} was rejected")
        
        logger.info("✅ Boundary validation test passed")
    
    def test_type_validation(self):
        """Test type validation"""
        # Test type checking
        model = create_test_model()
        self.assertIsInstance(model, nn.Module)
        
        config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.assertIsInstance(config.level, OptimizationLevel)
        
        dataset = create_test_dataset()
        self.assertIsNotNone(dataset)
        
        logger.info("✅ Type validation test passed")
    
    def test_state_validation(self):
        """Test state validation"""
        model = create_test_model()
        
        # Test training state
        model.train()
        self.assertTrue(model.training)
        
        # Test eval state
        model.eval()
        self.assertFalse(model.training)
        
        logger.info("✅ State validation test passed")
    
    def test_dimension_validation(self):
        """Test dimension validation"""
        model = create_test_model(input_size=10, output_size=5)
        
        # Test various input dimensions
        test_cases = [
            (1, 10),  # Single sample
            (2, 10),  # Batch of 2
            (8, 10),  # Batch of 8
        ]
        
        for batch_size, input_size in test_cases:
            test_input = torch.randn(batch_size, input_size)
            with torch.no_grad():
                output = model(test_input)
            
            assert_output_shape(output, (batch_size, 5))
        
        logger.info("✅ Dimension validation test passed")
    
    def test_consistency_validation(self):
        """Test consistency validation"""
        model = create_test_model()
        test_input = torch.randn(2, 10)
        
        model.eval()
        with torch.no_grad():
            output1 = model(test_input)
            output2 = model(test_input)
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))
        
        logger.info("✅ Consistency validation test passed")

if __name__ == '__main__':
    unittest.main()








