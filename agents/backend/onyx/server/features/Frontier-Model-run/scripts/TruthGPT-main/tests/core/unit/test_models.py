"""
Model Management Tests
Comprehensive tests for the unified model management system
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
from core import ModelManager, ModelConfig, ModelType
from core.architectures import TransformerModel, ConvolutionalModel, RecurrentModel, HybridModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestModelManager(unittest.TestCase):
    """Test model management functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.transformer_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_name="test_transformer",
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            vocab_size=1000
        )
        
        self.conv_config = ModelConfig(
            model_type=ModelType.CONVOLUTIONAL,
            model_name="test_conv",
            hidden_size=64,
            vocab_size=100
        )
        
        self.rnn_config = ModelConfig(
            model_type=ModelType.RECURRENT,
            model_name="test_rnn",
            hidden_size=64,
            num_layers=2,
            vocab_size=1000
        )
        
        self.hybrid_config = ModelConfig(
            model_type=ModelType.HYBRID,
            model_name="test_hybrid",
            hidden_size=128,
            num_layers=4,
            vocab_size=1000
        )
    
    def test_transformer_model_creation(self):
        """Test transformer model creation"""
        manager = ModelManager(self.transformer_config)
        model = manager.load_model()
        
        self.assertIsInstance(model, TransformerModel)
        self.assertEqual(model.hidden_size, 128)
        self.assertEqual(model.num_layers, 4)
        self.assertEqual(model.num_heads, 4)
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids)
        self.assertEqual(output.shape, (2, 10, 1000))
        
        logger.info("✅ Transformer model creation test passed")
    
    def test_convolutional_model_creation(self):
        """Test convolutional model creation"""
        manager = ModelManager(self.conv_config)
        model = manager.load_model()
        
        self.assertIsInstance(model, ConvolutionalModel)
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 32, 32)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 100))
        
        logger.info("✅ Convolutional model creation test passed")
    
    def test_recurrent_model_creation(self):
        """Test recurrent model creation"""
        manager = ModelManager(self.rnn_config)
        model = manager.load_model()
        
        self.assertIsInstance(model, RecurrentModel)
        self.assertEqual(model.hidden_size, 64)
        self.assertEqual(model.num_layers, 2)
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids)
        self.assertEqual(output.shape, (2, 10, 1000))
        
        logger.info("✅ Recurrent model creation test passed")
    
    def test_hybrid_model_creation(self):
        """Test hybrid model creation"""
        manager = ModelManager(self.hybrid_config)
        model = manager.load_model()
        
        self.assertIsInstance(model, HybridModel)
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        output = model(input_ids)
        self.assertEqual(output.shape, (2, 10, 1000))
        
        logger.info("✅ Hybrid model creation test passed")
    
    def test_model_info(self):
        """Test model information retrieval"""
        manager = ModelManager(self.transformer_config)
        model = manager.load_model()
        
        info = manager.get_model_info()
        
        self.assertIn("model_type", info)
        self.assertIn("total_parameters", info)
        self.assertIn("trainable_parameters", info)
        self.assertIn("device", info)
        self.assertIn("dtype", info)
        
        self.assertEqual(info["model_type"], "transformer")
        self.assertGreater(info["total_parameters"], 0)
        
        logger.info("✅ Model info test passed")
    
    def test_model_saving_and_loading(self):
        """Test model saving and loading"""
        manager = ModelManager(self.transformer_config)
        model = manager.load_model()
        
        # Save model
        save_path = "test_model.pth"
        manager.save_model(save_path)
        
        # Create new manager and load model
        new_manager = ModelManager(self.transformer_config)
        loaded_model = new_manager.load_model(save_path)
        
        # Compare model parameters
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(param1, param2))
        
        # Clean up
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
        
        logger.info("✅ Model saving and loading test passed")
    
    def test_custom_model_registration(self):
        """Test custom model registration"""
        manager = ModelManager(self.transformer_config)
        
        # Define custom model
        class CustomModel(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                self.linear = nn.Linear(input_size, output_size)
            
            def forward(self, x):
                return self.linear(x)
        
        # Register custom model
        manager.register_model("custom", CustomModel, {"input_size": 100, "output_size": 50})
        
        # Create custom model
        custom_model = manager.create_custom_model("custom")
        
        self.assertIsInstance(custom_model, CustomModel)
        
        # Test forward pass
        input_tensor = torch.randn(2, 100)
        output = custom_model(input_tensor)
        self.assertEqual(output.shape, (2, 50))
        
        logger.info("✅ Custom model registration test passed")
    
    def test_model_device_handling(self):
        """Test model device handling"""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            device="cpu",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            vocab_size=100
        )
        
        manager = ModelManager(config)
        model = manager.load_model()
        
        # Check device
        device = next(model.parameters()).device
        self.assertEqual(str(device), "cpu")
        
        logger.info("✅ Model device handling test passed")
    
    def test_model_precision_handling(self):
        """Test model precision handling"""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            precision="float16",
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            vocab_size=100
        )
        
        manager = ModelManager(config)
        model = manager.load_model()
        
        # Check precision
        dtype = next(model.parameters()).dtype
        self.assertEqual(dtype, torch.float16)
        
        logger.info("✅ Model precision handling test passed")
    
    def test_model_architecture_validation(self):
        """Test model architecture validation"""
        # Test invalid model type
        with self.assertRaises(ValueError):
            config = ModelConfig(model_type="invalid_type")
            manager = ModelManager(config)
            manager.load_model()
        
        logger.info("✅ Model architecture validation test passed")
    
    def test_model_parameter_counting(self):
        """Test model parameter counting"""
        model = self.model_manager.load_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertGreaterEqual(trainable_params, 0)
        self.assertLessEqual(trainable_params, total_params)
        
        logger.info(f"✅ Model parameter counting: {total_params} total, {trainable_params} trainable")
    
    def test_model_gradient_flow(self):
        """Test model gradient flow"""
        model = self.model_manager.load_model()
        
        # Test forward and backward pass
        test_input = torch.randn(2, 1000)
        output = model(test_input)
        
        # Compute loss and backward
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_gradients)
        
        logger.info("✅ Model gradient flow test passed")
    
    def test_model_device_transfer(self):
        """Test model device transfer"""
        model = self.model_manager.load_model()
        
        # Test CPU
        model_cpu = model.cpu()
        self.assertEqual(next(model_cpu.parameters()).device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            self.assertEqual(next(model_cuda.parameters()).device.type, 'cuda')
        
        logger.info("✅ Model device transfer test passed")
    
    def test_model_eval_mode(self):
        """Test model evaluation mode"""
        model = self.model_manager.load_model()
        
        # Set to eval mode
        model.eval()
        self.assertFalse(model.training)
        
        # Set back to train mode
        model.train()
        self.assertTrue(model.training)
        
        logger.info("✅ Model eval mode test passed")
    
    def test_model_state_dict_operations(self):
        """Test model state dict operations"""
        model1 = self.model_manager.load_model()
        model2 = self.model_manager.load_model()
        
        # Get state dicts
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()
        
        # State dicts should have same keys
        self.assertEqual(set(state_dict1.keys()), set(state_dict2.keys()))
        
        # Load state dict
        model2.load_state_dict(state_dict1)
        
        logger.info("✅ Model state dict operations test passed")
    
    def test_model_info_completeness(self):
        """Test model info completeness"""
        model = self.model_manager.load_model()
        info = self.model_manager.get_model_info()
        
        self.assertIsNotNone(info)
        self.assertIn('model_type', info)
        self.assertIn('parameters', info)
        
        logger.info("✅ Model info completeness test passed")
    
    def test_model_save_load_consistency(self):
        """Test model save/load consistency"""
        import tempfile
        import os
        
        model1 = self.model_manager.load_model()
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            save_path = f.name
        
        try:
            self.model_manager.save_model(save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Load model
            model2 = self.model_manager.load_model(save_path)
            self.assertIsNotNone(model2)
            
            # Compare parameter counts
            params1 = sum(p.numel() for p in model1.parameters())
            params2 = sum(p.numel() for p in model2.parameters())
            self.assertEqual(params1, params2)
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
        
        logger.info("✅ Model save/load consistency test passed")
    
    def test_model_forward_pass_consistency(self):
        """Test model forward pass consistency"""
        model = self.model_manager.load_model()
        test_input = torch.randn(2, 1000)
        
        # Multiple forward passes should be consistent
        output1 = model(test_input)
        output2 = model(test_input)
        
        # Outputs should be identical (same input, same model state)
        self.assertTrue(torch.allclose(output1, output2))
        
        logger.info("✅ Model forward pass consistency test passed")

if __name__ == '__main__':
    unittest.main()

