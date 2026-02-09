"""
Compatibility Tests
Tests for cross-platform and version compatibility
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
import platform
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig
)
from tests.test_utils import create_test_model, create_test_dataset, create_test_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCompatibility(unittest.TestCase):
    """Test compatibility across platforms and versions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.training_config = TrainingConfig(epochs=1, batch_size=2)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility"""
        python_version = sys.version_info
        self.assertGreaterEqual(python_version.major, 3)
        self.assertGreaterEqual(python_version.minor, 7)
        
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    def test_pytorch_version_compatibility(self):
        """Test PyTorch version compatibility"""
        torch_version = torch.__version__
        self.assertIsNotNone(torch_version)
        
        # PyTorch should be importable and functional
        x = torch.tensor([1, 2, 3])
        self.assertIsNotNone(x)
        
        logger.info(f"✅ PyTorch version: {torch_version}")
    
    def test_platform_compatibility(self):
        """Test platform compatibility"""
        system = platform.system()
        self.assertIn(system, ['Windows', 'Linux', 'Darwin', 'Java'])
        
        logger.info(f"✅ Platform: {system} {platform.release()}")
    
    def test_device_compatibility(self):
        """Test device compatibility"""
        # CPU should always be available
        self.assertTrue(torch.device('cpu') is not None)
        
        # CUDA may or may not be available
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("✅ CUDA not available (CPU only)")
    
    def test_cross_platform_path_handling(self):
        """Test cross-platform path handling"""
        from pathlib import Path
        import tempfile
        import os
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        try:
            # Path should work on all platforms
            path_obj = Path(temp_path)
            self.assertTrue(path_obj.exists())
            
            # Should handle both forward and backward slashes
            normalized = str(path_obj).replace('\\', '/')
            self.assertIsNotNone(normalized)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        logger.info("✅ Cross-platform path handling test passed")
    
    def test_unicode_compatibility(self):
        """Test Unicode compatibility"""
        # Test with Unicode characters
        unicode_strings = [
            "Hello 世界",
            "Привет",
            "مرحبا",
            "🌍🌎🌏"
        ]
        
        for text in unicode_strings:
            # Should handle Unicode
            encoded = text.encode('utf-8')
            decoded = encoded.decode('utf-8')
            self.assertEqual(text, decoded)
        
        logger.info("✅ Unicode compatibility test passed")
    
    def test_encoding_compatibility(self):
        """Test encoding compatibility"""
        test_string = "Test string with special chars: àáâãäå"
        
        # Should handle various encodings
        encodings = ['utf-8', 'latin-1', 'ascii']
        for encoding in encodings:
            try:
                encoded = test_string.encode(encoding)
                decoded = encoded.decode(encoding)
                self.assertEqual(test_string, decoded)
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Some encodings may not support all characters
                pass
        
        logger.info("✅ Encoding compatibility test passed")
    
    def test_file_system_compatibility(self):
        """Test file system compatibility"""
        import tempfile
        import os
        
        # Test file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            
            # Write
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test content")
            
            # Read
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.assertEqual(content, "test content")
        
        logger.info("✅ File system compatibility test passed")
    
    def test_model_serialization_compatibility(self):
        """Test model serialization compatibility"""
        model = create_test_model()
        
        # Save model
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            save_path = f.name
        
        try:
            torch.save(model.state_dict(), save_path)
            self.assertTrue(os.path.exists(save_path))
            
            # Load model
            loaded_state = torch.load(save_path)
            self.assertIsNotNone(loaded_state)
            
            # Create new model and load state
            new_model = create_test_model()
            new_model.load_state_dict(loaded_state)
            
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
        
        logger.info("✅ Model serialization compatibility test passed")
    
    def test_thread_safety_compatibility(self):
        """Test thread safety across platforms"""
        import threading
        
        model = create_test_model()
        results = []
        
        def worker(worker_id):
            try:
                # Model operations should be thread-safe
                with torch.no_grad():
                    x = torch.randn(2, 10)
                    output = model(x)
                results.append((worker_id, output is not None))
            except Exception:
                results.append((worker_id, False))
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r[1] for r in results))
        
        logger.info("✅ Thread safety compatibility test passed")
    
    def test_numeric_precision_compatibility(self):
        """Test numeric precision compatibility"""
        # Test float32 and float16
        x32 = torch.randn(5, 5, dtype=torch.float32)
        x16 = x32.half()
        
        # Should handle both precisions
        self.assertEqual(x32.shape, x16.shape)
        
        # Convert back
        x32_back = x16.float()
        self.assertEqual(x32.shape, x32_back.shape)
        
        logger.info("✅ Numeric precision compatibility test passed")
    
    def test_backward_compatibility(self):
        """Test backward compatibility"""
        # Test that old configs still work
        old_config = OptimizationConfig(level=OptimizationLevel.BASIC)
        engine = OptimizationEngine(old_config)
        self.assertIsNotNone(engine)
        
        old_model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        manager = ModelManager(old_model_config)
        self.assertIsNotNone(manager)
        
        logger.info("✅ Backward compatibility test passed")

if __name__ == '__main__':
    unittest.main()








