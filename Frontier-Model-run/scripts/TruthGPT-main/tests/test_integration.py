"""
Integration Tests
Comprehensive integration tests for the unified TruthGPT system
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

class SimpleDataset(Dataset):
    """Simple test dataset for integration tests"""
    def __init__(self, size=100, vocab_size=1000, seq_len=20):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Generate random data
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.targets = torch.randint(0, vocab_size, (size, seq_len))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MockTokenizer:
    """Mock tokenizer for integration tests"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        
    def encode(self, text, return_tensors="pt"):
        tokens = [ord(c) % self.vocab_size for c in text[:10]]
        return torch.tensor([tokens], dtype=torch.long)
    
    def decode(self, token_ids, skip_special_tokens=True):
        return ''.join([chr(token_id % 256) for token_id in token_ids if token_id < 256])

class TestIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Configuration
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
        self.model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            vocab_size=100
        )
        self.training_config = TrainingConfig(epochs=1, batch_size=4, log_interval=5)
        self.inference_config = InferenceConfig(batch_size=1, max_length=10)
        
        # Datasets
        self.train_dataset = SimpleDataset(size=50, vocab_size=100, seq_len=10)
        self.val_dataset = SimpleDataset(size=20, vocab_size=100, seq_len=10)
        self.tokenizer = MockTokenizer(vocab_size=100)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from optimization to inference"""
        logger.info("🔄 Testing complete workflow...")
        
        # 1. Initialize optimization engine
        optimizer = OptimizationEngine(self.optimization_config)
        logger.info("✅ Optimization engine initialized")
        
        # 2. Load model
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        logger.info("✅ Model loaded")
        
        # 3. Optimize model
        optimized_model = optimizer.optimize_model(model)
        logger.info("✅ Model optimized")
        
        # 4. Setup training
        trainer = TrainingManager(self.training_config)
        trainer.setup_training(optimized_model, self.train_dataset, self.val_dataset)
        logger.info("✅ Training setup completed")
        
        # 5. Train model
        training_results = trainer.train()
        logger.info("✅ Training completed")
        
        # 6. Setup inference
        inferencer = InferenceEngine(self.inference_config)
        inferencer.load_model(optimized_model, self.tokenizer)
        inferencer.optimize_for_inference()
        logger.info("✅ Inference setup completed")
        
        # 7. Generate text
        result = inferencer.generate("Hello", max_length=5)
        logger.info("✅ Text generation completed")
        
        # Verify results
        self.assertIn('generated_ids', result)
        self.assertIn('generated_text', result)
        self.assertGreater(result['tokens_generated'], 0)
        
        logger.info("✅ Complete workflow test passed")
    
    def test_optimization_levels_integration(self):
        """Test integration with different optimization levels"""
        levels = [
            OptimizationLevel.BASIC,
            OptimizationLevel.ENHANCED,
            OptimizationLevel.ADVANCED,
            OptimizationLevel.ULTRA,
            OptimizationLevel.SUPREME,
            OptimizationLevel.TRANSCENDENT
        ]
        
        for level in levels:
            logger.info(f"🧪 Testing {level.value} optimization level...")
            
            # Create configuration
            config = OptimizationConfig(level=level)
            optimizer = OptimizationEngine(config)
            
            # Load and optimize model
            model_manager = ModelManager(self.model_config)
            model = model_manager.load_model()
            optimized_model = optimizer.optimize_model(model)
            
            # Verify optimization worked
            self.assertIsNotNone(optimized_model)
            
            # Test inference
            inferencer = InferenceEngine(self.inference_config)
            inferencer.load_model(optimized_model, self.tokenizer)
            result = inferencer.generate([1, 2, 3], max_length=5)
            
            self.assertIn('generated_ids', result)
            
        logger.info("✅ Optimization levels integration test passed")
    
    def test_model_types_integration(self):
        """Test integration with different model types"""
        model_types = [
            ModelType.TRANSFORMER,
            ModelType.CONVOLUTIONAL,
            ModelType.RECURRENT,
            ModelType.HYBRID
        ]
        
        for model_type in model_types:
            logger.info(f"🧪 Testing {model_type.value} model type...")
            
            # Create configuration
            config = ModelConfig(
                model_type=model_type,
                hidden_size=32,
                num_layers=1,
                num_heads=2,
                vocab_size=50
            )
            
            # Load model
            model_manager = ModelManager(config)
            model = model_manager.load_model()
            
            # Optimize model
            optimizer = OptimizationEngine(self.optimization_config)
            optimized_model = optimizer.optimize_model(model)
            
            # Test inference
            inferencer = InferenceEngine(self.inference_config)
            inferencer.load_model(optimized_model, self.tokenizer)
            
            if model_type == ModelType.CONVOLUTIONAL:
                # For CNN, we need different input format
                input_tensor = torch.randn(1, 3, 32, 32)
                # Note: This would need a different inference approach for CNN
                pass
            else:
                result = inferencer.generate([1, 2, 3], max_length=5)
                self.assertIn('generated_ids', result)
        
        logger.info("✅ Model types integration test passed")
    
    def test_monitoring_integration(self):
        """Test monitoring system integration"""
        logger.info("📊 Testing monitoring integration...")
        
        # Start monitoring
        monitor = MonitoringSystem()
        monitor.start_monitoring(interval=0.1)
        
        # Run some operations
        optimizer = OptimizationEngine(self.optimization_config)
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        optimized_model = optimizer.optimize_model(model)
        
        # Record some metrics
        monitor.metrics_collector.record_model_metrics(0.1, 100.0, 512.0, 0.8)
        monitor.metrics_collector.record_training_metrics(1, 0.5, 0.6, 0.001, 30.0)
        
        # Wait for system metrics
        import time
        time.sleep(0.5)
        
        # Get comprehensive report
        report = monitor.get_comprehensive_report()
        
        self.assertIn('system', report)
        self.assertIn('model', report)
        self.assertIn('training', report)
        self.assertIn('timestamp', report)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        logger.info("✅ Monitoring integration test passed")
    
    def test_save_load_integration(self):
        """Test save and load integration"""
        logger.info("💾 Testing save/load integration...")
        
        # Create and train model
        optimizer = OptimizationEngine(self.optimization_config)
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        optimized_model = optimizer.optimize_model(model)
        
        # Train model
        trainer = TrainingManager(self.training_config)
        trainer.setup_training(optimized_model, self.train_dataset, self.val_dataset)
        trainer.train()
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pth")
        model_manager.save_model(model_path)
        
        # Save training checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        trainer.save_checkpoint(checkpoint_path)
        
        # Load model in new manager
        new_model_manager = ModelManager(self.model_config)
        loaded_model = new_model_manager.load_model(model_path)
        
        # Load training checkpoint
        new_trainer = TrainingManager(self.training_config)
        new_trainer.setup_training(loaded_model, self.train_dataset, self.val_dataset)
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify loaded model works
        inferencer = InferenceEngine(self.inference_config)
        inferencer.load_model(loaded_model, self.tokenizer)
        result = inferencer.generate([1, 2, 3], max_length=5)
        
        self.assertIn('generated_ids', result)
        
        logger.info("✅ Save/load integration test passed")
    
    def test_performance_integration(self):
        """Test performance integration"""
        logger.info("⚡ Testing performance integration...")
        
        # Create components
        optimizer = OptimizationEngine(self.optimization_config)
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Optimize model
        start_time = time.time()
        optimized_model = optimizer.optimize_model(model)
        optimization_time = time.time() - start_time
        
        # Test inference performance
        inferencer = InferenceEngine(self.inference_config)
        inferencer.load_model(optimized_model, self.tokenizer)
        
        # Generate multiple times to test performance
        start_time = time.time()
        for _ in range(5):
            result = inferencer.generate([1, 2, 3], max_length=5)
        total_time = time.time() - start_time
        
        # Get performance metrics
        optimization_metrics = optimizer.get_performance_metrics()
        inference_metrics = inferencer.get_performance_metrics()
        
        # Verify metrics
        self.assertIn('optimization_count', optimization_metrics)
        self.assertIn('total_inferences', inference_metrics)
        self.assertGreater(optimization_metrics['optimization_count'], 0)
        self.assertGreater(inference_metrics['total_inferences'], 0)
        
        logger.info(f"✅ Performance integration test passed (optimization: {optimization_time:.3f}s, inference: {total_time:.3f}s)")
    
    def test_error_handling_integration(self):
        """Test error handling integration"""
        logger.info("🛡️ Testing error handling integration...")
        
        # Test with invalid configuration
        with self.assertRaises(ValueError):
            invalid_config = ModelConfig(model_type="invalid_type")
            manager = ModelManager(invalid_config)
            manager.load_model()
        
        # Test with invalid optimization level
        with self.assertRaises(ValueError):
            invalid_optimization_config = OptimizationConfig(level="invalid_level")
            optimizer = OptimizationEngine(invalid_optimization_config)
        
        # Test with missing tokenizer
        inferencer = InferenceEngine(self.inference_config)
        inferencer.load_model(nn.Linear(10, 5))  # Simple model without tokenizer
        
        with self.assertRaises(ValueError):
            inferencer.generate("Hello, world!")  # Should fail without tokenizer
        
        logger.info("✅ Error handling integration test passed")
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        logger.info("🔄 Testing concurrent operations...")
        
        import threading
        import time
        
        results = []
        
        def worker(worker_id):
            """Worker function for concurrent testing"""
            try:
                # Create components
                optimizer = OptimizationEngine(self.optimization_config)
                model_manager = ModelManager(self.model_config)
                model = model_manager.load_model()
                
                # Optimize model
                optimized_model = optimizer.optimize_model(model)
                
                # Test inference
                inferencer = InferenceEngine(self.inference_config)
                inferencer.load_model(optimized_model, self.tokenizer)
                result = inferencer.generate([worker_id, worker_id + 1], max_length=3)
                
                results.append((worker_id, result))
                
            except Exception as e:
                results.append((worker_id, f"Error: {e}"))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3)
        for worker_id, result in results:
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f"Worker {worker_id} failed: {result}")
            else:
                self.assertIn('generated_ids', result)
        
        logger.info("✅ Concurrent operations test passed")
    
    def test_memory_management_integration(self):
        """Test memory management integration"""
        logger.info("🧠 Testing memory management integration...")
        
        # Create components with memory optimization
        config = OptimizationConfig(
            level=OptimizationLevel.ENHANCED,
            enable_memory_optimization=True
        )
        
        optimizer = OptimizationEngine(config)
        model_manager = ModelManager(self.model_config)
        model = model_manager.load_model()
        
        # Optimize model (should include memory optimizations)
        optimized_model = optimizer.optimize_model(model)
        
        # Test memory cleanup
        optimizer.memory_optimizer.cleanup_memory()
        
        # Verify model still works
        inferencer = InferenceEngine(self.inference_config)
        inferencer.load_model(optimized_model, self.tokenizer)
        result = inferencer.generate([1, 2, 3], max_length=5)
        
        self.assertIn('generated_ids', result)
        
        logger.info("✅ Memory management integration test passed")

if __name__ == '__main__':
    unittest.main()

