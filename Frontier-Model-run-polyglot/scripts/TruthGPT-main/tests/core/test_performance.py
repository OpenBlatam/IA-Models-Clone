"""
Performance Tests
Comprehensive performance and benchmark tests
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
import time
from core import (
    OptimizationEngine, OptimizationConfig, OptimizationLevel,
    ModelManager, ModelConfig, ModelType,
    TrainingManager, TrainingConfig,
    InferenceEngine, InferenceConfig
)
from tests.test_utils import (
    create_test_model, create_test_dataset, create_test_tokenizer,
    TestTimer, assert_performance_acceptable, assert_model_valid,
    get_model_parameters_count
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
        self.model_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            vocab_size=1000
        )
        self.training_config = TrainingConfig(epochs=1, batch_size=8)
        self.inference_config = InferenceConfig(batch_size=4, max_length=50)
    
    def test_optimization_performance(self):
        """Test optimization performance"""
        model = create_test_model(input_size=100, output_size=50, hidden_size=128)
        
        with TestTimer() as timer:
            optimizer = OptimizationEngine(self.optimization_config)
            optimized_model = optimizer.optimize_model(model)
        
        # Performance should be acceptable
        assert_performance_acceptable(timer.get_duration(), max_duration=2.0)
        assert_model_valid(optimized_model)
        
        logger.info(f"✅ Optimization performance: {timer.get_duration():.3f}s")
    
    def test_model_loading_performance(self):
        """Test model loading performance"""
        model_manager = ModelManager(self.model_config)
        
        with TestTimer() as timer:
            model = model_manager.load_model()
        
        assert_performance_acceptable(timer.get_duration(), max_duration=1.0)
        assert_model_valid(model)
        
        logger.info(f"✅ Model loading performance: {timer.get_duration():.3f}s")
    
    def test_training_performance(self):
        """Test training performance"""
        train_dataset = create_test_dataset(size=100, input_size=100, output_size=50)
        val_dataset = create_test_dataset(size=20, input_size=100, output_size=50)
        
        trainer = TrainingManager(self.training_config)
        model = create_test_model(input_size=100, output_size=50)
        trainer.setup_training(model, train_dataset, val_dataset)
        
        with TestTimer() as timer:
            results = trainer.train()
        
        assert_performance_acceptable(timer.get_duration(), max_duration=10.0)
        self.assertIsNotNone(results)
        
        logger.info(f"✅ Training performance: {timer.get_duration():.3f}s")
    
    def test_inference_performance(self):
        """Test inference performance"""
        model = create_test_model(input_size=100, output_size=50)
        tokenizer = create_test_tokenizer(vocab_size=1000)
        
        engine = InferenceEngine(self.inference_config)
        engine.load_model(model, tokenizer)
        
        # Single inference
        with TestTimer() as timer:
            result = engine.generate([1, 2, 3, 4, 5], max_length=20)
        
        assert_performance_acceptable(timer.get_duration(), max_duration=1.0)
        self.assertIn('generated_ids', result)
        
        logger.info(f"✅ Single inference performance: {timer.get_duration():.3f}s")
        
        # Batch inference
        prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        with TestTimer() as timer:
            results = engine.generate(prompts, max_length=20)
        
        assert_performance_acceptable(timer.get_duration(), max_duration=2.0)
        self.assertIn('generated_ids', results)
        
        logger.info(f"✅ Batch inference performance: {timer.get_duration():.3f}s")
    
    def test_throughput_benchmark(self):
        """Test inference throughput"""
        model = create_test_model(input_size=100, output_size=50)
        tokenizer = create_test_tokenizer(vocab_size=1000)
        
        engine = InferenceEngine(self.inference_config)
        engine.load_model(model, tokenizer)
        
        num_requests = 10
        start_time = time.time()
        
        for _ in range(num_requests):
            engine.generate([1, 2, 3], max_length=10)
        
        total_time = time.time() - start_time
        throughput = num_requests / total_time
        
        self.assertGreater(throughput, 1.0)  # At least 1 request per second
        logger.info(f"✅ Throughput: {throughput:.2f} requests/second")
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and optimize multiple models
            models = [create_test_model() for _ in range(5)]
            optimizer = OptimizationEngine(self.optimization_config)
            
            optimized_models = []
            for model in models:
                optimized = optimizer.optimize_model(model)
                optimized_models.append(optimized)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 500)  # Less than 500MB increase
            
            logger.info(f"✅ Memory efficiency: {memory_increase:.2f}MB increase")
        except ImportError:
            logger.warning("psutil not available, skipping memory test")
    
    def test_optimization_levels_performance(self):
        """Test performance across different optimization levels"""
        model = create_test_model()
        levels = [
            OptimizationLevel.BASIC,
            OptimizationLevel.ENHANCED,
            OptimizationLevel.ADVANCED
        ]
        
        results = {}
        for level in levels:
            config = OptimizationConfig(level=level)
            optimizer = OptimizationEngine(config)
            
            with TestTimer() as timer:
                optimized = optimizer.optimize_model(model)
            
            results[level.value] = timer.get_duration()
            assert_model_valid(optimized)
        
        # All levels should complete in reasonable time
        for level, duration in results.items():
            assert_performance_acceptable(duration, max_duration=3.0)
            logger.info(f"  {level}: {duration:.3f}s")
        
        logger.info("✅ Optimization levels performance test passed")
    
    def test_concurrent_performance(self):
        """Test concurrent operation performance"""
        import threading
        
        model = create_test_model()
        optimizer = OptimizationEngine(self.optimization_config)
        
        results = []
        
        def optimize_worker(worker_id):
            start = time.time()
            optimized = optimizer.optimize_model(model)
            duration = time.time() - start
            results.append((worker_id, duration, optimized is not None))
        
        threads = [threading.Thread(target=optimize_worker, args=(i,)) for i in range(5)]
        
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start_time
        
        # All should complete successfully
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r[2] for r in results))  # All optimized successfully
        
        # Total time should be reasonable
        assert_performance_acceptable(total_time, max_duration=5.0)
        
        logger.info(f"✅ Concurrent performance: {total_time:.3f}s for 5 operations")
    
    def test_model_size_impact(self):
        """Test performance with different model sizes"""
        sizes = [
            (10, 5, 20),   # Small
            (50, 25, 100), # Medium
            (100, 50, 200) # Large
        ]
        
        results = {}
        for input_size, output_size, hidden_size in sizes:
            model = create_test_model(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
            param_count = get_model_parameters_count(model)
            
            optimizer = OptimizationEngine(self.optimization_config)
            
            with TestTimer() as timer:
                optimized = optimizer.optimize_model(model)
            
            results[param_count] = timer.get_duration()
            assert_model_valid(optimized)
        
        # Performance should scale reasonably
        for param_count, duration in results.items():
            logger.info(f"  Model with {param_count} params: {duration:.3f}s")
        
        logger.info("✅ Model size impact test passed")
    
    def test_cache_performance_impact(self):
        """Test cache performance impact"""
        model = create_test_model()
        tokenizer = create_test_tokenizer()
        
        # Without cache
        config_no_cache = InferenceConfig(batch_size=1, use_cache=False)
        engine_no_cache = InferenceEngine(config_no_cache)
        engine_no_cache.load_model(model, tokenizer)
        
        start = time.time()
        for _ in range(5):
            engine_no_cache.generate([1, 2, 3], max_length=5)
        time_no_cache = time.time() - start
        
        # With cache
        config_cache = InferenceConfig(batch_size=1, use_cache=True, cache_size=10)
        engine_cache = InferenceEngine(config_cache)
        engine_cache.load_model(model, tokenizer)
        
        start = time.time()
        for _ in range(5):
            engine_cache.generate([1, 2, 3], max_length=5)
        time_with_cache = time.time() - start
        
        # Cache should help (or at least not hurt significantly)
        logger.info(f"✅ Cache performance: {time_no_cache:.3f}s without, {time_with_cache:.3f}s with")
        self.assertIsNotNone(engine_cache.cache_hits)

if __name__ == '__main__':
    unittest.main()








