"""
Benchmark Tests
Comprehensive benchmark tests for performance comparison
"""

import unittest
import sys
from pathlib import Path
import time

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
    InferenceEngine, InferenceConfig
)
from tests.test_utils import create_test_model, create_test_tokenizer, TestTimer
from tests.test_fixtures import get_fixture

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestBenchmarks(unittest.TestCase):
    """Benchmark tests for performance comparison"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimization_config = OptimizationConfig(level=OptimizationLevel.ENHANCED)
        self.model_config = ModelConfig(model_type=ModelType.TRANSFORMER)
        self.inference_config = InferenceConfig(batch_size=1)
    
    def benchmark_model_creation(self):
        """Benchmark model creation"""
        times = []
        for _ in range(10):
            with TestTimer() as timer:
                model = create_test_model()
            times.append(timer.get_duration())
        
        avg_time = sum(times) / len(times)
        logger.info(f"📊 Model creation: {avg_time*1000:.2f}ms average")
        return avg_time
    
    def benchmark_optimization(self):
        """Benchmark optimization"""
        model = create_test_model()
        optimizer = OptimizationEngine(self.optimization_config)
        
        times = []
        for _ in range(5):
            with TestTimer() as timer:
                optimized = optimizer.optimize_model(model)
            times.append(timer.get_duration())
        
        avg_time = sum(times) / len(times)
        logger.info(f"📊 Optimization: {avg_time*1000:.2f}ms average")
        return avg_time
    
    def benchmark_inference(self):
        """Benchmark inference"""
        model = create_test_model()
        tokenizer = create_test_tokenizer()
        engine = InferenceEngine(self.inference_config)
        engine.load_model(model, tokenizer)
        
        times = []
        for _ in range(20):
            with TestTimer() as timer:
                result = engine.generate([1, 2, 3], max_length=10)
            times.append(timer.get_duration())
        
        avg_time = sum(times) / len(times)
        logger.info(f"📊 Inference: {avg_time*1000:.2f}ms average")
        return avg_time
    
    def benchmark_batch_inference(self):
        """Benchmark batch inference"""
        model = create_test_model()
        tokenizer = create_test_tokenizer()
        config = InferenceConfig(batch_size=8)
        engine = InferenceEngine(config)
        engine.load_model(model, tokenizer)
        
        prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                   [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
        
        times = []
        for _ in range(5):
            with TestTimer() as timer:
                result = engine.generate(prompts, max_length=10)
            times.append(timer.get_duration())
        
        avg_time = sum(times) / len(times)
        logger.info(f"📊 Batch inference (8): {avg_time*1000:.2f}ms average")
        return avg_time
    
    def benchmark_optimization_levels(self):
        """Benchmark different optimization levels"""
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
            logger.info(f"📊 {level.value}: {timer.get_duration()*1000:.2f}ms")
        
        return results
    
    def benchmark_model_sizes(self):
        """Benchmark different model sizes"""
        sizes = [
            (10, 5),
            (50, 25),
            (100, 50),
            (200, 100)
        ]
        
        results = {}
        for input_size, output_size in sizes:
            model = create_test_model(input_size=input_size, output_size=output_size)
            optimizer = OptimizationEngine(self.optimization_config)
            
            with TestTimer() as timer:
                optimized = optimizer.optimize_model(model)
            
            param_count = sum(p.numel() for p in model.parameters())
            results[param_count] = timer.get_duration()
            logger.info(f"📊 Model ({param_count} params): {timer.get_duration()*1000:.2f}ms")
        
        return results
    
    def test_run_all_benchmarks(self):
        """Run all benchmarks"""
        logger.info("🚀 Running comprehensive benchmarks...")
        
        # Run all benchmarks
        creation_time = self.benchmark_model_creation()
        optimization_time = self.benchmark_optimization()
        inference_time = self.benchmark_inference()
        batch_time = self.benchmark_batch_inference()
        level_results = self.benchmark_optimization_levels()
        size_results = self.benchmark_model_sizes()
        
        # All should complete successfully
        self.assertGreater(creation_time, 0)
        self.assertGreater(optimization_time, 0)
        self.assertGreater(inference_time, 0)
        self.assertGreater(batch_time, 0)
        
        logger.info("✅ All benchmarks completed successfully")
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison"""
        # Compare single vs batch inference
        single_time = self.benchmark_inference()
        batch_time = self.benchmark_batch_inference()
        
        # Batch should be more efficient per item
        efficiency = (single_time * 8) / batch_time if batch_time > 0 else 0
        logger.info(f"📊 Batch efficiency: {efficiency:.2f}x")
        
        self.assertGreater(single_time, 0)
        self.assertGreater(batch_time, 0)

if __name__ == '__main__':
    unittest.main()








