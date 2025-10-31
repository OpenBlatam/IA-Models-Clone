"""
Inference Engine Tests
Comprehensive tests for the unified inference system
"""

import unittest
import torch
import torch.nn as nn
import logging
from core import InferenceEngine, InferenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple test model for inference"""
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        return self.linear(embedded)

class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        
    def encode(self, text, return_tensors="pt"):
        # Simple mock encoding
        tokens = [ord(c) % self.vocab_size for c in text[:10]]  # Limit to 10 tokens
        return torch.tensor([tokens], dtype=torch.long)
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Simple mock decoding
        return ''.join([chr(token_id % 256) for token_id in token_ids if token_id < 256])

class TestInferenceEngine(unittest.TestCase):
    """Test inference engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = InferenceConfig(
            batch_size=1,
            max_length=20,
            temperature=0.8,
            top_p=0.9,
            top_k=50
        )
        
        self.model = SimpleModel(vocab_size=100, hidden_size=64)
        self.tokenizer = MockTokenizer(vocab_size=100)
    
    def test_inference_engine_initialization(self):
        """Test inference engine initialization"""
        engine = InferenceEngine(self.config)
        
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.batch_size, 1)
        self.assertEqual(engine.config.max_length, 20)
        self.assertEqual(engine.config.temperature, 0.8)
        
        logger.info("✅ Inference engine initialization test passed")
    
    def test_model_loading(self):
        """Test model loading"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        self.assertIsNotNone(engine.model)
        self.assertIsNotNone(engine.tokenizer)
        
        logger.info("✅ Model loading test passed")
    
    def test_generation_with_tokens(self):
        """Test generation with token input"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        # Test with token input
        input_tokens = [1, 2, 3, 4, 5]
        result = engine.generate(input_tokens, max_length=10)
        
        self.assertIn('generated_ids', result)
        self.assertIn('generated_text', result)
        self.assertIn('generation_time', result)
        self.assertIn('tokens_generated', result)
        self.assertIn('tokens_per_second', result)
        
        self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        self.assertGreater(result['tokens_generated'], 0)
        
        logger.info("✅ Generation with tokens test passed")
    
    def test_generation_with_text(self):
        """Test generation with text input"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        # Test with text input
        input_text = "Hello world"
        result = engine.generate(input_text, max_length=10)
        
        self.assertIn('generated_ids', result)
        self.assertIn('generated_text', result)
        self.assertIn('generation_time', result)
        self.assertIn('tokens_generated', result)
        
        self.assertIsInstance(result['generated_text'], str)
        self.assertGreater(result['tokens_generated'], 0)
        
        logger.info("✅ Generation with text test passed")
    
    def test_sampling_generation(self):
        """Test sampling generation"""
        config = InferenceConfig(
            batch_size=1,
            max_length=10,
            do_sample=True,
            temperature=1.0,
            top_p=0.9
        )
        
        engine = InferenceEngine(config)
        engine.load_model(self.model, self.tokenizer)
        
        input_tokens = [1, 2, 3]
        result = engine.generate(input_tokens)
        
        self.assertIn('generated_ids', result)
        self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        
        logger.info("✅ Sampling generation test passed")
    
    def test_beam_generation(self):
        """Test beam search generation"""
        config = InferenceConfig(
            batch_size=1,
            max_length=10,
            do_sample=False,
            num_beams=2
        )
        
        engine = InferenceEngine(config)
        engine.load_model(self.model, self.tokenizer)
        
        input_tokens = [1, 2, 3]
        result = engine.generate(input_tokens)
        
        self.assertIn('generated_ids', result)
        self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        
        logger.info("✅ Beam generation test passed")
    
    def test_batch_generation(self):
        """Test batch generation"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        prompts = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        results = engine.batch_generate(prompts, max_length=10)
        
        self.assertEqual(len(results), len(prompts))
        for result in results:
            self.assertIn('generated_ids', result)
            self.assertIn('generated_text', result)
        
        logger.info("✅ Batch generation test passed")
    
    def test_caching_system(self):
        """Test caching system"""
        config = InferenceConfig(
            batch_size=1,
            max_length=10,
            use_cache=True,
            cache_size=100
        )
        
        engine = InferenceEngine(config)
        engine.load_model(self.model, self.tokenizer)
        
        # First generation (cache miss)
        input_tokens = [1, 2, 3]
        result1 = engine.generate(input_tokens)
        
        # Second generation (cache hit)
        result2 = engine.generate(input_tokens)
        
        # Check cache metrics
        metrics = engine.get_performance_metrics()
        self.assertIn('cache_hit_rate', metrics)
        self.assertIn('cache_size', metrics)
        
        logger.info("✅ Caching system test passed")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        # Generate some text
        input_tokens = [1, 2, 3]
        engine.generate(input_tokens, max_length=10)
        
        # Get metrics
        metrics = engine.get_performance_metrics()
        
        self.assertIn('total_inferences', metrics)
        self.assertIn('total_tokens_generated', metrics)
        self.assertIn('average_generation_time', metrics)
        self.assertIn('tokens_per_second', metrics)
        self.assertIn('cache_hit_rate', metrics)
        
        self.assertGreater(metrics['total_inferences'], 0)
        self.assertGreater(metrics['total_tokens_generated'], 0)
        
        logger.info("✅ Performance metrics test passed")
    
    def test_cache_clearing(self):
        """Test cache clearing"""
        config = InferenceConfig(
            batch_size=1,
            max_length=10,
            use_cache=True
        )
        
        engine = InferenceEngine(config)
        engine.load_model(self.model, self.tokenizer)
        
        # Generate some text to populate cache
        input_tokens = [1, 2, 3]
        engine.generate(input_tokens)
        
        # Clear cache
        engine.clear_cache()
        
        # Check cache is empty
        metrics = engine.get_performance_metrics()
        self.assertEqual(metrics['cache_size'], 0)
        
        logger.info("✅ Cache clearing test passed")
    
    def test_inference_optimization(self):
        """Test inference optimization"""
        engine = InferenceEngine(self.config)
        engine.load_model(self.model, self.tokenizer)
        
        # Optimize for inference
        engine.optimize_for_inference()
        
        # Model should still work
        input_tokens = [1, 2, 3]
        result = engine.generate(input_tokens, max_length=10)
        
        self.assertIn('generated_ids', result)
        
        logger.info("✅ Inference optimization test passed")
    
    def test_different_temperature_settings(self):
        """Test different temperature settings"""
        temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for temp in temperatures:
            config = InferenceConfig(
                batch_size=1,
                max_length=10,
                temperature=temp
            )
            
            engine = InferenceEngine(config)
            engine.load_model(self.model, self.tokenizer)
            
            input_tokens = [1, 2, 3]
            result = engine.generate(input_tokens)
            
            self.assertIn('generated_ids', result)
            self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        
        logger.info("✅ Different temperature settings test passed")
    
    def test_different_top_p_settings(self):
        """Test different top-p settings"""
        top_p_values = [0.1, 0.5, 0.9, 0.95, 1.0]
        
        for top_p in top_p_values:
            config = InferenceConfig(
                batch_size=1,
                max_length=10,
                top_p=top_p
            )
            
            engine = InferenceEngine(config)
            engine.load_model(self.model, self.tokenizer)
            
            input_tokens = [1, 2, 3]
            result = engine.generate(input_tokens)
            
            self.assertIn('generated_ids', result)
            self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        
        logger.info("✅ Different top-p settings test passed")
    
    def test_different_top_k_settings(self):
        """Test different top-k settings"""
        top_k_values = [1, 5, 10, 20, 50]
        
        for top_k in top_k_values:
            config = InferenceConfig(
                batch_size=1,
                max_length=10,
                top_k=top_k
            )
            
            engine = InferenceEngine(config)
            engine.load_model(self.model, self.tokenizer)
            
            input_tokens = [1, 2, 3]
            result = engine.generate(input_tokens)
            
            self.assertIn('generated_ids', result)
            self.assertGreater(len(result['generated_ids'][0]), len(input_tokens))
        
        logger.info("✅ Different top-k settings test passed")
    
    def test_max_length_handling(self):
        """Test max length handling"""
        max_lengths = [5, 10, 20, 50]
        
        for max_len in max_lengths:
            config = InferenceConfig(
                batch_size=1,
                max_length=max_len
            )
            
            engine = InferenceEngine(config)
            engine.load_model(self.model, self.tokenizer)
            
            input_tokens = [1, 2, 3]
            result = engine.generate(input_tokens, max_length=max_len)
            
            self.assertIn('generated_ids', result)
            # Generated length should not exceed max_length
            self.assertLessEqual(len(result['generated_ids'][0]), max_len + len(input_tokens))
        
        logger.info("✅ Max length handling test passed")
    
    def test_early_stopping(self):
        """Test early stopping"""
        config = InferenceConfig(
            batch_size=1,
            max_length=20,
            early_stopping=True
        )
        
        engine = InferenceEngine(config)
        engine.load_model(self.model, self.tokenizer)
        
        input_tokens = [1, 2, 3]
        result = engine.generate(input_tokens)
        
        self.assertIn('generated_ids', result)
        
        logger.info("✅ Early stopping test passed")

if __name__ == '__main__':
    unittest.main()

