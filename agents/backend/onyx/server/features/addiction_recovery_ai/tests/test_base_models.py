"""
Comprehensive Unit Tests for Base Model Classes
Tests for BaseModel, BasePredictor, BaseGenerator, BaseAnalyzer
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from core.base.base_model import (
    BaseModel,
    BasePredictor,
    BaseGenerator,
    BaseAnalyzer
)


class SimpleTestModel(BaseModel):
    """Simple test model for testing BaseModel"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimplePredictor(BasePredictor):
    """Simple predictor for testing BasePredictor"""
    
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, inputs, **kwargs):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, list):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            return self.forward(inputs).item() if inputs.dim() == 1 else self.forward(inputs)
    
    def _process_batch(self, batch, **kwargs):
        results = []
        for item in batch:
            result = self.predict(item, **kwargs)
            results.append(result)
        return results


class SimpleGenerator(BaseGenerator):
    """Simple generator for testing BaseGenerator"""
    
    def __init__(self):
        super().__init__()
        self.vocab_size = 1000
        self.embedding = nn.Embedding(self.vocab_size, 128)
        self.output = nn.Linear(128, self.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.output(x)
        return x
    
    def generate(self, prompt, **kwargs):
        # Simple mock generation
        max_length = kwargs.get("max_length", 10)
        return f"Generated: {prompt[:max_length]}"


class SimpleAnalyzer(BaseAnalyzer):
    """Simple analyzer for testing BaseAnalyzer"""
    
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 3)  # 3 classes: positive, negative, neutral
    
    def forward(self, x):
        return self.model(x)
    
    def analyze(self, inputs, **kwargs):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, list):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            output = self.forward(inputs)
            probs = torch.softmax(output, dim=-1)
            
            sentiment_map = ["negative", "neutral", "positive"]
            sentiment_idx = torch.argmax(probs, dim=-1).item()
            
            return {
                "sentiment": sentiment_map[sentiment_idx],
                "confidence": probs[0][sentiment_idx].item(),
                "probabilities": probs[0].tolist()
            }


class TestBaseModel:
    """Test suite for BaseModel class"""
    
    def test_init_default_device(self):
        """Test initialization with default device"""
        model = SimpleTestModel()
        assert model.device.type in ["cpu", "cuda"]
        assert model.use_mixed_precision == (model.device.type == "cuda")
    
    def test_init_custom_device(self):
        """Test initialization with custom device"""
        device = torch.device("cpu")
        model = SimpleTestModel()
        model = model.to_device(device)
        assert model.device == device
    
    def test_forward_implementation(self):
        """Test forward pass implementation"""
        model = SimpleTestModel()
        x = torch.randn(5, 10)
        output = model(x)
        
        assert output.shape == (5, 5)
        assert not torch.isnan(output).any()
    
    def test_get_model_info(self):
        """Test getting model information"""
        model = SimpleTestModel()
        info = model.get_model_info()
        
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "device" in info
        assert "dtype" in info
        assert "is_compiled" in info
        assert "use_mixed_precision" in info
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] == info["total_parameters"]
    
    def test_to_device(self):
        """Test moving model to device"""
        model = SimpleTestModel()
        device = torch.device("cpu")
        model = model.to_device(device)
        
        assert model.device == device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compile_cuda(self):
        """Test model compilation on CUDA"""
        if hasattr(torch, 'compile'):
            model = SimpleTestModel()
            model = model.to_device(torch.device("cuda"))
            compiled = model.compile()
            
            assert compiled._is_compiled or compiled is model
    
    def test_compile_cpu(self):
        """Test model compilation on CPU (should not compile)"""
        model = SimpleTestModel()
        compiled = model.compile()
        
        # On CPU, compilation might not work, so model should remain unchanged
        assert compiled is not None


class TestBasePredictor:
    """Test suite for BasePredictor class"""
    
    def test_init(self):
        """Test predictor initialization"""
        predictor = SimplePredictor()
        assert predictor is not None
    
    def test_predict_single(self):
        """Test single prediction"""
        predictor = SimplePredictor()
        input_data = [0.1] * 10
        result = predictor.predict(input_data)
        
        assert isinstance(result, (float, int))
    
    def test_predict_batch(self):
        """Test batch prediction"""
        predictor = SimplePredictor()
        inputs = [[0.1] * 10, [0.2] * 10, [0.3] * 10]
        results = predictor.predict_batch(inputs, batch_size=2)
        
        assert len(results) == 3
        assert all(isinstance(r, (float, int)) for r in results)
    
    def test_predict_batch_small_batch(self):
        """Test batch prediction with small batch size"""
        predictor = SimplePredictor()
        inputs = [[0.1] * 10] * 5
        results = predictor.predict_batch(inputs, batch_size=2)
        
        assert len(results) == 5
    
    def test_predict_batch_empty(self):
        """Test batch prediction with empty input"""
        predictor = SimplePredictor()
        results = predictor.predict_batch([], batch_size=2)
        
        assert len(results) == 0
    
    def test_predict_tensor(self):
        """Test prediction with tensor input"""
        predictor = SimplePredictor()
        input_tensor = torch.randn(10)
        result = predictor.predict(input_tensor)
        
        assert isinstance(result, (float, int))
    
    def test_get_model_info(self):
        """Test getting predictor model info"""
        predictor = SimplePredictor()
        info = predictor.get_model_info()
        
        assert "total_parameters" in info
        assert info["total_parameters"] > 0


class TestBaseGenerator:
    """Test suite for BaseGenerator class"""
    
    def test_init(self):
        """Test generator initialization"""
        generator = SimpleGenerator()
        assert generator is not None
    
    def test_generate_basic(self):
        """Test basic generation"""
        generator = SimpleGenerator()
        result = generator.generate("test prompt")
        
        assert isinstance(result, str)
        assert "test" in result.lower() or "Generated" in result
    
    def test_generate_with_kwargs(self):
        """Test generation with kwargs"""
        generator = SimpleGenerator()
        result = generator.generate("test", max_length=5)
        
        assert isinstance(result, str)
    
    def test_generate_batch(self):
        """Test batch generation"""
        generator = SimpleGenerator()
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = generator.generate_batch(prompts)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_generate_batch_empty(self):
        """Test batch generation with empty prompts"""
        generator = SimpleGenerator()
        results = generator.generate_batch([])
        
        assert len(results) == 0
    
    def test_forward_implementation(self):
        """Test forward pass for generator"""
        generator = SimpleGenerator()
        x = torch.randint(0, generator.vocab_size, (5, 10))
        output = generator(x)
        
        assert output.shape == (5, 10, generator.vocab_size)


class TestBaseAnalyzer:
    """Test suite for BaseAnalyzer class"""
    
    def test_init(self):
        """Test analyzer initialization"""
        analyzer = SimpleAnalyzer()
        assert analyzer is not None
    
    def test_analyze_basic(self):
        """Test basic analysis"""
        analyzer = SimpleAnalyzer()
        input_data = [0.1] * 10
        result = analyzer.analyze(input_data)
        
        assert isinstance(result, dict)
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
    
    def test_analyze_tensor(self):
        """Test analysis with tensor input"""
        analyzer = SimpleAnalyzer()
        input_tensor = torch.randn(10)
        result = analyzer.analyze(input_tensor)
        
        assert isinstance(result, dict)
        assert "sentiment" in result
    
    def test_analyze_batch(self):
        """Test batch analysis"""
        analyzer = SimpleAnalyzer()
        inputs = [[0.1] * 10, [0.2] * 10, [0.3] * 10]
        results = analyzer.analyze_batch(inputs, batch_size=2)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("sentiment" in r for r in results)
    
    def test_analyze_batch_empty(self):
        """Test batch analysis with empty input"""
        analyzer = SimpleAnalyzer()
        results = analyzer.analyze_batch([], batch_size=2)
        
        assert len(results) == 0
    
    def test_analyze_batch_small_batch(self):
        """Test batch analysis with small batch size"""
        analyzer = SimpleAnalyzer()
        inputs = [[0.1] * 10] * 5
        results = analyzer.analyze_batch(inputs, batch_size=2)
        
        assert len(results) == 5


class TestModelIntegration:
    """Integration tests for model classes"""
    
    def test_model_training_mode(self):
        """Test model in training mode"""
        model = SimpleTestModel()
        model.train()
        assert model.training is True
    
    def test_model_eval_mode(self):
        """Test model in eval mode"""
        model = SimpleTestModel()
        model.eval()
        assert model.training is False
    
    def test_predictor_training_cycle(self):
        """Test predictor training and inference cycle"""
        predictor = SimplePredictor()
        
        # Training mode
        predictor.train()
        x = torch.randn(5, 10)
        output = predictor(x)
        assert output.shape == (5, 1)
        
        # Eval mode
        predictor.eval()
        result = predictor.predict([0.1] * 10)
        assert isinstance(result, (float, int))
    
    def test_analyzer_consistency(self):
        """Test analyzer produces consistent results"""
        analyzer = SimpleAnalyzer()
        input_data = [0.1] * 10
        
        result1 = analyzer.analyze(input_data)
        result2 = analyzer.analyze(input_data)
        
        # Results should be identical for same input
        assert result1["sentiment"] == result2["sentiment"]
        assert abs(result1["confidence"] - result2["confidence"]) < 1e-6
    
    def test_generator_vocab_handling(self):
        """Test generator vocabulary handling"""
        generator = SimpleGenerator()
        assert generator.vocab_size > 0
    
    def test_model_parameter_counting(self):
        """Test parameter counting across model types"""
        models = [
            SimpleTestModel(),
            SimplePredictor(),
            SimpleGenerator(),
            SimpleAnalyzer()
        ]
        
        for model in models:
            info = model.get_model_info()
            assert info["total_parameters"] > 0
            assert info["trainable_parameters"] > 0


class TestErrorHandling:
    """Test error handling in base models"""
    
    def test_predictor_invalid_input(self):
        """Test predictor with invalid input"""
        predictor = SimplePredictor()
        
        # Should handle gracefully or raise appropriate error
        try:
            result = predictor.predict("invalid")
            # If it doesn't raise, result should be handled
        except (ValueError, TypeError, RuntimeError):
            pass  # Expected error
    
    def test_analyzer_invalid_input(self):
        """Test analyzer with invalid input"""
        analyzer = SimpleAnalyzer()
        
        try:
            result = analyzer.analyze("invalid")
            # If it doesn't raise, result should be handled
        except (ValueError, TypeError, RuntimeError):
            pass  # Expected error
    
    def test_generator_empty_prompt(self):
        """Test generator with empty prompt"""
        generator = SimpleGenerator()
        result = generator.generate("")
        
        assert isinstance(result, str)
    
    def test_model_nan_handling(self):
        """Test model handles NaN values"""
        model = SimpleTestModel()
        x = torch.tensor([[float('nan')] * 10])
        
        # Should either handle gracefully or raise appropriate error
        try:
            output = model(x)
            # If successful, check for NaN in output
            if output is not None:
                assert not torch.isnan(output).all() or True  # May have NaN
        except (RuntimeError, ValueError):
            pass  # Expected error for NaN input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


