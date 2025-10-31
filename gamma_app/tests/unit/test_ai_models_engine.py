"""
Gamma App - AI Models Engine Unit Tests
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import torch
from transformers import AutoTokenizer, AutoModel

from engines.ai_models_engine import AIModelsEngine, ModelConfig, FineTuningConfig

class TestAIModelsEngine:
    """Unit tests for AI Models Engine"""
    
    @pytest.fixture
    def ai_engine(self):
        """Create AI engine instance for testing"""
        return AIModelsEngine()
    
    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing"""
        return ModelConfig(
            name="gpt2-small",
            model_type="text_generation",
            model_path="gpt2",
            device="cpu",
            max_length=512,
            temperature=0.7
        )
    
    @pytest.fixture
    def fine_tuning_config(self):
        """Create fine-tuning configuration for testing"""
        return FineTuningConfig(
            learning_rate=5e-5,
            num_epochs=3,
            batch_size=4,
            warmup_steps=100,
            save_steps=500,
            eval_steps=500,
            logging_steps=100
        )
    
    def test_ai_engine_initialization(self, ai_engine):
        """Test AI engine initialization"""
        assert ai_engine is not None
        assert ai_engine.loaded_models == {}
        assert ai_engine.model_configs == {}
    
    @patch('engines.ai_models_engine.AutoTokenizer.from_pretrained')
    @patch('engines.ai_models_engine.AutoModel.from_pretrained')
    def test_load_model_success(self, mock_model, mock_tokenizer, ai_engine, model_config):
        """Test successful model loading"""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Load model
        result = ai_engine.load_model(model_config)
        
        assert result is True
        assert model_config.name in ai_engine.loaded_models
        assert model_config.name in ai_engine.model_configs
        mock_tokenizer.assert_called_once_with(model_config.model_path)
        mock_model.assert_called_once_with(model_config.model_path)
    
    def test_load_model_invalid_config(self, ai_engine):
        """Test model loading with invalid configuration"""
        invalid_config = ModelConfig(
            name="invalid-model",
            model_type="invalid_type",
            model_path="invalid_path",
            device="cpu"
        )
        
        result = ai_engine.load_model(invalid_config)
        assert result is False
    
    def test_unload_model(self, ai_engine, model_config):
        """Test model unloading"""
        # First load a model
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        # Unload the model
        result = ai_engine.unload_model(model_config.name)
        
        assert result is True
        assert model_config.name not in ai_engine.loaded_models
        assert model_config.name not in ai_engine.model_configs
    
    def test_unload_nonexistent_model(self, ai_engine):
        """Test unloading a model that doesn't exist"""
        result = ai_engine.unload_model("nonexistent-model")
        assert result is False
    
    @patch('engines.ai_models_engine.AutoTokenizer.from_pretrained')
    @patch('engines.ai_models_engine.AutoModel.from_pretrained')
    def test_generate_text(self, mock_model, mock_tokenizer, ai_engine, model_config):
        """Test text generation"""
        # Mock the model and tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer_instance.decode.return_value = "Generated text"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_model_instance
        
        # Load model first
        ai_engine.load_model(model_config)
        
        # Generate text
        result = ai_engine.generate_text(
            model_name=model_config.name,
            prompt="Test prompt",
            max_length=100
        )
        
        assert result == "Generated text"
        mock_tokenizer_instance.encode.assert_called_once()
        mock_model_instance.generate.assert_called_once()
    
    def test_generate_text_nonexistent_model(self, ai_engine):
        """Test text generation with nonexistent model"""
        result = ai_engine.generate_text(
            model_name="nonexistent-model",
            prompt="Test prompt"
        )
        assert result is None
    
    def test_get_model_status(self, ai_engine, model_config):
        """Test getting model status"""
        # Load a model
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        status = ai_engine.get_model_status(model_config.name)
        
        assert status is not None
        assert status["name"] == model_config.name
        assert status["loaded"] is True
        assert status["device"] == model_config.device
    
    def test_get_model_status_nonexistent(self, ai_engine):
        """Test getting status for nonexistent model"""
        status = ai_engine.get_model_status("nonexistent-model")
        assert status is None
    
    def test_list_available_models(self, ai_engine):
        """Test listing available models"""
        models = ai_engine.list_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check that each model has required fields
        for model in models:
            assert "name" in model
            assert "type" in model
            assert "size" in model
            assert "description" in model
    
    @patch('engines.ai_models_engine.torch.cuda.is_available')
    def test_get_system_info(self, mock_cuda, ai_engine):
        """Test getting system information"""
        mock_cuda.return_value = True
        
        info = ai_engine.get_system_info()
        
        assert "cpu_count" in info
        assert "memory_total" in info
        assert "memory_available" in info
        assert "cuda_available" in info
        assert info["cuda_available"] is True
    
    def test_optimize_model(self, ai_engine, model_config):
        """Test model optimization"""
        # Load a model first
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        result = ai_engine.optimize_model(model_config.name)
        
        assert result is True
    
    def test_optimize_nonexistent_model(self, ai_engine):
        """Test optimizing nonexistent model"""
        result = ai_engine.optimize_model("nonexistent-model")
        assert result is False
    
    @patch('engines.ai_models_engine.AutoTokenizer.from_pretrained')
    @patch('engines.ai_models_engine.AutoModel.from_pretrained')
    def test_fine_tune_model(self, mock_model, mock_tokenizer, ai_engine, model_config, fine_tuning_config):
        """Test model fine-tuning"""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Load model first
        ai_engine.load_model(model_config)
        
        # Mock training data
        training_data = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well, thank you!"}
        ]
        
        result = ai_engine.fine_tune_model(
            model_name=model_config.name,
            training_data=training_data,
            config=fine_tuning_config
        )
        
        assert result is True
    
    def test_fine_tune_nonexistent_model(self, ai_engine, fine_tuning_config):
        """Test fine-tuning nonexistent model"""
        training_data = [{"input": "test", "output": "test"}]
        
        result = ai_engine.fine_tune_model(
            model_name="nonexistent-model",
            training_data=training_data,
            config=fine_tuning_config
        )
        
        assert result is False
    
    def test_evaluate_model(self, ai_engine, model_config):
        """Test model evaluation"""
        # Load a model first
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        # Mock test data
        test_data = [
            {"input": "Hello", "expected": "Hi there!"},
            {"input": "How are you?", "expected": "I'm doing well!"}
        ]
        
        metrics = ai_engine.evaluate_model(
            model_name=model_config.name,
            test_data=test_data
        )
        
        assert metrics is not None
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
    
    def test_evaluate_nonexistent_model(self, ai_engine):
        """Test evaluating nonexistent model"""
        test_data = [{"input": "test", "expected": "test"}]
        
        metrics = ai_engine.evaluate_model(
            model_name="nonexistent-model",
            test_data=test_data
        )
        
        assert metrics is None
    
    def test_quantize_model(self, ai_engine, model_config):
        """Test model quantization"""
        # Load a model first
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        result = ai_engine.quantize_model(
            model_name=model_config.name,
            quantization_type="int8"
        )
        
        assert result is True
    
    def test_quantize_nonexistent_model(self, ai_engine):
        """Test quantizing nonexistent model"""
        result = ai_engine.quantize_model(
            model_name="nonexistent-model",
            quantization_type="int8"
        )
        
        assert result is False
    
    def test_get_model_metrics(self, ai_engine, model_config):
        """Test getting model metrics"""
        # Load a model first
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        metrics = ai_engine.get_model_metrics(model_config.name)
        
        assert metrics is not None
        assert "memory_usage" in metrics
        assert "inference_time" in metrics
        assert "throughput" in metrics
    
    def test_get_model_metrics_nonexistent(self, ai_engine):
        """Test getting metrics for nonexistent model"""
        metrics = ai_engine.get_model_metrics("nonexistent-model")
        assert metrics is None
    
    def test_cleanup_resources(self, ai_engine, model_config):
        """Test resource cleanup"""
        # Load a model first
        ai_engine.loaded_models[model_config.name] = MagicMock()
        ai_engine.model_configs[model_config.name] = model_config
        
        ai_engine.cleanup_resources()
        
        # All models should be unloaded
        assert len(ai_engine.loaded_models) == 0
        assert len(ai_engine.model_configs) == 0

























