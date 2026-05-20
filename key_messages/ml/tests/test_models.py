from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from ..models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Tests for ML Models Module
"""

    BaseMessageModel,
    GPT2MessageModel,
    BERTClassifierModel,
    CustomTransformerModel,
    ModelFactory,
    ModelEnsemble,
    ModelConfig,
    DEFAULT_GPT2_CONFIG,
    DEFAULT_BERT_CONFIG,
    DEFAULT_CUSTOM_CONFIG
)

class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self) -> Any:
        """Test ModelConfig creation with default values."""
        config = ModelConfig(model_name="test-model")
        
        assert config.model_name == "test-model"
        assert config.max_length == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.do_sample is True
    
    def test_model_config_custom_values(self) -> Any:
        """Test ModelConfig creation with custom values."""
        config = ModelConfig(
            model_name="custom-model",
            max_length=256,
            temperature=0.5,
            top_p=0.8,
            do_sample=False,
            device="cpu"
        )
        
        assert config.model_name == "custom-model"
        assert config.max_length == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.do_sample is False
        assert config.device == "cpu"

class TestBaseMessageModel:
    """Test BaseMessageModel abstract class."""
    
    def test_base_model_initialization(self) -> Any:
        """Test BaseMessageModel initialization."""
        config = ModelConfig(model_name="test")
        
        # Create a concrete implementation for testing
        class TestModel(BaseMessageModel):
            def forward(self, input_ids, attention_mask=None) -> Any:
                return torch.randn(input_ids.shape[0], input_ids.shape[1], 1000)
            
            def generate(self, prompt, **kwargs) -> Any:
                return f"Generated: {prompt}"
            
            def load_model(self, path) -> Any:
                pass
        
        model = TestModel(config)
        
        assert model.config == config
        assert model.device == torch.device("cpu")  # Default device
    
    def test_base_model_abstract_methods(self) -> Any:
        """Test that abstract methods raise NotImplementedError."""
        config = ModelConfig(model_name="test")
        model = BaseMessageModel(config)
        
        with pytest.raises(NotImplementedError):
            model.forward(torch.tensor([[1, 2, 3]]))
        
        with pytest.raises(NotImplementedError):
            model.generate("test prompt")
        
        with pytest.raises(NotImplementedError):
            model.load_model("test_path")

class TestGPT2MessageModel:
    """Test GPT2MessageModel class."""
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForCausalLM')
    def test_gpt2_model_initialization(self, mock_model, mock_tokenizer) -> Any:
        """Test GPT2MessageModel initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="gpt2", device="cpu")
        model = GPT2MessageModel(config)
        
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once()
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForCausalLM')
    def test_gpt2_forward_pass(self, mock_model, mock_tokenizer) -> Any:
        """Test GPT2MessageModel forward pass."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.return_value.logits = torch.randn(1, 10, 50257)
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="gpt2", device="cpu")
        model = GPT2MessageModel(config)
        
        # Test forward pass
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        output = model.forward(input_ids, attention_mask)
        
        assert output.shape == (1, 10, 50257)
        mock_model_instance.assert_called_once()
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForCausalLM')
    def test_gpt2_generate(self, mock_model, mock_tokenizer) -> Any:
        """Test GPT2MessageModel text generation."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer_instance.decode.return_value = "Prompt: test prompt generated text"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="gpt2", device="cpu")
        model = GPT2MessageModel(config)
        
        # Test generation
        result = model.generate("test prompt")
        
        assert "generated text" in result
        mock_model_instance.generate.assert_called_once()

class TestBERTClassifierModel:
    """Test BERTClassifierModel class."""
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForSequenceClassification')
    def test_bert_model_initialization(self, mock_model, mock_tokenizer) -> Any:
        """Test BERTClassifierModel initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="bert-base-uncased", device="cpu")
        model = BERTClassifierModel(config, num_labels=5)
        
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance
        assert model.num_labels == 5
        mock_tokenizer.from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_model.from_pretrained.assert_called_once()
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForSequenceClassification')
    def test_bert_classify(self, mock_model, mock_tokenizer) -> Any:
        """Test BERTClassifierModel classification."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model_instance.return_value.logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="bert-base-uncased", device="cpu")
        model = BERTClassifierModel(config, num_labels=5)
        
        # Test classification
        result = model.classify("test text")
        
        assert isinstance(result, dict)
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result.values())
        assert abs(sum(result.values()) - 1.0) < 1e-6  # Probabilities sum to 1

class TestCustomTransformerModel:
    """Test CustomTransformerModel class."""
    
    def test_custom_model_initialization(self) -> Any:
        """Test CustomTransformerModel initialization."""
        config = ModelConfig(model_name="custom", device="cpu")
        model = CustomTransformerModel(config, vocab_size=1000, d_model=256)
        
        assert model.vocab_size == 1000
        assert model.d_model == 256
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.transformer, nn.TransformerEncoder)
        assert isinstance(model.output_layer, nn.Linear)
    
    def test_custom_model_forward_pass(self) -> Any:
        """Test CustomTransformerModel forward pass."""
        config = ModelConfig(model_name="custom", device="cpu")
        model = CustomTransformerModel(config, vocab_size=1000, d_model=256)
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        output = model.forward(input_ids, attention_mask)
        
        assert output.shape == (1, 5, 1000)  # (batch_size, seq_len, vocab_size)
    
    def test_custom_model_generate(self) -> Any:
        """Test CustomTransformerModel generation."""
        config = ModelConfig(model_name="custom", device="cpu")
        model = CustomTransformerModel(config, vocab_size=1000, d_model=256)
        
        result = model.generate("test prompt")
        
        assert isinstance(result, str)
        assert "test prompt" in result

class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_model_factory_available_models(self) -> Any:
        """Test ModelFactory.get_available_models."""
        available_models = ModelFactory.get_available_models()
        
        assert "gpt2" in available_models
        assert "bert" in available_models
        assert "custom" in available_models
    
    @patch('ml.models.GPT2MessageModel')
    def test_model_factory_create_gpt2(self, mock_gpt2) -> Any:
        """Test ModelFactory.create_model for GPT-2."""
        config = ModelConfig(model_name="gpt2")
        mock_gpt2.return_value = Mock()
        
        model = ModelFactory.create_model("gpt2", config)
        
        assert model is not None
        mock_gpt2.assert_called_once_with(config)
    
    @patch('ml.models.BERTClassifierModel')
    def test_model_factory_create_bert(self, mock_bert) -> Any:
        """Test ModelFactory.create_model for BERT."""
        config = ModelConfig(model_name="bert-base-uncased")
        mock_bert.return_value = Mock()
        
        model = ModelFactory.create_model("bert", config, num_labels=3)
        
        assert model is not None
        mock_bert.assert_called_once_with(config, num_labels=3)
    
    @patch('ml.models.CustomTransformerModel')
    def test_model_factory_create_custom(self, mock_custom) -> Any:
        """Test ModelFactory.create_model for custom model."""
        config = ModelConfig(model_name="custom")
        mock_custom.return_value = Mock()
        
        model = ModelFactory.create_model("custom", config, vocab_size=1000)
        
        assert model is not None
        mock_custom.assert_called_once_with(config, vocab_size=1000)
    
    def test_model_factory_invalid_model_type(self) -> Any:
        """Test ModelFactory.create_model with invalid model type."""
        config = ModelConfig(model_name="invalid")
        
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model("invalid", config)

class TestModelEnsemble:
    """Test ModelEnsemble class."""
    
    def test_model_ensemble_initialization(self) -> Any:
        """Test ModelEnsemble initialization."""
        # Create mock models
        model1 = Mock()
        model2 = Mock()
        models = [model1, model2]
        
        # Test with default weights
        ensemble = ModelEnsemble(models)
        
        assert ensemble.models == models
        assert ensemble.weights == [0.5, 0.5]
    
    def test_model_ensemble_custom_weights(self) -> Any:
        """Test ModelEnsemble initialization with custom weights."""
        model1 = Mock()
        model2 = Mock()
        models = [model1, model2]
        weights = [0.7, 0.3]
        
        ensemble = ModelEnsemble(models, weights)
        
        assert ensemble.models == models
        assert ensemble.weights == weights
    
    def test_model_ensemble_weight_mismatch(self) -> Any:
        """Test ModelEnsemble initialization with weight mismatch."""
        model1 = Mock()
        model2 = Mock()
        models = [model1, model2]
        weights = [0.5]  # Only one weight for two models
        
        with pytest.raises(ValueError, match="Number of weights must match number of models"):
            ModelEnsemble(models, weights)
    
    def test_model_ensemble_generate(self) -> Any:
        """Test ModelEnsemble text generation."""
        # Create mock models
        model1 = Mock()
        model1.generate.return_value = "output1"
        model2 = Mock()
        model2.generate.return_value = "output2"
        
        models = [model1, model2]
        weights = [0.6, 0.4]
        ensemble = ModelEnsemble(models, weights)
        
        result = ensemble.generate_ensemble("test prompt")
        
        assert isinstance(result, str)
        model1.generate.assert_called_once_with("test prompt")
        model2.generate.assert_called_once_with("test prompt")
    
    def test_model_ensemble_classify(self) -> Any:
        """Test ModelEnsemble classification."""
        # Create mock models with classify method
        model1 = Mock()
        model1.classify.return_value = {"positive": 0.8, "negative": 0.2}
        model2 = Mock()
        model2.classify.return_value = {"positive": 0.6, "negative": 0.4}
        
        models = [model1, model2]
        weights = [0.6, 0.4]
        ensemble = ModelEnsemble(models, weights)
        
        result = ensemble.classify_ensemble("test text")
        
        assert isinstance(result, dict)
        assert "positive" in result
        assert "negative" in result
        # Weighted average: 0.8*0.6 + 0.6*0.4 = 0.72
        assert abs(result["positive"] - 0.72) < 1e-6

class TestModelConfigurations:
    """Test default model configurations."""
    
    def test_default_gpt2_config(self) -> Any:
        """Test DEFAULT_GPT2_CONFIG."""
        assert DEFAULT_GPT2_CONFIG.model_name == "gpt2"
        assert DEFAULT_GPT2_CONFIG.max_length == 512
        assert DEFAULT_GPT2_CONFIG.temperature == 0.7
        assert DEFAULT_GPT2_CONFIG.top_p == 0.9
        assert DEFAULT_GPT2_CONFIG.do_sample is True
    
    def test_default_bert_config(self) -> Any:
        """Test DEFAULT_BERT_CONFIG."""
        assert DEFAULT_BERT_CONFIG.model_name == "bert-base-uncased"
        assert DEFAULT_BERT_CONFIG.max_length == 512
        assert DEFAULT_BERT_CONFIG.temperature == 1.0
        assert DEFAULT_BERT_CONFIG.do_sample is False
    
    def test_default_custom_config(self) -> Any:
        """Test DEFAULT_CUSTOM_CONFIG."""
        assert DEFAULT_CUSTOM_CONFIG.model_name == "custom"
        assert DEFAULT_CUSTOM_CONFIG.max_length == 256
        assert DEFAULT_CUSTOM_CONFIG.temperature == 0.8
        assert DEFAULT_CUSTOM_CONFIG.do_sample is True

class TestModelPersistence:
    """Test model saving and loading."""
    
    @patch('ml.models.AutoTokenizer')
    @patch('ml.models.AutoModelForCausalLM')
    def test_gpt2_model_save_load(self, mock_model, mock_tokenizer) -> Any:
        """Test GPT2MessageModel save and load functionality."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        config = ModelConfig(model_name="gpt2", device="cpu")
        model = GPT2MessageModel(config)
        
        # Test save
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model")
            model.save_model(save_path)
            
            # Verify save was called
            mock_model_instance.save_pretrained.assert_called_once_with(save_path)
            mock_tokenizer_instance.save_pretrained.assert_called_once_with(save_path)
            
            # Test load
            model.load_model(save_path)
            mock_tokenizer.from_pretrained.assert_called_with(save_path)
            mock_model.from_pretrained.assert_called_with(save_path)

match __name__:
    case "__main__":
    pytest.main([__file__]) 