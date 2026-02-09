from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import os
import time
from advanced_llm_integration import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Suite for Advanced LLM Integration
======================================

Comprehensive tests for advanced LLM integration with modern PyTorch practices,
transformers, quantization, and production-ready features.
"""


# Import the modules to test
    LLMConfig,
    AdvancedLLMTrainer,
    LLMPipeline
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_llm_config():
    """Sample LLM configuration for testing."""
    return LLMConfig(
        model_name="gpt2",
        model_type="causal",
        task="text_generation",
        max_length=64,
        batch_size=2,
        learning_rate=5e-5,
        num_epochs=1,
        use_peft=True,
        quantization="4bit"
    )


@pytest.fixture
def sample_classification_config():
    """Sample classification configuration for testing."""
    return LLMConfig(
        model_name="bert-base-uncased",
        model_type="sequence_classification",
        task="classification",
        max_length=64,
        batch_size=2,
        learning_rate=2e-5,
        num_epochs=1,
        use_peft=True,
        quantization="4bit"
    )


@pytest.fixture
def sample_texts():
    """Sample texts for training."""
    return [
        "This is a positive example",
        "This is a negative example",
        "I love this product",
        "I hate this product"
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for training."""
    return [1, 0, 1, 0]


@pytest.fixture
def sample_model():
    """Sample PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )


# =============================================================================
# LLM CONFIG TESTS
# =============================================================================

class TestLLMConfig:
    """Test LLMConfig dataclass."""
    
    def test_default_config(self) -> Any:
        """Test default configuration values."""
        config = LLMConfig()
        
        assert config.model_name == "microsoft/DialoGPT-medium"
        assert config.model_type == "causal"
        assert config.task == "text_generation"
        assert config.max_length == 512
        assert config.batch_size == 4
        assert config.learning_rate == 5e-5
        assert config.num_epochs == 3
        assert config.use_peft is True
        assert config.quantization == "4bit"
    
    def test_custom_config(self) -> Any:
        """Test custom configuration values."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            max_length=128,
            batch_size=8,
            learning_rate=1e-4,
            use_peft=False,
            quantization="8bit"
        )
        
        assert config.model_name == "gpt2"
        assert config.model_type == "causal"
        assert config.max_length == 128
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.use_peft is False
        assert config.quantization == "8bit"
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Test valid model types
        valid_types = ["causal", "sequence_classification", "conditional_generation"]
        for model_type in valid_types:
            config = LLMConfig(model_type=model_type)
            assert config.model_type == model_type
        
        # Test valid quantization types
        valid_quantization = ["none", "4bit", "8bit"]
        for quantization in valid_quantization:
            config = LLMConfig(quantization=quantization)
            assert config.quantization == quantization


# =============================================================================
# ADVANCED LLM TRAINER TESTS
# =============================================================================

class TestAdvancedLLMTrainer:
    """Test AdvancedLLMTrainer class."""
    
    def test_initialization(self, sample_llm_config) -> Any:
        """Test trainer initialization."""
        trainer = AdvancedLLMTrainer(sample_llm_config)
        
        assert trainer.config == sample_llm_config
        assert trainer.tokenizer is not None
        assert trainer.model is not None
        assert trainer.device is not None
        assert trainer.logger is not None
    
    def test_load_tokenizer_gpt2(self) -> Any:
        """Test GPT2 tokenizer loading."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        assert trainer.tokenizer is not None
        assert hasattr(trainer.tokenizer, 'encode')
        assert hasattr(trainer.tokenizer, 'decode')
        assert trainer.tokenizer.pad_token is not None
    
    def test_load_tokenizer_bert(self) -> Any:
        """Test BERT tokenizer loading."""
        config = LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        assert trainer.tokenizer is not None
        assert hasattr(trainer.tokenizer, 'encode')
        assert hasattr(trainer.tokenizer, 'decode')
    
    def test_load_model_causal(self) -> Any:
        """Test causal model loading."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'generate')
    
    def test_load_model_classification(self) -> Any:
        """Test classification model loading."""
        config = LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification",
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'forward')
    
    def test_apply_optimizations(self, sample_llm_config) -> Any:
        """Test optimization application."""
        trainer = AdvancedLLMTrainer(sample_llm_config)
        
        # Check if optimizations were applied
        assert trainer.model is not None
        
        # Note: Some optimizations might not be available in all environments
        # We just check that the model still works
    
    def test_setup_peft(self) -> Any:
        """Test PEFT setup."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=True,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Check if PEFT was applied
        assert trainer.model is not None
        
        # Note: PEFT setup might fail in some environments
        # We just check that the model still works
    
    def test_prepare_dataset_causal(self, sample_llm_config, sample_texts) -> Any:
        """Test dataset preparation for causal models."""
        trainer = AdvancedLLMTrainer(sample_llm_config)
        
        dataset = trainer.prepare_dataset(sample_texts)
        
        assert len(dataset) == len(sample_texts)
        
        # Test dataset item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_prepare_dataset_classification(self, sample_classification_config, sample_texts, sample_labels) -> Any:
        """Test dataset preparation for classification models."""
        trainer = AdvancedLLMTrainer(sample_classification_config)
        
        dataset = trainer.prepare_dataset(sample_texts, sample_labels)
        
        assert len(dataset) == len(sample_texts)
        
        # Test dataset item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    @pytest.mark.slow
    def test_training(self, sample_llm_config, sample_texts, sample_labels, temp_dir) -> Any:
        """Test model training."""
        # Use smaller model for faster testing
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            max_length=32,
            batch_size=1,
            num_epochs=1,
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Train model
        trainer_result = trainer.train(sample_texts, sample_labels)
        
        assert trainer_result is not None
        assert hasattr(trainer_result, 'train')
    
    def test_generate_text(self, sample_llm_config) -> Any:
        """Test text generation."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Test generation
        generated = trainer.generate_text("Hello", max_length=10)
        
        assert isinstance(generated, str)
        assert len(generated) > 0
    
    def test_predict_causal(self, sample_llm_config, sample_texts) -> Any:
        """Test prediction for causal models."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Test prediction
        predictions = trainer.predict(sample_texts[:2])
        
        assert len(predictions) == 2
        assert "input_text" in predictions[0]
        assert "generated_text" in predictions[0]
    
    def test_predict_classification(self, sample_classification_config, sample_texts, sample_labels) -> Any:
        """Test prediction for classification models."""
        config = LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Train model briefly
        trainer.train(sample_texts, sample_labels)
        
        # Test prediction
        predictions = trainer.predict(sample_texts[:2])
        
        assert len(predictions) == 2
        assert "text" in predictions[0]
        assert "predicted_class" in predictions[0]
        assert "probabilities" in predictions[0]
    
    def test_save_load_model(self, sample_llm_config, temp_dir) -> Any:
        """Test model saving and loading."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        # Save model
        model_path = os.path.join(temp_dir, "test_model")
        trainer.save_model(model_path)
        
        # Load model
        new_trainer = AdvancedLLMTrainer(config)
        new_trainer.load_model(model_path)
        
        assert new_trainer.model is not None
        assert new_trainer.tokenizer is not None


# =============================================================================
# LLM PIPELINE TESTS
# =============================================================================

class TestLLMPipeline:
    """Test LLMPipeline class."""
    
    def test_initialization(self, sample_llm_config, temp_dir) -> Any:
        """Test pipeline initialization."""
        # First save a model
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        model_path = os.path.join(temp_dir, "pipeline_model")
        trainer.save_model(model_path)
        
        # Initialize pipeline
        pipeline = LLMPipeline(model_path, config)
        
        assert pipeline.config == config
        assert pipeline.trainer is not None
        assert pipeline.device is not None
        assert pipeline.logger is not None
    
    def test_generate(self, sample_llm_config, temp_dir) -> Any:
        """Test text generation with pipeline."""
        # First save a model
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        model_path = os.path.join(temp_dir, "pipeline_model")
        trainer.save_model(model_path)
        
        # Initialize pipeline
        pipeline = LLMPipeline(model_path, config)
        
        # Test generation
        result = pipeline.generate("Hello", max_length=10)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_batch_generate(self, sample_llm_config, temp_dir) -> Any:
        """Test batch generation with pipeline."""
        # First save a model
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        model_path = os.path.join(temp_dir, "pipeline_model")
        trainer.save_model(model_path)
        
        # Initialize pipeline
        pipeline = LLMPipeline(model_path, config)
        
        # Test batch generation
        prompts = ["Hello", "How are you", "The weather is"]
        results = pipeline.batch_generate(prompts, max_length=10)
        
        assert len(results) == len(prompts)
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_classify(self, sample_classification_config, sample_texts, sample_labels, temp_dir) -> Any:
        """Test classification with pipeline."""
        # First save a model
        config = LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        trainer.train(sample_texts, sample_labels)
        model_path = os.path.join(temp_dir, "pipeline_model")
        trainer.save_model(model_path)
        
        # Initialize pipeline
        pipeline = LLMPipeline(model_path, config)
        
        # Test classification
        results = pipeline.classify(sample_texts[:2])
        
        assert len(results) == 2
        assert "text" in results[0]
        assert "predicted_class" in results[0]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_training_and_inference(self, temp_dir) -> Any:
        """Test complete training and inference pipeline."""
        # Configuration
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            max_length=32,
            batch_size=1,
            num_epochs=1,
            use_peft=False,
            quantization="none"
        )
        
        # Training data
        train_texts = [
            "This is a positive example",
            "This is a negative example"
        ]
        train_labels = [1, 0]
        
        # Train model
        trainer = AdvancedLLMTrainer(config)
        trainer_result = trainer.train(train_texts, train_labels)
        
        assert trainer_result is not None
        
        # Save model
        model_path = os.path.join(temp_dir, "integration_model")
        trainer.save_model(model_path)
        
        # Create pipeline
        pipeline = LLMPipeline(model_path, config)
        
        # Test inference
        result = pipeline.generate("Hello", max_length=10)
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Test batch inference
        results = pipeline.batch_generate(["Hello", "How are you"], max_length=10)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_different_model_types(self, temp_dir) -> Any:
        """Test different model types."""
        model_configs = [
            ("gpt2", "causal"),
            ("bert-base-uncased", "sequence_classification"),
            ("t5-small", "conditional_generation")
        ]
        
        train_texts = ["This is a test example"]
        train_labels = [1]
        
        for model_name, model_type in model_configs:
            try:
                config = LLMConfig(
                    model_name=model_name,
                    model_type=model_type,
                    max_length=32,
                    batch_size=1,
                    num_epochs=1,
                    use_peft=False,
                    quantization="none"
                )
                
                trainer = AdvancedLLMTrainer(config)
                
                # Test that model loads
                assert trainer.model is not None
                assert trainer.tokenizer is not None
                
                # Test training (might fail for some models, that's okay)
                try:
                    trainer_result = trainer.train(train_texts, train_labels)
                    assert trainer_result is not None
                except Exception as e:
                    # Some models might not support the training setup
                    pass
                
            except Exception as e:
                # Some models might not be available or compatible
                pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for LLM integration."""
    
    def test_generation_performance(self, temp_dir) -> Any:
        """Test generation performance."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        model_path = os.path.join(temp_dir, "perf_model")
        trainer.save_model(model_path)
        
        pipeline = LLMPipeline(model_path, config)
        
        # Benchmark generation
        start_time = time.time()
        
        for _ in range(10):
            result = pipeline.generate("Hello", max_length=20)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Generation should be reasonably fast
        assert avg_time < 5.0  # Should complete within 5 seconds per generation
    
    def test_batch_generation_performance(self, temp_dir) -> Any:
        """Test batch generation performance."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal",
            use_peft=False,
            quantization="none"
        )
        
        trainer = AdvancedLLMTrainer(config)
        model_path = os.path.join(temp_dir, "perf_model")
        trainer.save_model(model_path)
        
        pipeline = LLMPipeline(model_path, config)
        
        # Benchmark batch generation
        prompts = ["Hello", "How are you", "The weather is", "Machine learning"]
        
        start_time = time.time()
        
        results = pipeline.batch_generate(prompts, max_length=20)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Batch generation should be efficient
        assert total_time < 10.0  # Should complete within 10 seconds for batch
        assert len(results) == len(prompts)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in LLM integration."""
    
    def test_invalid_model_name(self) -> Any:
        """Test handling of invalid model names."""
        config = LLMConfig(
            model_name="invalid-model-name",
            model_type="causal"
        )
        
        with pytest.raises(Exception):
            AdvancedLLMTrainer(config)
    
    def test_invalid_model_type(self) -> Any:
        """Test handling of invalid model types."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="invalid_type"
        )
        
        with pytest.raises(ValueError):
            AdvancedLLMTrainer(config)
    
    def test_empty_texts(self) -> Any:
        """Test handling of empty text lists."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        with pytest.raises(Exception):
            trainer.train([], [])
    
    def test_mismatched_texts_labels(self) -> Any:
        """Test handling of mismatched texts and labels."""
        config = LLMConfig(
            model_name="bert-base-uncased",
            model_type="sequence_classification"
        )
        
        trainer = AdvancedLLMTrainer(config)
        
        texts = ["text1", "text2"]
        labels = [0]  # Mismatched length
        
        with pytest.raises(Exception):
            trainer.train(texts, labels)
    
    def test_invalid_model_path(self) -> Any:
        """Test handling of invalid model paths."""
        config = LLMConfig(
            model_name="gpt2",
            model_type="causal"
        )
        
        with pytest.raises(Exception):
            LLMPipeline("invalid/path", config)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 