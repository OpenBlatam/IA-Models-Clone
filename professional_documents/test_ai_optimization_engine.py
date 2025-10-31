"""
Comprehensive Test Suite for AI Optimization Engine
Test suite for advanced deep learning and LLM optimization features
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import json
import os
from pathlib import Path

# Import the modules to test
from ai_optimization_engine import (
    AIOptimizationEngine, ModelConfig, ModelType, OptimizationStrategy,
    AdvancedDataset, LoRALayer, QuantizedLinear, AdvancedTransformer,
    DiffusionModelWrapper, TrainingMetrics
)

class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_model_config_creation(self):
        """Test basic model configuration creation"""
        config = ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            max_length=512,
            batch_size=16
        )
        
        assert config.model_name == "gpt2"
        assert config.model_type == ModelType.LLM
        assert config.max_length == 512
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5  # Default value
        assert config.mixed_precision == True  # Default value
    
    def test_model_config_advanced_params(self):
        """Test advanced configuration parameters"""
        config = ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            lora_rank=32,
            lora_alpha=64,
            quantization_bits=4,
            pruning_ratio=0.2
        )
        
        assert config.lora_rank == 32
        assert config.lora_alpha == 64
        assert config.quantization_bits == 4
        assert config.pruning_ratio == 0.2

class TestLoRALayer:
    """Test LoRA layer implementation"""
    
    def test_lora_layer_creation(self):
        """Test LoRA layer creation and initialization"""
        lora_layer = LoRALayer(in_features=768, out_features=768, rank=16, alpha=32)
        
        assert lora_layer.rank == 16
        assert lora_layer.alpha == 32
        assert lora_layer.scaling == 32 / 16  # alpha / rank
        assert lora_layer.lora_A.in_features == 768
        assert lora_layer.lora_A.out_features == 16
        assert lora_layer.lora_B.in_features == 16
        assert lora_layer.lora_B.out_features == 768
    
    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass"""
        lora_layer = LoRALayer(in_features=10, out_features=5, rank=4, alpha=8)
        x = torch.randn(2, 10)
        
        output = lora_layer(x)
        
        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_lora_layer_dropout(self):
        """Test LoRA layer with dropout"""
        lora_layer = LoRALayer(in_features=10, out_features=5, rank=4, alpha=8, dropout=0.5)
        x = torch.randn(2, 10)
        
        # Test with dropout enabled
        lora_layer.train()
        output1 = lora_layer(x)
        output2 = lora_layer(x)
        
        # Outputs should be different due to dropout
        assert not torch.equal(output1, output2)
        
        # Test with dropout disabled
        lora_layer.eval()
        output3 = lora_layer(x)
        output4 = lora_layer(x)
        
        # Outputs should be the same without dropout
        assert torch.equal(output3, output4)

class TestQuantizedLinear:
    """Test quantized linear layer"""
    
    def test_quantized_linear_creation(self):
        """Test quantized linear layer creation"""
        quantized_layer = QuantizedLinear(in_features=10, out_features=5, bits=8)
        
        assert quantized_layer.bits == 8
        assert quantized_layer.in_features == 10
        assert quantized_layer.out_features == 5
        assert quantized_layer.weight.shape == (5, 10)
        assert quantized_layer.bias.shape == (5,)
    
    def test_quantized_linear_forward(self):
        """Test quantized linear layer forward pass"""
        quantized_layer = QuantizedLinear(in_features=10, out_features=5, bits=8)
        x = torch.randn(2, 10)
        
        output = quantized_layer(x)
        
        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_weight_quantization(self):
        """Test weight quantization functionality"""
        quantized_layer = QuantizedLinear(in_features=10, out_features=5, bits=4)
        
        # Set some weights
        quantized_layer.weight.data = torch.randn(5, 10) * 10
        
        quantized_weights = quantized_layer.quantize_weights()
        
        # Check that weights are quantized to 4-bit precision
        max_val = 2 ** (4 - 1) - 1  # 7 for 4-bit
        min_val = -max_val
        
        assert quantized_weights.max() <= max_val
        assert quantized_weights.min() >= min_val

class TestAdvancedDataset:
    """Test advanced dataset class"""
    
    def test_dataset_creation(self):
        """Test dataset creation with mock tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
        }
        
        data = [
            {'text': 'Sample text 1'},
            {'text': 'Sample text 2'},
            {'text': 'Sample text 3'}
        ]
        
        dataset = AdvancedDataset(data, mock_tokenizer, max_length=5)
        
        assert len(dataset) == 3
        assert dataset.max_length == 5
        assert dataset.augmentation == True
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
        }
        
        data = [
            {'text': 'Sample text', 'labels': [0, 1, 0]}
        ]
        
        dataset = AdvancedDataset(data, mock_tokenizer, max_length=5)
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert item['labels'].shape == (3,)

class TestTrainingMetrics:
    """Test training metrics tracking"""
    
    def test_training_metrics_creation(self):
        """Test training metrics initialization"""
        metrics = TrainingMetrics()
        
        assert metrics.epoch == 0
        assert metrics.step == 0
        assert metrics.loss == 0.0
        assert metrics.learning_rate == 0.0
        assert metrics.perplexity == 0.0
    
    def test_training_metrics_update(self):
        """Test training metrics update"""
        metrics = TrainingMetrics()
        
        metrics.epoch = 1
        metrics.step = 100
        metrics.loss = 0.5
        metrics.learning_rate = 1e-4
        metrics.perplexity = 2.5
        
        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-4
        assert metrics.perplexity == 2.5

class TestAIOptimizationEngine:
    """Test AI Optimization Engine main class"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing"""
        return AIOptimizationEngine()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample model configuration"""
        return ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            max_length=256,
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=1
        )
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        return [
            {"text": "This is a sample training text."},
            {"text": "Another example for training."},
            {"text": "More training data here."},
            {"text": "Final training example."}
        ]
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.models == {}
        assert engine.metrics == {}
        assert engine.training_history == []
        assert engine.optimization_strategies == {}
        assert engine.device in ["cpu", "cuda", "mps"]
    
    @pytest.mark.asyncio
    async def test_create_model_success(self, engine, sample_config):
        """Test successful model creation"""
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model.from_pretrained.return_value = Mock()
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            result = await engine.create_model("test_model", sample_config)
            
            assert result == True
            assert "test_model" in engine.models
            assert engine.models["test_model"]["status"] == "initialized"
            assert "test_model" in engine.metrics
    
    @pytest.mark.asyncio
    async def test_create_model_failure(self, engine, sample_config):
        """Test model creation failure"""
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model:
            mock_model.from_pretrained.side_effect = Exception("Model loading failed")
            
            result = await engine.create_model("test_model", sample_config)
            
            assert result == False
            assert "test_model" not in engine.models
    
    @pytest.mark.asyncio
    async def test_train_model_success(self, engine, sample_config, sample_training_data):
        """Test successful model training"""
        # First create a model
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model_instance.return_value = Mock(loss=torch.tensor(0.5))
            mock_model.from_pretrained.return_value = mock_model_instance
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
                'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
            }
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            await engine.create_model("test_model", sample_config)
        
        # Mock the training components
        with patch('ai_optimization_engine.DataLoader') as mock_dataloader, \
             patch('ai_optimization_engine.AdamW') as mock_optimizer, \
             patch('ai_optimization_engine.get_linear_schedule_with_warmup') as mock_scheduler:
            
            # Mock dataloader
            mock_batch = {
                'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
                'attention_mask': torch.tensor([[1, 1, 1, 0, 0]]),
                'labels': torch.tensor([[1, 2, 3, 0, 0]])
            }
            mock_dataloader.return_value = [mock_batch]
            
            # Mock optimizer and scheduler
            mock_optimizer.return_value = Mock()
            mock_scheduler.return_value = Mock()
            
            result = await engine.train_model("test_model", sample_training_data)
            
            assert result == True
            assert engine.models["test_model"]["status"] == "trained"
    
    @pytest.mark.asyncio
    async def test_optimize_model_quantization(self, engine, sample_config):
        """Test model quantization optimization"""
        # Create a model first
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            await engine.create_model("test_model", sample_config)
        
        # Test quantization
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            result = await engine.optimize_model(
                "test_model", 
                "quantization", 
                {"bits": 8}
            )
            
            assert result == True
            assert "optimizations" in engine.models["test_model"]
            assert len(engine.models["test_model"]["optimizations"]) == 1
            assert engine.models["test_model"]["optimizations"][0]["type"] == "quantization"
    
    @pytest.mark.asyncio
    async def test_optimize_model_pruning(self, engine, sample_config):
        """Test model pruning optimization"""
        # Create a model first
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            await engine.create_model("test_model", sample_config)
        
        # Test pruning
        with patch('torch.nn.utils.prune.ln_structured') as mock_prune:
            result = await engine.optimize_model(
                "test_model", 
                "pruning", 
                {"ratio": 0.1}
            )
            
            assert result == True
            assert "optimizations" in engine.models["test_model"]
            assert len(engine.models["test_model"]["optimizations"]) == 1
            assert engine.models["test_model"]["optimizations"][0]["type"] == "pruning"
    
    @pytest.mark.asyncio
    async def test_generate_content_text(self, engine, sample_config):
        """Test text content generation"""
        # Create a model first
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model.from_pretrained.return_value = mock_model_instance
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
            mock_tokenizer_instance.decode.return_value = "Generated text content"
            mock_tokenizer_instance.eos_token_id = 0
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            await engine.create_model("test_model", sample_config)
        
        # Test generation
        result = await engine.generate_content(
            "test_model",
            "Generate a story about",
            {"max_length": 50, "temperature": 0.8}
        )
        
        assert result["type"] == "text"
        assert result["content"] == "Generated text content"
        assert result["prompt"] == "Generate a story about"
    
    @pytest.mark.asyncio
    async def test_generate_content_image(self, engine):
        """Test image content generation with diffusion model"""
        # Create diffusion model config
        diffusion_config = ModelConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            model_type=ModelType.DIFFUSION
        )
        
        # Create a diffusion model
        with patch('ai_optimization_engine.StableDiffusionPipeline') as mock_pipeline:
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.generate.return_value = Mock(images=["generated_image"])
            mock_pipeline.from_pretrained.return_value = mock_pipeline_instance
            
            await engine.create_model("diffusion_model", diffusion_config)
        
        # Test generation
        result = await engine.generate_content(
            "diffusion_model",
            "A beautiful landscape",
            {"num_inference_steps": 20}
        )
        
        assert result["type"] == "image"
        assert result["content"] == "generated_image"
        assert result["prompt"] == "A beautiful landscape"
    
    @pytest.mark.asyncio
    async def test_get_model_performance(self, engine, sample_config):
        """Test model performance metrics retrieval"""
        # Create a model first
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            await engine.create_model("test_model", sample_config)
        
        # Update some metrics
        engine.metrics["test_model"].loss = 0.5
        engine.metrics["test_model"].accuracy = 0.85
        engine.models["test_model"]["final_metrics"] = {"perplexity": 2.3}
        
        # Test performance retrieval
        performance = await engine.get_model_performance("test_model")
        
        assert performance["model_id"] == "test_model"
        assert performance["status"] == "initialized"
        assert performance["training_metrics"]["loss"] == 0.5
        assert performance["training_metrics"]["accuracy"] == 0.85
        assert performance["final_metrics"]["perplexity"] == 2.3
        assert "system_metrics" in performance
    
    @pytest.mark.asyncio
    async def test_export_model(self, engine, sample_config):
        """Test model export functionality"""
        # Create a model first
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            await engine.create_model("test_model", sample_config)
        
        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "exported_model.pt")
            
            with patch('torch.save') as mock_save:
                result = await engine.export_model("test_model", export_path, "pytorch")
                
                assert result == True
                mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, engine):
        """Test resource cleanup"""
        with patch('pickle.dump') as mock_dump, \
             patch('torch.cuda.empty_cache') as mock_cache:
            
            await engine.cleanup_resources()
            
            mock_dump.assert_called_once()
            if torch.cuda.is_available():
                mock_cache.assert_called_once()

class TestIntegration:
    """Integration tests for the complete workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from model creation to generation"""
        engine = AIOptimizationEngine()
        
        config = ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            max_length=128,
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=1,
            optimization_strategy=OptimizationStrategy.LORA
        )
        
        training_data = [
            {"text": "Sample training text 1"},
            {"text": "Sample training text 2"},
            {"text": "Sample training text 3"},
            {"text": "Sample training text 4"}
        ]
        
        # Mock all external dependencies
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer, \
             patch('ai_optimization_engine.DataLoader') as mock_dataloader, \
             patch('ai_optimization_engine.AdamW') as mock_optimizer, \
             patch('ai_optimization_engine.get_linear_schedule_with_warmup') as mock_scheduler:
            
            # Setup mocks
            mock_model_instance = Mock()
            mock_model_instance.return_value = Mock(loss=torch.tensor(0.5))
            mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_model.from_pretrained.return_value = mock_model_instance
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
                'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
            }
            mock_tokenizer_instance.decode.return_value = "Generated content"
            mock_tokenizer_instance.eos_token_id = 0
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_batch = {
                'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),
                'attention_mask': torch.tensor([[1, 1, 1, 0, 0]]),
                'labels': torch.tensor([[1, 2, 3, 0, 0]])
            }
            mock_dataloader.return_value = [mock_batch]
            
            mock_optimizer.return_value = Mock()
            mock_scheduler.return_value = Mock()
            
            # Execute workflow
            # 1. Create model
            create_result = await engine.create_model("workflow_model", config)
            assert create_result == True
            
            # 2. Train model
            train_result = await engine.train_model("workflow_model", training_data)
            assert train_result == True
            
            # 3. Optimize model
            optimize_result = await engine.optimize_model(
                "workflow_model", 
                "quantization", 
                {"bits": 8}
            )
            assert optimize_result == True
            
            # 4. Generate content
            generation_result = await engine.generate_content(
                "workflow_model",
                "Test prompt",
                {"max_length": 50}
            )
            assert generation_result["type"] == "text"
            assert generation_result["content"] == "Generated content"
            
            # 5. Get performance
            performance = await engine.get_model_performance("workflow_model")
            assert performance["model_id"] == "workflow_model"
            assert performance["status"] == "trained"

# Performance tests
class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during operations"""
        engine = AIOptimizationEngine()
        
        config = ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            batch_size=1,  # Small batch for testing
            max_length=64
        )
        
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            # Create multiple models to test memory management
            for i in range(3):
                await engine.create_model(f"model_{i}", config)
            
            assert len(engine.models) == 3
            
            # Cleanup should work
            await engine.cleanup_resources()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent model operations"""
        engine = AIOptimizationEngine()
        
        config = ModelConfig(
            model_name="gpt2",
            model_type=ModelType.LLM,
            batch_size=1,
            max_length=32
        )
        
        with patch('ai_optimization_engine.AutoModelForCausalLM') as mock_model, \
             patch('ai_optimization_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_model_instance = Mock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            # Create multiple models concurrently
            tasks = []
            for i in range(5):
                task = engine.create_model(f"concurrent_model_{i}", config)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            assert all(results) == True
            assert len(engine.models) == 5

# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
























