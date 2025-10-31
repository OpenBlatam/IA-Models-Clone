from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from typing import List
from deep_learning_models import (
        import time
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Deep Learning Models Tests
==========================

Comprehensive test suite for deep learning models and training pipelines.
"""


    ModelConfig, ModelArchitecture, TaskType, BlogDataset,
    CustomTransformerModel, LSTMAttentionModel, CNNLSTMModel,
    ModelFactory, DeepLearningTrainer, AttentionMechanism
)


class TestModelConfig:
    """Test model configuration."""
    
    def test_config_initialization(self) -> Any:
        """Test configuration initialization."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=256
        )
        
        assert config.architecture == ModelArchitecture.CUSTOM_TRANSFORMER
        assert config.task_type == TaskType.SENTIMENT_CLASSIFICATION
        assert config.num_classes == 2
        assert config.hidden_size == 256
        assert config.use_mixed_precision == True
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_gpus=2
        )
        
        # Should enable distributed training with multiple GPUs
        assert config.use_distributed_training == True
    
    def test_mixed_precision_disabled_cpu(self) -> Any:
        """Test mixed precision disabled on CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            config = ModelConfig(
                architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
                task_type=TaskType.SENTIMENT_CLASSIFICATION,
                use_mixed_precision=True
            )
            
            assert config.use_mixed_precision == False


class TestAttentionMechanism:
    """Test attention mechanism."""
    
    def test_attention_initialization(self) -> Any:
        """Test attention mechanism initialization."""
        attention = AttentionMechanism(hidden_size=256, num_heads=8)
        
        assert attention.hidden_size == 256
        assert attention.num_heads == 8
        assert attention.head_size == 32
    
    def test_attention_forward_pass(self) -> Any:
        """Test attention forward pass."""
        attention = AttentionMechanism(hidden_size=256, num_heads=8)
        
        batch_size, seq_len, hidden_size = 2, 10, 256
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        
        output = attention(x, mask)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
    
    def test_attention_with_mask(self) -> Any:
        """Test attention with padding mask."""
        attention = AttentionMechanism(hidden_size=256, num_heads=8)
        
        batch_size, seq_len, hidden_size = 2, 10, 256
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        mask[0, 5:] = 0  # Pad second half of first sequence
        
        output = attention(x, mask)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()


class TestCustomTransformerModel:
    """Test custom transformer model."""
    
    @pytest.fixture
    def config(self) -> Any:
        return ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=256,
            num_layers=2,
            attention_heads=8,
            max_length=128
        )
    
    @pytest.fixture
    def model(self, config) -> Any:
        return CustomTransformerModel(config)
    
    def test_model_initialization(self, model, config) -> Any:
        """Test model initialization."""
        assert isinstance(model, nn.Module)
        assert model.config == config
        
        # Check components
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'position_encoding')
        assert hasattr(model, 'transformer_layers')
        assert len(model.transformer_layers) == config.num_layers
        assert hasattr(model, 'classifier')
    
    def test_model_forward_pass_classification(self, model) -> Any:
        """Test model forward pass for classification."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, 2)
        assert not torch.isnan(outputs['logits']).any()
    
    def test_model_forward_pass_regression(self) -> Any:
        """Test model forward pass for regression."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.QUALITY_REGRESSION,
            hidden_size=256,
            num_layers=2,
            attention_heads=8,
            max_length=128
        )
        
        model = CustomTransformerModel(config)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'predictions' in outputs
        assert outputs['predictions'].shape == (batch_size,)
        assert not torch.isnan(outputs['predictions']).any()
    
    def test_model_forward_pass_multitask(self) -> Any:
        """Test model forward pass for multi-task."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.MULTI_TASK,
            hidden_size=256,
            num_layers=2,
            attention_heads=8,
            max_length=128
        )
        
        model = CustomTransformerModel(config)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'sentiment_logits' in outputs
        assert 'quality_predictions' in outputs
        assert 'readability_predictions' in outputs
        assert outputs['sentiment_logits'].shape == (batch_size, 2)
        assert outputs['quality_predictions'].shape == (batch_size,)
        assert outputs['readability_predictions'].shape == (batch_size,)


class TestLSTMAttentionModel:
    """Test LSTM with attention model."""
    
    @pytest.fixture
    def config(self) -> Any:
        return ModelConfig(
            architecture=ModelArchitecture.LSTM_ATTENTION,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=256,
            lstm_hidden_size=128,
            attention_heads=8,
            bidirectional=True,
            max_length=128
        )
    
    @pytest.fixture
    def model(self, config) -> Any:
        return LSTMAttentionModel(config)
    
    def test_model_initialization(self, model, config) -> Any:
        """Test model initialization."""
        assert isinstance(model, nn.Module)
        assert model.config == config
        
        # Check components
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'classifier')
    
    def test_model_forward_pass(self, model) -> Any:
        """Test model forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, 2)
        assert not torch.isnan(outputs['logits']).any()


class TestCNNLSTMModel:
    """Test CNN-LSTM hybrid model."""
    
    @pytest.fixture
    def config(self) -> Any:
        return ModelConfig(
            architecture=ModelArchitecture.CNN_LSTM,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=256,
            lstm_hidden_size=128,
            cnn_filters=[64, 128],
            cnn_kernel_sizes=[3, 4],
            attention_heads=8,
            bidirectional=True,
            max_length=128
        )
    
    @pytest.fixture
    def model(self, config) -> Any:
        return CNNLSTMModel(config)
    
    def test_model_initialization(self, model, config) -> Any:
        """Test model initialization."""
        assert isinstance(model, nn.Module)
        assert model.config == config
        
        # Check components
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'conv_layers')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'classifier')
    
    def test_model_forward_pass(self, model) -> Any:
        """Test model forward pass."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, 2)
        assert not torch.isnan(outputs['logits']).any()


class TestModelFactory:
    """Test model factory."""
    
    def test_create_custom_transformer(self) -> Any:
        """Test creating custom transformer model."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2
        )
        
        model = ModelFactory.create_model(config)
        
        assert isinstance(model, CustomTransformerModel)
        assert model.config == config
    
    def test_create_lstm_attention(self) -> Any:
        """Test creating LSTM attention model."""
        config = ModelConfig(
            architecture=ModelArchitecture.LSTM_ATTENTION,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2
        )
        
        model = ModelFactory.create_model(config)
        
        assert isinstance(model, LSTMAttentionModel)
        assert model.config == config
    
    def test_create_cnn_lstm(self) -> Any:
        """Test creating CNN-LSTM model."""
        config = ModelConfig(
            architecture=ModelArchitecture.CNN_LSTM,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2
        )
        
        model = ModelFactory.create_model(config)
        
        assert isinstance(model, CNNLSTMModel)
        assert model.config == config
    
    @patch('deep_learning_models.TRANSFORMERS_AVAILABLE', False)
    def test_create_pretrained_model_no_transformers(self) -> Any:
        """Test creating pretrained model without transformers."""
        config = ModelConfig(
            architecture=ModelArchitecture.BERT,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2
        )
        
        with pytest.raises(ImportError):
            ModelFactory.create_model(config)
    
    def test_unsupported_architecture(self) -> Any:
        """Test unsupported architecture."""
        config = ModelConfig(
            architecture="unsupported",
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2
        )
        
        with pytest.raises(ValueError):
            ModelFactory.create_model(config)


class TestBlogDataset:
    """Test blog dataset."""
    
    @pytest.fixture
    def sample_data(self) -> Any:
        texts = [
            "This is a positive blog post.",
            "This is a negative blog post.",
            "This is a neutral blog post."
        ]
        labels = [1, 0, 1]
        return texts, labels
    
    def test_dataset_initialization(self, sample_data) -> Any:
        """Test dataset initialization."""
        texts, labels = sample_data
        dataset = BlogDataset(texts, labels, task_type=TaskType.SENTIMENT_CLASSIFICATION)
        
        assert len(dataset) == 3
        assert dataset.texts == texts
        assert dataset.labels == labels
    
    def test_dataset_without_labels(self, sample_data) -> Any:
        """Test dataset without labels."""
        texts, _ = sample_data
        dataset = BlogDataset(texts, task_type=TaskType.SENTIMENT_CLASSIFICATION)
        
        assert len(dataset) == 3
        assert dataset.labels is None
    
    def test_dataset_validation(self) -> Any:
        """Test dataset validation."""
        texts = ["Text 1", "Text 2"]
        labels = [1]  # Mismatched lengths
        
        with pytest.raises(ValueError):
            BlogDataset(texts, labels, task_type=TaskType.SENTIMENT_CLASSIFICATION)
    
    def test_dataset_getitem_with_tokenizer(self, sample_data) -> Optional[Dict[str, Any]]:
        """Test dataset getitem with tokenizer."""
        texts, labels = sample_data
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.return_value = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'token_type_ids': torch.zeros(1, 10)
        }
        
        dataset = BlogDataset(texts, labels, tokenizer, task_type=TaskType.SENTIMENT_CLASSIFICATION)
        
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'token_type_ids' in item
        assert 'labels' in item
        assert item['labels'].dtype == torch.long
    
    def test_dataset_getitem_without_tokenizer(self, sample_data) -> Optional[Dict[str, Any]]:
        """Test dataset getitem without tokenizer."""
        texts, labels = sample_data
        dataset = BlogDataset(texts, labels, task_type=TaskType.SENTIMENT_CLASSIFICATION)
        
        item = dataset[0]
        
        assert 'text' in item
        assert 'labels' in item
        assert item['text'] == texts[0]
        assert item['labels'].item() == labels[0]
    
    def test_dataset_regression_labels(self, sample_data) -> Any:
        """Test dataset with regression labels."""
        texts, _ = sample_data
        labels = [0.8, 0.2, 0.5]  # Float labels for regression
        
        dataset = BlogDataset(texts, labels, task_type=TaskType.QUALITY_REGRESSION)
        
        item = dataset[0]
        assert item['labels'].dtype == torch.float


class TestDeepLearningTrainer:
    """Test deep learning trainer."""
    
    @pytest.fixture
    def config(self) -> Any:
        return ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=128,
            num_layers=2,
            attention_heads=4,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=2,
            use_mixed_precision=False,  # Disable for testing
            use_gradient_accumulation=False  # Disable for testing
        )
    
    @pytest.fixture
    def model(self, config) -> Any:
        return ModelFactory.create_model(config)
    
    @pytest.fixture
    def trainer(self, config, model) -> Any:
        return DeepLearningTrainer(config, model)
    
    def test_trainer_initialization(self, trainer, config) -> Any:
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.device is not None
        assert trainer.optimizer is not None
        assert trainer.scaler is None  # Mixed precision disabled
        assert len(trainer.train_losses) == 0
        assert len(trainer.val_losses) == 0
    
    def test_trainer_with_mixed_precision(self, config, model) -> Any:
        """Test trainer with mixed precision."""
        config.use_mixed_precision = True
        trainer = DeepLearningTrainer(config, model)
        
        assert trainer.scaler is not None
    
    def test_compute_loss_classification(self, trainer) -> Any:
        """Test loss computation for classification."""
        batch_size = 2
        outputs = {'logits': torch.randn(batch_size, 2)}
        labels = {'labels': torch.randint(0, 2, (batch_size,))}
        
        loss = trainer._compute_loss(outputs, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_compute_loss_regression(self, config, model) -> Any:
        """Test loss computation for regression."""
        config.task_type = TaskType.QUALITY_REGRESSION
        trainer = DeepLearningTrainer(config, model)
        
        batch_size = 2
        outputs = {'predictions': torch.randn(batch_size)}
        labels = {'labels': torch.randn(batch_size)}
        
        loss = trainer._compute_loss(outputs, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_compute_loss_multitask(self, config, model) -> Any:
        """Test loss computation for multi-task."""
        config.task_type = TaskType.MULTI_TASK
        trainer = DeepLearningTrainer(config, model)
        
        batch_size = 2
        outputs = {
            'sentiment_logits': torch.randn(batch_size, 2),
            'quality_predictions': torch.randn(batch_size),
            'readability_predictions': torch.randn(batch_size)
        }
        labels = {
            'sentiment_labels': torch.randint(0, 2, (batch_size,)),
            'quality_labels': torch.randn(batch_size),
            'readability_labels': torch.randn(batch_size)
        }
        
        loss = trainer._compute_loss(outputs, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_unsupported_task_type(self, config, model) -> Any:
        """Test unsupported task type."""
        config.task_type = "unsupported"
        trainer = DeepLearningTrainer(config, model)
        
        outputs = {'logits': torch.randn(2, 2)}
        labels = {'labels': torch.randint(0, 2, (2,))}
        
        with pytest.raises(ValueError):
            trainer._compute_loss(outputs, labels)
    
    def test_save_and_load_model(self, trainer, tmp_path) -> Any:
        """Test model saving and loading."""
        model_path = tmp_path / "test_model.pth"
        
        # Save model
        trainer.save_model(str(model_path))
        assert model_path.exists()
        
        # Load model
        trainer.load_model(str(model_path))
        assert len(trainer.train_losses) >= 0  # Should not raise error


class TestIntegration:
    """Integration tests."""
    
    @pytest.fixture
    def sample_data(self) -> Any:
        texts = [
            "This is an excellent blog post about technology.",
            "I didn't like this article at all.",
            "This is a neutral article with facts.",
            "Amazing content with great insights!",
            "Terrible writing and poor structure."
        ]
        labels = [1, 0, 1, 1, 0]
        return texts, labels
    
    def test_end_to_end_training(self, sample_data) -> Any:
        """Test end-to-end training workflow."""
        texts, labels = sample_data
        
        # Configuration
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=128,
            num_layers=2,
            attention_heads=4,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=1,  # Single epoch for testing
            use_mixed_precision=False,
            use_gradient_accumulation=False
        )
        
        # Create model and dataset
        model = ModelFactory.create_model(config)
        dataset = BlogDataset(texts, labels, task_type=config.task_type)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create trainer
        trainer = DeepLearningTrainer(config, model)
        
        # Train
        history = trainer.train(train_dataset, val_dataset)
        
        # Verify results
        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) > 0
        assert len(history['val_losses']) > 0
        assert all(loss > 0 for loss in history['train_losses'])
        assert all(loss > 0 for loss in history['val_losses'])


class TestPerformance:
    """Performance tests."""
    
    def test_model_memory_usage(self) -> Any:
        """Test model memory usage."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=256,
            num_layers=4,
            attention_heads=8
        )
        
        model = ModelFactory.create_model(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable
    
    def test_forward_pass_speed(self) -> Any:
        """Test forward pass speed."""
        config = ModelConfig(
            architecture=ModelArchitecture.CUSTOM_TRANSFORMER,
            task_type=TaskType.SENTIMENT_CLASSIFICATION,
            num_classes=2,
            hidden_size=128,
            num_layers=2,
            attention_heads=4
        )
        
        model = ModelFactory.create_model(config)
        model.eval()
        
        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids, attention_mask)
        
        # Measure speed
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids, attention_mask)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Should complete within reasonable time
        assert avg_time < 1.0  # Less than 1 second per forward pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 