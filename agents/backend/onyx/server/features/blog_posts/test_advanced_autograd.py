from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
import tempfile
import os
import warnings
from advanced_autograd_models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive tests for Advanced PyTorch Autograd Models
Tests all components including weight initialization, loss functions,
optimization algorithms, attention mechanisms, positional encodings,
LoRA/P-tuning fine-tuning, and proper tokenization
"""


# Import advanced autograd models
    ModelConfig, AdvancedPositionalEncoding, AdvancedMultiHeadAttention,
    LoRALayer, LoRALinear, P_TuningLayer, AdvancedLossFunctions,
    AdvancedOptimizers, AdvancedTokenizer, AdvancedTransformerModel,
    AdvancedTrainingPipeline
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestModelConfig:
    """Test ModelConfig dataclass"""
    
    def test_model_config_defaults(self) -> Any:
        """Test default configuration values"""
        config = ModelConfig()
        
        assert config.vocab_size == 50257
        assert config.d_model == 768
        assert config.n_layers == 12
        assert config.n_heads == 12
        assert config.d_ff == 3072
        assert config.max_seq_len == 512
        assert config.dropout == 0.1
        assert config.activation == "gelu"
        assert config.layer_norm_eps == 1e-5
        assert config.use_relative_pos is True
        assert config.max_relative_position == 32
        assert config.use_rope is False
        assert config.rope_dim == 64
        assert config.use_flash_attention is False
        assert config.use_xformers is False
        assert config.gradient_checkpointing is True
        assert config.mixed_precision is True
        assert config.weight_decay == 0.01
        assert config.learning_rate == 1e-4
        assert config.warmup_steps == 1000
        assert config.max_grad_norm == 1.0
        assert config.label_smoothing == 0.1
        assert config.focal_loss_alpha == 1.0
        assert config.focal_loss_gamma == 2.0
    
    def test_model_config_custom(self) -> Any:
        """Test custom configuration values"""
        config = ModelConfig(
            vocab_size=10000,
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_seq_len=256,
            dropout=0.2,
            activation="relu",
            use_rope=True,
            use_flash_attention=True
        )
        
        assert config.vocab_size == 10000
        assert config.d_model == 512
        assert config.n_layers == 6
        assert config.n_heads == 8
        assert config.d_ff == 2048
        assert config.max_seq_len == 256
        assert config.dropout == 0.2
        assert config.activation == "relu"
        assert config.use_rope is True
        assert config.use_flash_attention is True


class TestAdvancedPositionalEncoding:
    """Test AdvancedPositionalEncoding module"""
    
    def test_sinusoidal_encoding_shape(self) -> Any:
        """Test sinusoidal positional encoding shape"""
        d_model = 512
        max_len = 1000
        pe = AdvancedPositionalEncoding(d_model, max_len, encoding_type="sinusoidal")
        
        x = torch.randn(50, d_model)  # (seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_learnable_encoding_shape(self) -> Any:
        """Test learnable positional encoding shape"""
        d_model = 256
        max_len = 500
        pe = AdvancedPositionalEncoding(d_model, max_len, encoding_type="learnable")
        
        x = torch.randn(30, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_rope_encoding_shape(self) -> Any:
        """Test RoPE positional encoding shape"""
        d_model = 128
        max_len = 200
        pe = AdvancedPositionalEncoding(d_model, max_len, encoding_type="rope")
        
        x = torch.randn(20, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_positional_encoding_gradients(self) -> Any:
        """Test positional encoding gradients"""
        d_model = 64
        pe = AdvancedPositionalEncoding(d_model, encoding_type="sinusoidal", use_learnable=True)
        
        x = torch.randn(10, d_model, requires_grad=True)
        output = pe(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check learnable parameters have gradients
        if pe.learnable_pe is not None:
            assert pe.learnable_pe.grad is not None
            assert not torch.isnan(pe.learnable_pe.grad).any()
    
    def test_positional_encoding_device(self) -> Any:
        """Test positional encoding device transfer"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            d_model = 128
            pe = AdvancedPositionalEncoding(d_model, encoding_type="sinusoidal").to(device)
            
            x = torch.randn(15, d_model, device=device)
            output = pe(x)
            
            assert output.device == device
            assert not torch.isnan(output).any()
    
    def test_invalid_encoding_type(self) -> Any:
        """Test invalid encoding type raises error"""
        with pytest.raises(ValueError):
            AdvancedPositionalEncoding(64, encoding_type="invalid")


class TestAdvancedMultiHeadAttention:
    """Test AdvancedMultiHeadAttention module"""
    
    def test_standard_attention_shape(self) -> Any:
        """Test standard attention shape"""
        d_model = 512
        n_heads = 8
        attention = AdvancedMultiHeadAttention(d_model, n_heads, attention_type="standard")
        
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_relative_attention_shape(self) -> Any:
        """Test relative attention shape"""
        d_model = 256
        n_heads = 4
        attention = AdvancedMultiHeadAttention(d_model, n_heads, attention_type="relative")
        
        batch_size = 2
        seq_len = 15
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_local_attention_shape(self) -> Any:
        """Test local attention shape"""
        d_model = 128
        n_heads = 2
        attention = AdvancedMultiHeadAttention(d_model, n_heads, attention_type="local")
        
        batch_size = 3
        seq_len = 25
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_sparse_attention_shape(self) -> Any:
        """Test sparse attention shape"""
        d_model = 64
        n_heads = 1
        attention = AdvancedMultiHeadAttention(d_model, n_heads, attention_type="sparse")
        
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_with_mask(self) -> Any:
        """Test attention with mask"""
        d_model = 256
        n_heads = 4
        attention = AdvancedMultiHeadAttention(d_model, n_heads)
        
        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 15:] = 0  # Mask last 5 positions
        
        output = attention(x, mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_gradients(self) -> Any:
        """Test attention gradients"""
        d_model = 128
        n_heads = 2
        attention = AdvancedMultiHeadAttention(d_model, n_heads)
        
        x = torch.randn(3, 10, d_model, requires_grad=True)
        output = attention(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_attention_device(self) -> Any:
        """Test attention device transfer"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            d_model = 64
            attention = AdvancedMultiHeadAttention(d_model, 1).to(device)
            
            x = torch.randn(2, 8, d_model, device=device)
            output = attention(x)
            
            assert output.device == device
            assert not torch.isnan(output).any()


class TestLoRALayer:
    """Test LoRA layer"""
    
    def test_lora_layer_shape(self) -> Any:
        """Test LoRA layer shape"""
        in_features = 256
        out_features = 128
        rank = 16
        lora = LoRALayer(in_features, out_features, rank)
        
        x = torch.randn(4, in_features)
        output = lora(x)
        
        assert output.shape == (4, out_features)
        assert not torch.isnan(output).any()
    
    def test_lora_layer_gradients(self) -> Any:
        """Test LoRA layer gradients"""
        in_features = 128
        out_features = 64
        rank = 8
        lora = LoRALayer(in_features, out_features, rank)
        
        x = torch.randn(2, in_features, requires_grad=True)
        output = lora(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check LoRA parameters have gradients
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None
        assert not torch.isnan(lora.lora_A.weight.grad).any()
        assert not torch.isnan(lora.lora_B.weight.grad).any()
    
    def test_lora_scaling(self) -> Any:
        """Test LoRA scaling factor"""
        in_features = 64
        out_features = 32
        rank = 4
        alpha = 16.0
        lora = LoRALayer(in_features, out_features, rank, alpha)
        
        x = torch.randn(1, in_features)
        output = lora(x)
        
        # Check scaling is applied
        expected_scaling = alpha / rank
        assert lora.scaling == expected_scaling


class TestLoRALinear:
    """Test LoRA linear layer"""
    
    def test_lora_linear_shape(self) -> Any:
        """Test LoRA linear layer shape"""
        in_features = 128
        out_features = 64
        linear = nn.Linear(in_features, out_features)
        lora_linear = LoRALinear(linear, rank=8)
        
        x = torch.randn(3, in_features)
        output = lora_linear(x)
        
        assert output.shape == (3, out_features)
        assert not torch.isnan(output).any()
    
    def test_lora_linear_gradients(self) -> Any:
        """Test LoRA linear layer gradients"""
        in_features = 64
        out_features = 32
        linear = nn.Linear(in_features, out_features)
        lora_linear = LoRALinear(linear, rank=4)
        
        x = torch.randn(2, in_features, requires_grad=True)
        output = lora_linear(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check LoRA parameters have gradients
        assert lora_linear.lora.lora_A.weight.grad is not None
        assert lora_linear.lora.lora_B.weight.grad is not None
        
        # Check original linear weights are frozen
        assert not lora_linear.linear.weight.requires_grad


class TestP_TuningLayer:
    """Test P-tuning layer"""
    
    def test_p_tuning_shape(self) -> Any:
        """Test P-tuning layer shape"""
        d_model = 256
        prompt_length = 10
        p_tuning = P_TuningLayer(d_model, prompt_length)
        
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, d_model)
        output = p_tuning(x)
        
        expected_seq_len = prompt_length + seq_len
        assert output.shape == (batch_size, expected_seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_p_tuning_gradients(self) -> Any:
        """Test P-tuning layer gradients"""
        d_model = 128
        prompt_length = 5
        p_tuning = P_TuningLayer(d_model, prompt_length)
        
        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = p_tuning(x)
        
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check prompt embeddings have gradients
        assert p_tuning.prompt_embeddings.grad is not None
        assert not torch.isnan(p_tuning.prompt_embeddings.grad).any()


class TestAdvancedLossFunctions:
    """Test advanced loss functions"""
    
    def test_focal_loss(self) -> Any:
        """Test focal loss"""
        batch_size = 4
        num_classes = 3
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = AdvancedLossFunctions.focal_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss).any()
    
    def test_label_smoothing_loss(self) -> Any:
        """Test label smoothing loss"""
        batch_size = 3
        num_classes = 5
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = AdvancedLossFunctions.label_smoothing_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss).any()
    
    def test_contrastive_loss(self) -> Any:
        """Test contrastive loss"""
        batch_size = 4
        embedding_dim = 64
        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, 2, (batch_size,))
        
        loss = AdvancedLossFunctions.contrastive_loss(embeddings, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss).any()
    
    def test_triplet_loss(self) -> Any:
        """Test triplet loss"""
        batch_size = 3
        embedding_dim = 32
        anchor = torch.randn(batch_size, embedding_dim)
        positive = torch.randn(batch_size, embedding_dim)
        negative = torch.randn(batch_size, embedding_dim)
        
        loss = AdvancedLossFunctions.triplet_loss(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss).any()


class TestAdvancedOptimizers:
    """Test advanced optimizers"""
    
    def test_create_optimizer(self) -> Any:
        """Test optimizer creation"""
        config = ModelConfig()
        model = nn.Linear(64, 32)
        
        optimizer = AdvancedOptimizers.create_optimizer(model, config)
        
        assert isinstance(optimizer, optim.AdamW)
        assert len(optimizer.param_groups) == 2  # With and without weight decay
    
    def test_create_scheduler(self) -> Any:
        """Test scheduler creation"""
        config = ModelConfig()
        model = nn.Linear(32, 16)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        scheduler = AdvancedOptimizers.create_scheduler(optimizer, config, 1000)
        
        assert isinstance(scheduler, optim.lr_scheduler._LRScheduler)


class TestAdvancedTokenizer:
    """Test advanced tokenizer"""
    
    def test_tokenizer_initialization(self) -> Any:
        """Test tokenizer initialization"""
        tokenizer = AdvancedTokenizer("gpt2", max_length=256)
        
        assert tokenizer.max_length == 256
        assert tokenizer.padding == "max_length"
        assert tokenizer.truncation is True
        assert tokenizer.tokenizer.pad_token is not None
    
    def test_tokenize_text_single(self) -> Any:
        """Test tokenizing single text"""
        tokenizer = AdvancedTokenizer("gpt2", max_length=128)
        text = "Hello world, this is a test."
        
        tokenized = tokenizer.tokenize_text(text)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert tokenized['input_ids'].shape[1] <= 128
        assert not torch.isnan(tokenized['input_ids']).any()
    
    def test_tokenize_text_batch(self) -> Any:
        """Test tokenizing batch of texts"""
        tokenizer = AdvancedTokenizer("gpt2", max_length=64)
        texts = [
            "First sentence for testing.",
            "Second sentence for testing.",
            "Third sentence for testing."
        ]
        
        tokenized = tokenizer.tokenize_text(texts)
        
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert tokenized['input_ids'].shape[0] == 3
        assert tokenized['input_ids'].shape[1] <= 64
    
    def test_create_attention_mask(self) -> Any:
        """Test attention mask creation"""
        tokenizer = AdvancedTokenizer("gpt2")
        input_ids = torch.randint(0, 1000, (2, 10))
        
        attention_mask = tokenizer.create_attention_mask(input_ids)
        
        assert attention_mask.shape == input_ids.shape
        assert attention_mask.dtype == torch.long
        assert not torch.isnan(attention_mask).any()
    
    def test_decode_tokens(self) -> Any:
        """Test token decoding"""
        tokenizer = AdvancedTokenizer("gpt2")
        token_ids = torch.randint(0, 1000, (2, 5))
        
        decoded = tokenizer.decode_tokens(token_ids)
        
        assert isinstance(decoded, list)
        assert len(decoded) == 2
        assert all(isinstance(text, str) for text in decoded)


class TestAdvancedTransformerModel:
    """Test advanced transformer model"""
    
    def test_model_initialization(self) -> Any:
        """Test model initialization"""
        config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=512,
            max_seq_len=64
        )
        
        model = AdvancedTransformerModel(config)
        
        assert model.config == config
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.pos_encoding, AdvancedPositionalEncoding)
        assert len(model.transformer_layers) == config.n_layers
        assert isinstance(model.output_layer, nn.Linear)
    
    def test_model_forward_shape(self) -> Any:
        """Test model forward pass shape"""
        config = ModelConfig(
            vocab_size=500,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=32
        )
        
        model = AdvancedTransformerModel(config)
        batch_size = 3
        seq_len = 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert 'hidden_states' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, config.d_model)
        assert outputs['loss'] is None  # No labels provided
    
    def test_model_forward_with_labels(self) -> Any:
        """Test model forward pass with labels"""
        config = ModelConfig(
            vocab_size=300,
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=128,
            max_seq_len=16
        )
        
        model = AdvancedTransformerModel(config)
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert outputs['loss'] is not None
        assert outputs['loss'].item() > 0
        assert not torch.isnan(outputs['loss']).any()
    
    def test_model_gradients(self) -> Any:
        """Test model gradients"""
        config = ModelConfig(
            vocab_size=200,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=8
        )
        
        model = AdvancedTransformerModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 5), requires_grad=True)
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()
        
        assert input_ids.grad is not None
        assert not torch.isnan(input_ids.grad).any()
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_device(self) -> Any:
        """Test model device transfer"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            config = ModelConfig(
                vocab_size=100,
                d_model=32,
                n_layers=1,
                n_heads=1,
                d_ff=128,
                max_seq_len=16
            )
            
            model = AdvancedTransformerModel(config).to(device)
            input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
            
            outputs = model(input_ids)
            
            assert outputs['logits'].device == device
            assert outputs['hidden_states'].device == device
            assert not torch.isnan(outputs['logits']).any()


class TestAdvancedTrainingPipeline:
    """Test advanced training pipeline"""
    
    def test_pipeline_initialization(self) -> Any:
        """Test training pipeline initialization"""
        config = ModelConfig(
            vocab_size=100,
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=128,
            max_seq_len=16
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=16)
        
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        assert pipeline.model == model
        assert pipeline.config == config
        assert pipeline.tokenizer == tokenizer
        assert isinstance(pipeline.optimizer, optim.AdamW)
        assert isinstance(pipeline.scheduler, optim.lr_scheduler._LRScheduler)
    
    def test_train_step(self) -> Any:
        """Test training step"""
        config = ModelConfig(
            vocab_size=50,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=8,
            mixed_precision=False  # Disable for testing
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=8)
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 6)),
            'attention_mask': torch.ones(2, 6),
            'labels': torch.randint(0, config.vocab_size, (2, 6))
        }
        
        metrics = pipeline.train_step(batch)
        
        assert 'loss' in metrics
        assert 'learning_rate' in metrics
        assert metrics['loss'] > 0
        assert metrics['learning_rate'] > 0
        assert not np.isnan(metrics['loss'])
    
    def test_evaluate_step(self) -> Any:
        """Test evaluation step"""
        config = ModelConfig(
            vocab_size=30,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=8
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=8)
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 6)),
            'attention_mask': torch.ones(2, 6),
            'labels': torch.randint(0, config.vocab_size, (2, 6))
        }
        
        metrics = pipeline.evaluate_step(batch)
        
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 0
        assert not np.isnan(metrics['loss'])
        assert not np.isnan(metrics['perplexity'])
    
    def test_generate_text(self) -> Any:
        """Test text generation"""
        config = ModelConfig(
            vocab_size=100,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=16
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=16)
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        prompt = "Hello world"
        generated_text = pipeline.generate_text(prompt, max_length=10)
        
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        assert prompt in generated_text


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_training(self) -> Any:
        """Test end-to-end training pipeline"""
        config = ModelConfig(
            vocab_size=50,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=8,
            mixed_precision=False
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=8)
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        # Create simple dataset
        texts = ["Hello world", "Test sentence", "Another test"]
        tokenized_data = tokenizer.tokenize_text(texts)
        
        # Training loop
        for epoch in range(2):
            batch = {
                'input_ids': tokenized_data['input_ids'],
                'attention_mask': tokenized_data['attention_mask'],
                'labels': tokenized_data['input_ids'].clone()
            }
            
            train_metrics = pipeline.train_step(batch)
            eval_metrics = pipeline.evaluate_step(batch)
            
            assert train_metrics['loss'] > 0
            assert eval_metrics['loss'] > 0
            assert not np.isnan(train_metrics['loss'])
            assert not np.isnan(eval_metrics['loss'])
    
    def test_model_serialization(self) -> Any:
        """Test model saving and loading"""
        config = ModelConfig(
            vocab_size=100,
            d_model=32,
            n_layers=1,
            n_heads=1,
            d_ff=128,
            max_seq_len=16
        )
        
        model = AdvancedTransformerModel(config)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            temp_path = f.name
        
        try:
            # Load model
            new_model = AdvancedTransformerModel(config)
            new_model.load_state_dict(torch.load(temp_path))
            
            # Test both models produce same output
            input_ids = torch.randint(0, config.vocab_size, (2, 8))
            output1 = model(input_ids)
            output2 = new_model(input_ids)
            
            assert torch.allclose(output1['logits'], output2['logits'], atol=1e-6)
            
        finally:
            os.unlink(temp_path)
    
    def test_mixed_precision_training(self) -> Any:
        """Test mixed precision training"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
        
        config = ModelConfig(
            vocab_size=50,
            d_model=16,
            n_layers=1,
            n_heads=1,
            d_ff=64,
            max_seq_len=8,
            mixed_precision=True
        )
        
        model = AdvancedTransformerModel(config)
        tokenizer = AdvancedTokenizer("gpt2", max_length=8)
        pipeline = AdvancedTrainingPipeline(model, config, tokenizer)
        
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, 6)),
            'attention_mask': torch.ones(2, 6),
            'labels': torch.randint(0, config.vocab_size, (2, 6))
        }
        
        metrics = pipeline.train_step(batch)
        
        assert 'loss' in metrics
        assert metrics['loss'] > 0
        assert not np.isnan(metrics['loss'])


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 